#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ConvNeXt-Base and Swin-B classification on Kvasir v1 (8 classes, 4000 images).
Reproduces the MedMamba split: 2408 train / 392 val / 1200 test.

Two training modes are supported:
  --pretrained false  → train from scratch (matches MedMamba paper's setup exactly)
  --pretrained true   → fine-tune ImageNet-pretrained weights (fair comparison to Qwen)

MedMamba training settings (from paper):
  - AdamW, lr=1e-4, B1=0.9, B2=0.999, weight_decay=1e-4
  - 150 epochs, batch_size=64, early stopping
  - No data augmentation, no pre-training
  - Input: 224x224x3

Usage:
    # Match MedMamba exactly (from scratch)
    python kvasir_cnn_train.py --data_root /path/to/kvasir --model convnext_base
    python kvasir_cnn_train.py --data_root /path/to/kvasir --model swin_b

    # With ImageNet pretraining (fair comparison to Qwen)
    python kvasir_cnn_train.py --data_root /path/to/kvasir --model convnext_base --pretrained
    python kvasir_cnn_train.py --data_root /path/to/kvasir --model swin_b --pretrained
"""

import os
import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.models import (
    convnext_base, ConvNeXt_Base_Weights,
    swin_b,        Swin_B_Weights,
    densenet169,   DenseNet169_Weights
)
from PIL import Image
from tqdm.auto import tqdm


# -----------------------
# Dataset info
# -----------------------
KVASIR_CLASSES = [
    "dyed-lifted-polyps",
    "dyed-resection-margins",
    "esophagitis",
    "normal-cecum",
    "normal-pylorus",
    "normal-z-line",
    "polyps",
    "ulcerative-colitis",
]
NUM_CLASSES = 8

TRAIN_TOTAL = 2408
VAL_TOTAL   = 392
TEST_TOTAL  = 1200

TRAIN_PER_CLASS = TRAIN_TOTAL // NUM_CLASSES   # 301
VAL_PER_CLASS   = VAL_TOTAL   // NUM_CLASSES   # 49
TEST_PER_CLASS  = TEST_TOTAL  // NUM_CLASSES   # 150


# -----------------------
# Utils
# -----------------------
def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------
# Dataset
# -----------------------
class KvasirDataset(Dataset):
    """
    Two split modes, both reproducing MedMamba totals (2408 / 392 / 1200):

    split_mode="balanced"  — enforces exactly equal images per class
        train: 301 per class = 2408 total
        val:    49 per class =  392 total
        test:  150 per class = 1200 total

    split_mode="random"    — pools all 4000 images and splits randomly
        Total counts are the same (2408/392/1200) but class distribution
        within each split reflects natural randomness rather than forced balance.
    """

    def __init__(self, data_root: str, split: str = "train",
                 split_mode: str = "balanced", seed: int = 42,
                 transform=None):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        assert split_mode in ("balanced", "random"), f"Unknown split_mode: {split_mode}"
        self.split = split
        self.split_mode = split_mode
        self.data_root = Path(data_root)
        self.transform = transform

        for cls in KVASIR_CLASSES:
            if not (self.data_root / cls).is_dir():
                raise FileNotFoundError(
                    f"Expected class folder not found: {self.data_root / cls}\n"
                    f"Make sure --data_root points to the unzipped kvasir-dataset folder."
                )

        self.samples: List[Tuple[Path, int]] = []
        all_labels: List[int] = []
        rng = np.random.default_rng(seed)

        if split_mode == "balanced":
            for class_idx, cls_name in enumerate(KVASIR_CLASSES):
                cls_dir = self.data_root / cls_name
                img_paths = sorted([
                    p for p in cls_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
                ])
                if len(img_paths) < TRAIN_PER_CLASS + VAL_PER_CLASS + TEST_PER_CLASS:
                    raise ValueError(
                        f"Class '{cls_name}' has only {len(img_paths)} images, "
                        f"need at least {TRAIN_PER_CLASS + VAL_PER_CLASS + TEST_PER_CLASS}."
                    )
                shuffled = rng.permutation(len(img_paths))
                train_idx = shuffled[:TRAIN_PER_CLASS]
                val_idx   = shuffled[TRAIN_PER_CLASS:TRAIN_PER_CLASS + VAL_PER_CLASS]
                test_idx  = shuffled[TRAIN_PER_CLASS + VAL_PER_CLASS:
                                      TRAIN_PER_CLASS + VAL_PER_CLASS + TEST_PER_CLASS]
                chosen = {"train": train_idx, "val": val_idx, "test": test_idx}[split]
                for i in chosen:
                    self.samples.append((img_paths[i], class_idx))
                    all_labels.append(class_idx)

        else:  # random
            all_paths: List[Path] = []
            all_cls:   List[int]  = []
            for class_idx, cls_name in enumerate(KVASIR_CLASSES):
                cls_dir = self.data_root / cls_name
                img_paths = sorted([
                    p for p in cls_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
                ])
                all_paths.extend(img_paths)
                all_cls.extend([class_idx] * len(img_paths))

            shuffled = rng.permutation(len(all_paths))
            split_idx = {
                "train": shuffled[:TRAIN_TOTAL],
                "val":   shuffled[TRAIN_TOTAL:TRAIN_TOTAL + VAL_TOTAL],
                "test":  shuffled[TRAIN_TOTAL + VAL_TOTAL:TRAIN_TOTAL + VAL_TOTAL + TEST_TOTAL],
            }[split]
            for i in split_idx:
                self.samples.append((all_paths[i], all_cls[i]))
                all_labels.append(all_cls[i])

        self.labels = np.array(all_labels, dtype=np.int64)
        counts = np.bincount(self.labels, minlength=NUM_CLASSES)
        print(f"[Kvasir] split='{split}' ({split_mode}): "
              f"{len(self.samples)} images | per-class: {counts.tolist()}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, str(path)


# -----------------------
# Transforms
# -----------------------
def get_transforms(pretrained: bool, split: str, img_size: int = 224):
    """
    MedMamba uses no augmentation. We match that for the from-scratch run.
    For the pretrained run we still skip augmentation for fair comparison,
    but use ImageNet mean/std normalisation as the pretrained backbone expects it.
    """
    if pretrained:
        # ImageNet normalisation expected by torchvision pretrained weights
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
    else:
        # Simple [0,1] normalisation, no ImageNet stats (matches from-scratch baseline)
        mean = [0.0, 0.0, 0.0]
        std  = [1.0, 1.0, 1.0]

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


# -----------------------
# Model builder
# -----------------------
def build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    print(f"\n{'='*60}")
    print(f"Building {model_name} | pretrained={pretrained}")
    print(f"{'='*60}")

    if model_name == "convnext_base":
        if pretrained:
            weights = ConvNeXt_Base_Weights.IMAGENET1K_V1
            model = convnext_base(weights=weights)
            print(f"  Loaded ImageNet weights: {weights}")
        else:
            model = convnext_base(weights=None)
            print("  Random initialisation (no pretraining)")
        # Replace classifier head
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        print(f"  Classifier head: {in_features} → {num_classes}")

    elif model_name == "swin_b":
        if pretrained:
            weights = Swin_B_Weights.IMAGENET1K_V1
            model = swin_b(weights=weights)
            print(f"  Loaded ImageNet weights: {weights}")
        else:
            model = swin_b(weights=None)
            print("  Random initialisation (no pretraining)")
        # Replace classifier head
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        print(f"  Classifier head: {in_features} → {num_classes}")
    
    elif model_name == "densenet169":
        if pretrained:
            weights = DenseNet169_Weights.IMAGENET1K_V1
            model = densenet169(weights=weights)
            print(f"  Loaded ImageNet weights: {weights}")
        else:
            model = densenet169(weights=None)
            print("  Random initialisation (no pretraining)")
        # Replace classifier head
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        print(f"  Classifier head: {in_features} → {num_classes}")

    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'convnext_base' or 'swin_b'.")

    total  = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")
    return model


# -----------------------
# Train / Eval
# -----------------------
def train_one_epoch(model, dl, optimizer, class_weights, device, epoch, total_epochs,
                    use_amp=True):
    model.train()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = use_amp and device.startswith("cuda") and dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    tot_loss = tot_correct = tot_seen = 0
    pbar = tqdm(dl, desc=f"[Epoch {epoch:03d}/{total_epochs:03d}] train", leave=False)

    for imgs, y, _ in pbar:
        imgs = imgs.to(device, non_blocking=True)
        y    = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda",
                                 enabled=use_amp and device.startswith("cuda"),
                                 dtype=dtype):
            logits = model(imgs)
            loss   = F.cross_entropy(logits, y, weight=class_weights)

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        bs = y.size(0)
        tot_loss    += loss.item() * bs
        tot_correct += (logits.argmax(1) == y).sum().item()
        tot_seen    += bs
        pbar.set_postfix(loss=f"{tot_loss/tot_seen:.4f}",
                         acc=f"{tot_correct/tot_seen:.4f}")

    return {"train_loss": tot_loss / tot_seen, "train_acc": tot_correct / tot_seen}


@torch.no_grad()
def evaluate(model, dl, device, desc, num_classes):
    model.eval()
    all_y, all_p, all_probs = [], [], []
    tot_loss = tot_seen = 0

    pbar = tqdm(dl, desc=f"[{desc}]", leave=False)
    for imgs, y, _ in pbar:
        imgs = imgs.to(device, non_blocking=True)
        y    = y.to(device, non_blocking=True)

        logits = model(imgs)
        loss   = F.cross_entropy(logits, y)

        bs = y.size(0)
        tot_loss += loss.item() * bs
        tot_seen += bs
        all_y.append(y.cpu())
        all_p.append(logits.argmax(1).cpu())
        all_probs.append(F.softmax(logits, 1).cpu())
        pbar.set_postfix(loss=f"{tot_loss/tot_seen:.4f}")

    y_true  = torch.cat(all_y)
    y_pred  = torch.cat(all_p)
    y_probs = torch.cat(all_probs)

    support = y_true.numel()
    acc = (y_true == y_pred).sum().item() / support

    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    precisions, sensitivities, specificities, f1s = [], [], [], []
    total_cm = cm.sum().item()
    for c in range(num_classes):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        tn = total_cm - tp - fp - fn
        prec = tp / max(1, tp + fp)
        sens = tp / max(1, tp + fn)
        spec = tn / max(1, tn + fp)
        f1   = 2 * prec * sens / max(1e-9, prec + sens)
        precisions.append(prec)
        sensitivities.append(sens)
        specificities.append(spec)
        f1s.append(f1)

    macro_prec = float(np.mean(precisions))
    macro_sens = float(np.mean(sensitivities))
    macro_spec = float(np.mean(specificities))
    macro_f1   = float(np.mean(f1s))

    try:
        from sklearn.metrics import roc_auc_score
        y_true_np  = y_true.numpy()
        y_probs_np = y_probs.numpy()
        per_class_auc = []
        for c in range(num_classes):
            yb = (y_true_np == c).astype(int)
            if yb.sum() > 0 and (1 - yb).sum() > 0:
                per_class_auc.append(roc_auc_score(yb, y_probs_np[:, c]))
            else:
                per_class_auc.append(float("nan"))
        valid_aucs = [a for a in per_class_auc if not np.isnan(a)]
        macro_auc  = float(np.mean(valid_aucs)) if valid_aucs else float("nan")
        try:
            auc_ovr = roc_auc_score(y_true_np, y_probs_np,
                                    multi_class="ovr", average="weighted")
        except Exception:
            auc_ovr = float("nan")
    except ImportError:
        per_class_auc = [float("nan")] * num_classes
        macro_auc = auc_ovr = float("nan")

    return {
        "accuracy":          acc,
        "loss":              tot_loss / support,
        "support":           support,
        "macro_precision":   macro_prec,
        "macro_sensitivity": macro_sens,
        "macro_specificity": macro_spec,
        "macro_recall":      macro_sens,
        "macro_f1":          macro_f1,
        "macro_auc":         macro_auc,
        "auc_ovr_weighted":  auc_ovr,
        "per_class_precision":   precisions,
        "per_class_sensitivity": sensitivities,
        "per_class_specificity": specificities,
        "per_class_f1":          f1s,
        "per_class_auc":         per_class_auc,
    }, cm


# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(
        description="ConvNeXt-Base / Swin-B on Kvasir v1 (MedMamba comparison)")

    # Data
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to unzipped kvasir-dataset folder")
    parser.add_argument("--out_dir", type=str, default="./kvasir_cnn_runs2",
                        help="Output directory")
    parser.add_argument("--split_mode", type=str, default="random",
                        choices=["balanced", "random"],
                        help="balanced: exact equal quota per class; "
                             "random: pool and split randomly at same totals")
    parser.add_argument("--train_fraction", type=float, default=1.0,
                        help="Fraction of training data to use (stratified)")

    # Model
    parser.add_argument("--model", type=str, default="convnext_base",
                        choices=["convnext_base", "swin_b", "densenet169"],
                        help="Model architecture")
    parser.add_argument("--pretrained", action="store_true",
                        help="Use ImageNet pretrained weights (default: from scratch, "
                             "matching MedMamba paper)")

    # Training — defaults match MedMamba paper exactly
    parser.add_argument("--epochs", type=int, default=15,
                        help="Max epochs (MedMamba used 150 with early stopping)")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial LR (MedMamba: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="AdamW weight decay (MedMamba: 1e-4)")
    parser.add_argument("--early_stop_patience", type=int, default=20,
                        help="Early stopping patience in epochs (0 = disabled)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--img_size", type=int, default=224,
                        help="Square input resolution (Resize to img_size x img_size)")

    args = parser.parse_args()

    if not (0.0 < args.train_fraction <= 1.0):
        raise ValueError(f"--train_fraction must be in (0, 1], got {args.train_fraction}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pretrained_str = "imagenet_pretrained" if args.pretrained else "from_scratch"

    print(f"\n{'='*70}")
    print(f"Training {args.model} ({pretrained_str}) on Kvasir v1")
    print(f"{'='*70}")
    print(f"Data root:       {args.data_root}")
    print(f"Output:          {out_dir}")
    print(f"Split:           {TRAIN_TOTAL} train / {VAL_TOTAL} val / {TEST_TOTAL} test "
          f"({args.split_mode})")
    print(f"Device:          {device}")
    print(f"Pretrained:      {args.pretrained} "
          f"{'(matches MedMamba)' if not args.pretrained else '(ImageNet weights)'}")
    print(f"Train fraction:  {args.train_fraction*100:.1f}%")

    # Transforms
    tf_train = get_transforms(args.pretrained, "train", args.img_size)
    tf_eval  = get_transforms(args.pretrained, "val", args.img_size)

    # Datasets
    ds_train = KvasirDataset(args.data_root, split="train",
                              split_mode=args.split_mode, seed=args.seed,
                              transform=tf_train)
    ds_val   = KvasirDataset(args.data_root, split="val",
                              split_mode=args.split_mode, seed=args.seed,
                              transform=tf_eval)
    ds_test  = KvasirDataset(args.data_root, split="test",
                              split_mode=args.split_mode, seed=args.seed,
                              transform=tf_eval)

    # Stratified subsample training set if requested
    if args.train_fraction < 1.0:
        n_total = len(ds_train)
        rng = np.random.default_rng(args.seed)
        indices = []
        for c in range(NUM_CLASSES):
            class_idx = np.where(ds_train.labels == c)[0]
            n_keep = max(1, int(len(class_idx) * args.train_fraction))
            chosen = rng.choice(class_idx, size=n_keep, replace=False).tolist()
            indices.extend(chosen)
        indices = sorted(indices)
        ds_train_sub = Subset(ds_train, indices)
        ds_train_sub.labels = np.array([ds_train.labels[i] for i in indices])
        ds_train = ds_train_sub
        print(f"  → Subsampled to {len(indices)}/{n_total} training samples "
              f"({args.train_fraction*100:.1f}%, stratified)")

    # Class weights
    class_counts = np.bincount(ds_train.labels, minlength=NUM_CLASSES)
    if args.use_class_weights:
        total = class_counts.sum()
        weights = [total / (NUM_CLASSES * max(1, int(c))) for c in class_counts]
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f"\nClass weights: {class_weights.cpu().tolist()}")
    else:
        class_weights = None
        print(f"\nClass counts (train): {class_counts.tolist()}")

    # DataLoaders
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    # Model
    model = build_model(args.model, NUM_CLASSES, args.pretrained).to(device)

    # Optimizer — AdamW matching MedMamba paper
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    # Training loop
    best_val_score  = -1.0
    best_state      = None
    history         = []
    early_stop_ctr  = 0

    print(f"\nStarting training for up to {args.epochs} epochs "
          f"(early stop patience={args.early_stop_patience})...")

    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(
            model, dl_train, optimizer, class_weights,
            device=device, epoch=epoch, total_epochs=args.epochs,
        )
        print(f"[Epoch {epoch:03d}] train_loss={stats['train_loss']:.4f}  "
              f"train_acc={stats['train_acc']:.4f}")

        val_metrics, val_cm = evaluate(
            model, dl_val, device=device, desc="val", num_classes=NUM_CLASSES,
        )
        print(f"[Epoch {epoch:03d}] VAL  "
              f"acc={val_metrics['accuracy']:.4f}  "
              f"f1={val_metrics['macro_f1']:.4f}  "
              f"auc={val_metrics['macro_auc']:.4f}")

        history.append({
            "epoch":        epoch,
            "train_loss":   stats["train_loss"],
            "train_acc":    stats["train_acc"],
            "val_loss":     val_metrics["loss"],
            "val_acc":      val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_macro_auc":val_metrics["macro_auc"],
        })

        if val_metrics["accuracy"] > best_val_score:
            best_val_score = val_metrics["accuracy"]
            best_state = {
                "model": {k: v.cpu() for k, v in model.state_dict().items()},
                "val_metrics": val_metrics,
                "val_cm": val_cm.clone(),
                "epoch": epoch,
            }
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
            }, out_dir / "best_model.pt")
            early_stop_ctr = 0
        else:
            early_stop_ctr += 1
            if args.early_stop_patience > 0 and early_stop_ctr >= args.early_stop_patience:
                print(f"\n[Early stop] No improvement for {args.early_stop_patience} epochs. "
                      f"Stopping at epoch {epoch}.")
                break

    # Load best and test
    if best_state is not None:
        model.load_state_dict(best_state["model"])

    test_metrics, test_cm = evaluate(
        model, dl_test, device=device, desc="test", num_classes=NUM_CLASSES,
    )

    p   = test_metrics['macro_precision']   * 100
    se  = test_metrics['macro_sensitivity'] * 100
    sp  = test_metrics['macro_specificity'] * 100
    f1  = test_metrics['macro_f1']          * 100
    oa  = test_metrics['accuracy']          * 100
    auc = test_metrics['macro_auc']

    print(f"\n{'='*70}")
    print(f"TEST RESULTS — {args.model} ({pretrained_str})")
    print(f"{'='*70}")
    print(f"{'P(%)':>8} {'Se(%)':>8} {'Sp(%)':>8} {'F1(%)':>8} {'OA(%)':>8} {'AUC':>8}")
    print(f"{'-'*56}")
    print(f"{p:>8.1f} {se:>8.1f} {sp:>8.1f} {f1:>8.1f} {oa:>8.1f} {auc:>8.3f}")
    print(f"{'='*70}")

    # Print confusion matrix
    short_names = [c.replace("dyed-", "d-").replace("normal-", "n-")
                   .replace("ulcerative-colitis", "uc") for c in KVASIR_CLASSES]
    col_w = 6
    print(f"\nConfusion Matrix (rows=true, cols=predicted):")
    print(f"{'':>22}", end="")
    for sn in short_names:
        print(f"{sn:>{col_w}}", end="")
    print()
    print(" " * 22 + "-" * (col_w * NUM_CLASSES))
    cm_np = test_cm.numpy()
    for i, sn in enumerate(short_names):
        print(f"{KVASIR_CLASSES[i]:>22}", end="")
        for j in range(NUM_CLASSES):
            print(f"{cm_np[i, j]:>{col_w}}", end="")
        print()

    # Save confusion matrix heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))
        cm_float  = test_cm.numpy().astype(float)
        row_sums  = cm_float.sum(axis=1, keepdims=True).clip(min=1)
        cm_pct    = cm_float / row_sums * 100

        im = ax.imshow(cm_pct, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Row %", fontsize=11)

        tick_labels = [c.replace("-", "\n") for c in KVASIR_CLASSES]
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(tick_labels, fontsize=8, ha="center")
        ax.set_yticklabels(tick_labels, fontsize=8)

        thresh = 50.0
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                color = "white" if cm_pct[i, j] > thresh else "black"
                ax.text(j, i,
                        f"{int(cm_float[i, j])}\n({cm_pct[i, j]:.1f}%)",
                        ha="center", va="center", fontsize=7, color=color)

        ax.set_xlabel("Predicted label", fontsize=12)
        ax.set_ylabel("True label", fontsize=12)
        ax.set_title(
            f"Kvasir v1 — {args.model} ({pretrained_str})\n"
            f"OA={oa:.1f}%  F1={f1:.1f}%  AUC={auc:.3f}", fontsize=12)

        plt.tight_layout()
        cm_path = out_dir / "confusion_matrix.png"
        fig.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n✓ Saved confusion matrix → {cm_path}")
    except ImportError:
        print("Warning: matplotlib not available — skipping confusion matrix plot")

    # Save checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_name": args.model,
        "pretrained": args.pretrained,
        "num_classes": NUM_CLASSES,
        "classes": KVASIR_CLASSES,
        "train_fraction": args.train_fraction,
        "split_mode": args.split_mode,
        "best_val": {
            "epoch": best_state["epoch"] if best_state else None,
            "val_metrics": best_state["val_metrics"] if best_state else None,
            "val_confusion_matrix": best_state["val_cm"].tolist() if best_state else None,
        },
        "test_metrics": test_metrics,
        "test_confusion_matrix": test_cm.tolist(),
    }, out_dir / "checkpoint.pt")

    # Training history text file
    history_path = out_dir / "training_history.txt"
    with open(history_path, "w") as f:
        f.write(f"Kvasir v1 — {args.model} ({pretrained_str}) Training History\n")
        f.write(f"{'='*80}\n\n")
        f.write("Configuration:\n")
        f.write(f"  Model:          {args.model}\n")
        f.write(f"  Pretrained:     {args.pretrained} ({pretrained_str})\n")
        f.write(f"  Data root:      {args.data_root}\n")
        f.write(f"  Split:          {TRAIN_TOTAL} / {VAL_TOTAL} / {TEST_TOTAL} ({args.split_mode})\n")
        f.write(f"  Train fraction: {args.train_fraction*100:.1f}%\n")
        f.write(f"  Batch size:     {args.batch_size}\n")
        f.write(f"  LR:             {args.lr}\n")
        f.write(f"  Weight decay:   {args.weight_decay}\n")
        f.write(f"  Max epochs:     {args.epochs}\n")
        f.write(f"  Early stop:     {args.early_stop_patience}\n")
        f.write(f"  Class weights:  {args.use_class_weights}\n\n")

        f.write("Training History:\n")
        f.write(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} "
                f"{'Val Loss':>12} {'Val Acc':>10} {'Val F1':>10} {'Val AUC':>10}\n")
        f.write(f"{'-'*90}\n")
        for h in history:
            f.write(f"{h['epoch']:>6} {h['train_loss']:>12.4f} {h['train_acc']:>10.4f} "
                    f"{h['val_loss']:>12.4f} {h['val_acc']:>10.4f} "
                    f"{h['val_macro_f1']:>10.4f} {h['val_macro_auc']:>10.4f}\n")

        f.write(f"\n{'='*90}\n")
        f.write(f"Best Val Accuracy: {best_val_score:.4f} "
                f"(Epoch {best_state['epoch'] if best_state else 'N/A'})\n\n")
        f.write("Test Results:\n")
        f.write(f"  {'P(%)':>8} {'Se(%)':>8} {'Sp(%)':>8} {'F1(%)':>8} "
                f"{'OA(%)':>8} {'AUC':>8}\n")
        f.write(f"  {p:>8.1f} {se:>8.1f} {sp:>8.1f} {f1:>8.1f} "
                f"{oa:>8.1f} {auc:>8.3f}\n\n")

        header = f"{'Class':<35} {'Prec':>8} {'Sens':>8} {'Spec':>8} {'F1':>8} {'AUC':>8}\n"
        f.write("Per-class Metrics:\n")
        f.write(header)
        f.write("-" * 75 + "\n")
        for i, cls in enumerate(KVASIR_CLASSES):
            auc_str = f"{test_metrics['per_class_auc'][i]:.4f}" \
                      if not np.isnan(test_metrics['per_class_auc'][i]) else "  N/A"
            f.write(f"{cls:<35} "
                    f"{test_metrics['per_class_precision'][i]:>8.4f} "
                    f"{test_metrics['per_class_sensitivity'][i]:>8.4f} "
                    f"{test_metrics['per_class_specificity'][i]:>8.4f} "
                    f"{test_metrics['per_class_f1'][i]:>8.4f} "
                    f"{auc_str:>8}\n")

    # JSON metrics
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "dataset": "kvasir_v1",
            "model": args.model,
            "pretrained": args.pretrained,
            "pretrained_str": pretrained_str,
            "num_classes": NUM_CLASSES,
            "class_names": KVASIR_CLASSES,
            "split": {"train": TRAIN_TOTAL, "val": VAL_TOTAL, "test": TEST_TOTAL},
            "split_mode": args.split_mode,
            "train_fraction": args.train_fraction,
            "test_metrics": test_metrics,
            "test_confusion_matrix": test_cm.tolist(),
            "best_val": {
                "epoch": best_state["epoch"] if best_state else None,
                "val_metrics": best_state["val_metrics"] if best_state else None,
                "val_confusion_matrix": best_state["val_cm"].tolist() if best_state else None,
            },
            "training_config": {
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "img_size": args.img_size,
                "batch_size": args.batch_size,
                "max_epochs": args.epochs,
                "early_stop_patience": args.early_stop_patience,
                "use_class_weights": args.use_class_weights,
                "train_fraction": args.train_fraction,
                "split_mode": args.split_mode,
            },
        }, f, indent=2)

    print(f"\n✓ Saved checkpoint       → {out_dir / 'checkpoint.pt'}")
    print(f"✓ Saved best model       → {out_dir / 'best_model.pt'}")
    print(f"✓ Saved training history → {history_path}")
    print(f"✓ Saved metrics          → {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()