#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ConvNeXt-Base and Swin-B classification on PAD-UFES-20 (6 classes, 2298 images).
Split is done at the PATIENT level to prevent data leakage.

Note: BOD (Bowen's disease) has 0 images in this dataset → 6 classes used.

Two training modes:
  default      → train from scratch (matches MedMamba paper exactly)
  --pretrained → fine-tune ImageNet pretrained weights

MedMamba training settings:
  - AdamW, lr=1e-4, B1=0.9, B2=0.999, weight_decay=1e-4
  - 150 epochs, batch_size=64, early stopping
  - No data augmentation, no pre-training

Usage:
    # Match MedMamba exactly (from scratch)
    python pad_ufes_cnn_train.py --data_root /scratch/cui0011/pad_ufes --model convnext_base
    python pad_ufes_cnn_train.py --data_root /scratch/cui0011/pad_ufes --model swin_b

    # With ImageNet pretraining
    python pad_ufes_cnn_train.py --data_root /scratch/cui0011/pad_ufes --model convnext_base --pretrained
    python pad_ufes_cnn_train.py --data_root /scratch/cui0011/pad_ufes --model swin_b --pretrained
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple
from collections import Counter

import numpy as np
import pandas as pd
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
CLASSES     = ["BCC", "SCC", "ACK", "SEK", "MEL", "NEV"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

TRAIN_TARGET = 1384
VAL_TARGET   = 227
TEST_TARGET  = 687
TOTAL        = TRAIN_TARGET + VAL_TARGET + TEST_TARGET  # 2298

TRAIN_FRAC = TRAIN_TARGET / TOTAL   # ~0.602
VAL_FRAC   = VAL_TARGET   / TOTAL   # ~0.099


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
# Patient-level split
# -----------------------
def make_patient_split(df: pd.DataFrame, seed: int = 42):
    """
    Split patients into train/val/test so that:
      - All images of a patient appear in exactly one split
      - Class distribution is approximately preserved (stratified by primary class)
      - Image totals are close to MedMamba's 1384/227/687

    Returns three sets of patient_ids: train_pids, val_pids, test_pids
    """
    rng = np.random.default_rng(seed)

    primary_class = (
        df.groupby("patient_id")["diagnostic"]
        .agg(lambda x: x.value_counts().index[0])
        .reset_index()
        .rename(columns={"diagnostic": "primary_class"})
    )

    train_pids, val_pids, test_pids = [], [], []

    for cls in CLASSES:
        pids = primary_class[primary_class["primary_class"] == cls]["patient_id"].values
        pids = rng.permutation(pids)
        n = len(pids)

        n_train = round(n * TRAIN_FRAC)
        n_val   = round(n * VAL_FRAC)
        n_train = max(1, n_train)
        n_val   = max(1, n_val)
        n_test  = n - n_train - n_val
        if n_test < 1:
            n_train = max(1, n - 2)
            n_val   = 1
            n_test  = max(1, n - n_train - n_val)

        train_pids.extend(pids[:n_train])
        val_pids.extend(pids[n_train:n_train + n_val])
        test_pids.extend(pids[n_train + n_val:n_train + n_val + n_test])

    return set(train_pids), set(val_pids), set(test_pids)


def print_split_stats(df, train_pids, val_pids, test_pids):
    def split_df(pids):
        return df[df["patient_id"].isin(pids)]

    splits = {"train": split_df(train_pids),
              "val":   split_df(val_pids),
              "test":  split_df(test_pids)}

    print(f"\n{'='*70}")
    print(f"SPLIT SUMMARY  (target: {TRAIN_TARGET}/{VAL_TARGET}/{TEST_TARGET})")
    print(f"{'='*70}")
    print(f"{'':>8} {'train':>10} {'val':>10} {'test':>10} {'total':>10}")
    print(f"{'-'*50}")
    print(f"{'patients':>8} {len(train_pids):>10} {len(val_pids):>10} "
          f"{len(test_pids):>10} "
          f"{len(train_pids)+len(val_pids)+len(test_pids):>10}")
    print(f"{'images':>8} {len(splits['train']):>10} {len(splits['val']):>10} "
          f"{len(splits['test']):>10} "
          f"{sum(len(splits[s]) for s in splits):>10}")
    print(f"\nPer-class image counts:")
    print(f"{'Class':>6} {'train':>8} {'val':>8} {'test':>8}")
    print(f"{'-'*34}")
    for cls in CLASSES:
        counts = {s: (splits[s]["diagnostic"] == cls).sum() for s in splits}
        print(f"{cls:>6} {counts['train']:>8} {counts['val']:>8} {counts['test']:>8}")


# -----------------------
# Dataset
# -----------------------
class PADUFESDataset(Dataset):
    """
    PAD-UFES-20 dataset.

    Expected structure:
        data_root/
            metadata.csv
            images/
                imgs_part_1/   ← PNGs named as img_id column
                imgs_part_2/
                ...
    """

    def __init__(self, data_root: str, split: str = "train",
                 seed: int = 42, transform=None):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.split      = split
        self.data_root  = Path(data_root)
        self.img_dir    = self.data_root / "images"
        self.transform  = transform

        if not self.img_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory not found: {self.img_dir}\n"
                f"Expected images under {self.img_dir}/imgs_part_*/"
            )

        # Build img_id → full path lookup across all imgs_part_* subfolders
        self._img_lookup: dict = {}
        part_dirs = sorted(self.img_dir.glob("imgs_part_*"))
        if not part_dirs:
            part_dirs = [self.img_dir]
        for part_dir in part_dirs:
            for p in part_dir.iterdir():
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    self._img_lookup[p.name] = p
        print(f"[PAD-UFES] Found {len(self._img_lookup)} images across "
              f"{len(part_dirs)} folder(s): {[d.name for d in part_dirs]}")

        df = pd.read_csv(self.data_root / "metadata.csv")
        df = df[df["diagnostic"].isin(CLASSES)].reset_index(drop=True)

        train_pids, val_pids, test_pids = make_patient_split(df, seed=seed)
        split_pids = {"train": train_pids, "val": val_pids, "test": test_pids}[split]

        if split == "train":
            print_split_stats(df, train_pids, val_pids, test_pids)

        subset = df[df["patient_id"].isin(split_pids)].reset_index(drop=True)

        self.samples: List[Tuple[Path, int]] = []
        missing = []
        for _, row in subset.iterrows():
            img_id   = row["img_id"]
            img_path = self._img_lookup.get(img_id)
            if img_path is None:
                missing.append(img_id)
                continue
            self.samples.append((img_path, CLASS_TO_IDX[row["diagnostic"]]))

        if missing:
            print(f"[WARNING] {len(missing)} image files not found — skipped")

        self.labels = np.array([s[1] for s in self.samples], dtype=np.int64)
        counts = np.bincount(self.labels, minlength=NUM_CLASSES)
        print(f"[PAD-UFES] split='{split}': {len(self.samples)} images | "
              f"per-class: { {CLASSES[i]: int(counts[i]) for i in range(NUM_CLASSES)} }")

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
def get_transforms(pretrained: bool, img_size: int = 224):
    if pretrained:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
    else:
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
            model   = convnext_base(weights=weights)
            print(f"  Loaded ImageNet weights: {weights}")
        else:
            model = convnext_base(weights=None)
            print("  Random initialisation (no pretraining)")
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        print(f"  Classifier head: {in_features} → {num_classes}")

    elif model_name == "swin_b":
        if pretrained:
            weights = Swin_B_Weights.IMAGENET1K_V1
            model   = swin_b(weights=weights)
            print(f"  Loaded ImageNet weights: {weights}")
        else:
            model = swin_b(weights=None)
            print("  Random initialisation (no pretraining)")
        in_features = model.head.in_features
        model.head  = nn.Linear(in_features, num_classes)
        print(f"  Classifier head: {in_features} → {num_classes}")

    elif model_name =="densenet169":
        if pretrained:
            weights = DenseNet169_Weights.IMAGENET1K_V1
            model   = densenet169(weights=weights)
            print(f"  Loaded ImageNet weights: {weights}")
        else:
            model = densenet169(weights=None)
            print("  Random initialisation (no pretraining)")
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        print(f"  Classifier head: {in_features} → {num_classes}")

    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'convnext_base' or 'swin_b'.")

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Params: {total/1e6:.1f}M total, {trainable/1e6:.1f}M trainable")
    return model


# -----------------------
# Train / Eval
# -----------------------
def train_one_epoch(model, dl, optimizer, class_weights, device, epoch,
                    total_epochs, use_amp=True):
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
        valid     = [a for a in per_class_auc if not np.isnan(a)]
        macro_auc = float(np.mean(valid)) if valid else float("nan")
        try:
            auc_ovr = roc_auc_score(y_true_np, y_probs_np,
                                    multi_class="ovr", average="weighted")
        except Exception:
            auc_ovr = float("nan")
    except ImportError:
        per_class_auc = [float("nan")] * num_classes
        macro_auc = auc_ovr = float("nan")

    return {
        "accuracy":              acc,
        "loss":                  tot_loss / support,
        "support":               support,
        "macro_precision":       macro_prec,
        "macro_sensitivity":     macro_sens,
        "macro_specificity":     macro_spec,
        "macro_recall":          macro_sens,
        "macro_f1":              macro_f1,
        "macro_auc":             macro_auc,
        "auc_ovr_weighted":      auc_ovr,
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
        description="ConvNeXt-Base / Swin-B on PAD-UFES-20 (MedMamba comparison)")

    # Data
    parser.add_argument("--data_root", type=str,
                        default="/scratch/cui0011/pad_ufes",
                        help="Root folder containing metadata.csv and images/")
    parser.add_argument("--out_dir", type=str, default="./pad_ufes_cnn_runs2")
    parser.add_argument("--train_fraction", type=float, default=1.0,
                        help="Fraction of training data to use (stratified)")

    # Model
    parser.add_argument("--model", type=str, default="convnext_base",
                        choices=["convnext_base", "swin_b", "densenet169"])
    parser.add_argument("--pretrained", action="store_true",
                        help="Use ImageNet pretrained weights "
                             "(default: from scratch, matching MedMamba)")

    # Training — defaults match MedMamba paper
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--early_stop_patience", type=int, default=20)
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
    print(f"Training {args.model} ({pretrained_str}) on PAD-UFES-20")
    print(f"{'='*70}")
    print(f"Data root:       {args.data_root}")
    print(f"Output:          {out_dir}")
    print(f"Classes:         {NUM_CLASSES} {CLASSES}")
    print(f"Split target:    {TRAIN_TARGET} / {VAL_TARGET} / {TEST_TARGET}")
    print(f"Split strategy:  patient-level stratified")
    print(f"Device:          {device}")
    print(f"Pretrained:      {args.pretrained} "
          f"{'(matches MedMamba)' if not args.pretrained else '(ImageNet weights)'}")
    print(f"Train fraction:  {args.train_fraction*100:.1f}%")

    tf_train = get_transforms(args.pretrained, args.img_size)
    tf_eval  = get_transforms(args.pretrained, args.img_size)

    # Datasets
    ds_train = PADUFESDataset(args.data_root, split="train",
                               seed=args.seed, transform=tf_train)
    ds_val   = PADUFESDataset(args.data_root, split="val",
                               seed=args.seed, transform=tf_eval)
    ds_test  = PADUFESDataset(args.data_root, split="test",
                               seed=args.seed, transform=tf_eval)

    # Stratified subsample of training set if requested
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
        ds_sub = Subset(ds_train, indices)
        ds_sub.labels = np.array([ds_train.labels[i] for i in indices])
        ds_train = ds_sub
        print(f"  → Subsampled to {len(indices)}/{n_total} training samples "
              f"({args.train_fraction*100:.1f}%, stratified)")

    # Class weights
    class_counts = np.bincount(ds_train.labels, minlength=NUM_CLASSES)
    if args.use_class_weights:
        total   = class_counts.sum()
        weights = [total / (NUM_CLASSES * max(1, int(c))) for c in class_counts]
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f"\nClass weights: {class_weights.cpu().tolist()}")
    else:
        class_weights = None
        print(f"\nClass counts (train): "
              f"{ {CLASSES[i]: int(class_counts[i]) for i in range(NUM_CLASSES)} }")

    # DataLoaders
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    # Model
    model = build_model(args.model, NUM_CLASSES, args.pretrained).to(device)

    # Optimizer — AdamW matching MedMamba
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    # Training loop
    best_val_score = -1.0
    best_state     = None
    history        = []
    early_stop_ctr = 0

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
            "epoch":         epoch,
            "train_loss":    stats["train_loss"],
            "train_acc":     stats["train_acc"],
            "val_loss":      val_metrics["loss"],
            "val_acc":       val_metrics["accuracy"],
            "val_macro_f1":  val_metrics["macro_f1"],
            "val_macro_auc": val_metrics["macro_auc"],
        })

        if val_metrics["accuracy"] > best_val_score:
            best_val_score = val_metrics["accuracy"]
            best_state = {
                "model":       {k: v.cpu() for k, v in model.state_dict().items()},
                "val_metrics": val_metrics,
                "val_cm":      val_cm.clone(),
                "epoch":       epoch,
            }
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch":            epoch,
                "val_metrics":      val_metrics,
            }, out_dir / "best_model.pt")
            early_stop_ctr = 0
        else:
            early_stop_ctr += 1
            if args.early_stop_patience > 0 and early_stop_ctr >= args.early_stop_patience:
                print(f"\n[Early stop] No improvement for {args.early_stop_patience} "
                      f"epochs. Stopping at epoch {epoch}.")
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
    print(f"TEST RESULTS — {args.model} ({pretrained_str}) on PAD-UFES-20")
    print(f"{'='*70}")
    print(f"{'P(%)':>8} {'Se(%)':>8} {'Sp(%)':>8} {'F1(%)':>8} {'OA(%)':>8} {'AUC':>8}")
    print(f"{'-'*56}")
    print(f"{p:>8.1f} {se:>8.1f} {sp:>8.1f} {f1:>8.1f} {oa:>8.1f} {auc:>8.3f}")
    print(f"{'='*70}")

    # Confusion matrix terminal print
    print(f"\nConfusion Matrix (rows=true, cols=predicted):")
    col_w = 7
    print(f"{'':>6}", end="")
    for cls in CLASSES:
        print(f"{cls:>{col_w}}", end="")
    print()
    print(" " * 6 + "-" * (col_w * NUM_CLASSES))
    cm_np = test_cm.numpy()
    for i, cls in enumerate(CLASSES):
        print(f"{cls:>6}", end="")
        for j in range(NUM_CLASSES):
            print(f"{cm_np[i, j]:>{col_w}}", end="")
        print()

    # Save confusion matrix heatmap
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 7))
        cm_float = cm_np.astype(float)
        row_sums = cm_float.sum(axis=1, keepdims=True).clip(min=1)
        cm_pct   = cm_float / row_sums * 100

        im = ax.imshow(cm_pct, interpolation="nearest", cmap="Blues",
                       vmin=0, vmax=100)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Row %", fontsize=11)

        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(CLASSES, fontsize=10)
        ax.set_yticklabels(CLASSES, fontsize=10)

        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                color = "white" if cm_pct[i, j] > 50 else "black"
                ax.text(j, i,
                        f"{int(cm_float[i, j])}\n({cm_pct[i, j]:.1f}%)",
                        ha="center", va="center", fontsize=8, color=color)

        ax.set_xlabel("Predicted label", fontsize=12)
        ax.set_ylabel("True label", fontsize=12)
        ax.set_title(
            f"PAD-UFES-20 — {args.model} ({pretrained_str})\n"
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
        "model_state_dict":      model.state_dict(),
        "model_name":            args.model,
        "pretrained":            args.pretrained,
        "num_classes":           NUM_CLASSES,
        "classes":               CLASSES,
        "train_fraction":        args.train_fraction,
        "best_val": {
            "epoch":               best_state["epoch"] if best_state else None,
            "val_metrics":         best_state["val_metrics"] if best_state else None,
            "val_confusion_matrix":best_state["val_cm"].tolist() if best_state else None,
        },
        "test_metrics":          test_metrics,
        "test_confusion_matrix": test_cm.tolist(),
    }, out_dir / "checkpoint.pt")

    # Training history text file
    history_path = out_dir / "training_history.txt"
    with open(history_path, "w") as f:
        f.write(f"PAD-UFES-20 — {args.model} ({pretrained_str}) Training History\n")
        f.write(f"{'='*80}\n\n")
        f.write("Configuration:\n")
        f.write(f"  Model:          {args.model}\n")
        f.write(f"  Pretrained:     {args.pretrained} ({pretrained_str})\n")
        f.write(f"  Classes:        {CLASSES}\n")
        f.write(f"  Split:          patient-level stratified\n")
        f.write(f"  Split target:   {TRAIN_TARGET} / {VAL_TARGET} / {TEST_TARGET}\n")
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

        f.write("Per-class Metrics:\n")
        f.write(f"{'Class':<8} {'Prec':>8} {'Sens':>8} {'Spec':>8} "
                f"{'F1':>8} {'AUC':>8}\n")
        f.write("-" * 50 + "\n")
        for i, cls in enumerate(CLASSES):
            auc_str = f"{test_metrics['per_class_auc'][i]:.4f}" \
                      if not np.isnan(test_metrics['per_class_auc'][i]) else "  N/A"
            f.write(f"{cls:<8} "
                    f"{test_metrics['per_class_precision'][i]:>8.4f} "
                    f"{test_metrics['per_class_sensitivity'][i]:>8.4f} "
                    f"{test_metrics['per_class_specificity'][i]:>8.4f} "
                    f"{test_metrics['per_class_f1'][i]:>8.4f} "
                    f"{auc_str:>8}\n")

    # JSON metrics
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "dataset":        "pad_ufes_20",
            "model":          args.model,
            "pretrained":     args.pretrained,
            "pretrained_str": pretrained_str,
            "num_classes":    NUM_CLASSES,
            "class_names":    CLASSES,
            "split_strategy": "patient_level_stratified",
            "split_target":   {"train": TRAIN_TARGET, "val": VAL_TARGET,
                               "test": TEST_TARGET},
            "train_fraction": args.train_fraction,
            "test_metrics":   test_metrics,
            "test_confusion_matrix": test_cm.tolist(),
            "best_val": {
                "epoch":               best_state["epoch"] if best_state else None,
                "val_metrics":         best_state["val_metrics"] if best_state else None,
                "val_confusion_matrix":best_state["val_cm"].tolist() if best_state else None,
            },
            "training_config": {
                "lr":                   args.lr,
                "weight_decay":         args.weight_decay,
                "img_size":             args.img_size,
                "batch_size":           args.batch_size,
                "max_epochs":           args.epochs,
                "early_stop_patience":  args.early_stop_patience,
                "use_class_weights":    args.use_class_weights,
                "train_fraction":       args.train_fraction,
            },
        }, f, indent=2)

    print(f"\n✓ Saved checkpoint       → {out_dir / 'checkpoint.pt'}")
    print(f"✓ Saved best model       → {out_dir / 'best_model.pt'}")
    print(f"✓ Saved training history → {history_path}")
    print(f"✓ Saved metrics          → {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()