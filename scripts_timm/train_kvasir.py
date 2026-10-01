#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-class classification on Kvasir v1 (8 classes, 4000 images)
using timm baseline models with pretrained ImageNet weights.

Supported models:
    deit_base       - DeiT-base (Facebook, ViT-style, ~86M)
    efficientnetv2m - EfficientNetV2-M (Google, ~54M)
    pvtv2b3         - PVTv2-b3 (Wang et al., ~45M)
    davit_base      - DaViT-base (Microsoft, ~87M)
    cvt21           - CvT-21 (Microsoft, ~32M)

Reproduces the MedMamba split: 2408 train / 392 val / 1200 test.
Uses pretrained ImageNet weights (unlike MedMamba baselines).

Usage:
    python kvasir_timm_baselines.py --model deit_base
    python kvasir_timm_baselines.py --model pvtv2b3 --epochs 30 --lr 1e-4
    python kvasir_timm_baselines.py --model efficientnetv2m --batch_size 32
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
from tqdm.auto import tqdm

try:
    import timm
except ImportError:
    raise ImportError("timm is required: pip install timm")


# -----------------------
# Model registry
# -----------------------
# Maps CLI name → (timm model string, recommended input size)
MODEL_REGISTRY = {
    "deit_base":       ("deit_base_patch16_224",          224),
    "efficientnetv2m": ("tf_efficientnetv2_m",             480),
    "pvtv2b3":         ("pvt_v2_b3",                       224),
    "davit_base":      ("davit_base",                      224),
    "cvt21":           ("cvt_21",                          224),
}

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
# General utils
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
    Kvasir v1 dataset with MedMamba-compatible balanced split.

    split_mode="balanced": exactly equal images per class
        train: 301 per class = 2408 total
        val:    49 per class =  392 total
        test:  150 per class = 1200 total
    """

    def __init__(self, data_root: str, split: str = "train",
                 split_mode: str = "balanced", seed: int = 42,
                 img_size: int = 224):
        assert split in ("train", "val", "test")
        assert split_mode in ("balanced", "random")
        self.split    = split
        self.img_size = img_size
        self.data_root = Path(data_root)

        for cls in KVASIR_CLASSES:
            if not (self.data_root / cls).is_dir():
                raise FileNotFoundError(
                    f"Expected class folder not found: {self.data_root / cls}")

        self.samples: List[Tuple[Path, int]] = []
        rng = np.random.default_rng(seed)
        all_labels = []

        if split_mode == "balanced":
            for class_idx, cls_name in enumerate(KVASIR_CLASSES):
                cls_dir = self.data_root / cls_name
                img_paths = sorted([
                    p for p in cls_dir.iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
                ])
                shuffled = rng.permutation(len(img_paths))
                train_idx = shuffled[:TRAIN_PER_CLASS]
                val_idx   = shuffled[TRAIN_PER_CLASS:TRAIN_PER_CLASS + VAL_PER_CLASS]
                test_idx  = shuffled[TRAIN_PER_CLASS + VAL_PER_CLASS:
                                     TRAIN_PER_CLASS + VAL_PER_CLASS + TEST_PER_CLASS]
                chosen = {"train": train_idx, "val": val_idx, "test": test_idx}[split]
                for i in chosen:
                    self.samples.append((img_paths[i], class_idx))
                    all_labels.append(class_idx)
        else:
            all_paths, all_cls = [], []
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
        # Resize to model input size
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        # Convert to tensor and normalize (ImageNet stats)
        img_np = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
        tensor = torch.from_numpy(img_np.transpose(2, 0, 1))  # [C, H, W]
        return tensor, label, str(path)


def make_collate(batch):
    imgs, ys, paths = zip(*batch)
    return torch.stack(imgs), torch.tensor(ys, dtype=torch.long), paths


# -----------------------
# Model builder
# -----------------------
def build_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    timm_name, _ = MODEL_REGISTRY[model_name]
    print(f"\nLoading timm model: {timm_name} (pretrained={pretrained})")

    # Extra kwargs per model to avoid non-plain-tensor outputs
    extra_kwargs = {}
    if model_name == "deit_base":
        # DeiT returns (logits, distillation_logits) tuple by default;
        # distilled=False makes it return a plain tensor like every other model.
        extra_kwargs["distilled"] = False

    model = timm.create_model(
        timm_name,
        pretrained=pretrained,
        num_classes=num_classes,
        **extra_kwargs,
    )
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✓ Loaded {timm_name}")
    print(f"  Total params:     {total:,}")
    print(f"  Trainable params: {trainable:,}")
    return model


def _get_logits(out) -> torch.Tensor:
    """
    Unwrap model output to a plain logit tensor.
    Handles: plain tensor, tuple/list (DeiT distillation fallback),
    and HuggingFace-style objects with a .logits attribute.
    """
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        return out[0]
    if hasattr(out, "logits"):
        return out.logits
    raise TypeError(f"Cannot extract logits from model output of type {type(out)}")


# -----------------------
# Train / Eval
# -----------------------
def train_one_epoch(model, dl, optimizer, class_weights,
                    device, amp, epoch, total_epochs):
    model.train()

    dtype = (torch.bfloat16
             if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
             else torch.float16)
    use_scaler = amp and device.startswith("cuda") and dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    tot_loss = tot_correct = tot_seen = 0
    pbar = tqdm(dl, desc=f"[Epoch {epoch:02d}/{total_epochs:02d}] train", leave=False)

    for imgs, y, _ in pbar:
        imgs = imgs.to(device, non_blocking=True)
        y    = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp and device.startswith("cuda"),
                                 dtype=dtype):
            logits = _get_logits(model(imgs))
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

        logits = _get_logits(model(imgs))
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

        # Debug: print shapes and unique classes once
        # print(f"\n[DEBUG] y_true shape: {y_true_np.shape}, "
        #       f"unique classes: {np.unique(y_true_np).tolist()}")
        # print(f"[DEBUG] y_probs shape: {y_probs_np.shape}, "
        #       f"dtype: {y_probs_np.dtype}, "
        #       f"sum(axis=1) sample: {y_probs_np[:3].sum(axis=1).tolist()}")

        per_class_auc = []
        for c in range(num_classes):
            yb = (y_true_np == c).astype(int)
            if yb.sum() > 0 and (1 - yb).sum() > 0:
                try:
                    per_class_auc.append(roc_auc_score(yb, y_probs_np[:, c]))
                except Exception as e:
                    print(f"[WARN] AUC failed for class {c}: {e}")
                    per_class_auc.append(float("nan"))
            else:
                print(f"[WARN] Class {c} missing from split — skipping AUC")
                per_class_auc.append(float("nan"))
        valid     = [a for a in per_class_auc if not np.isnan(a)]
        macro_auc = float(np.mean(valid)) if valid else float("nan")
        try:
            auc_ovr = roc_auc_score(y_true_np, y_probs_np,
                                    multi_class="ovr", average="weighted")
        except Exception as e:
            print(f"[WARN] AUC OvR failed: {e}")
            auc_ovr = float("nan")
    except ImportError:
        print("[WARN] sklearn not available — AUC will not be computed")
        per_class_auc = [float("nan")] * num_classes
        macro_auc = auc_ovr = float("nan")
    except Exception as e:
        print(f"[WARN] AUC computation failed entirely: {e}")
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
        description="Train timm baseline models on Kvasir v1")

    # Model
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model to train: " + ", ".join(MODEL_REGISTRY.keys()))
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Disable pretrained weights (matches MedMamba from-scratch setting)")

    # Data
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to unzipped kvasir-dataset folder")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (auto-generated from model name if not set)")
    parser.add_argument("--split_mode", type=str, default="random",
                        choices=["balanced", "random"])
    parser.add_argument("--train_fraction", type=float, default=1.0)

    # Training
    parser.add_argument("--epochs",      type=int,   default=30)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--weight_decay",type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int,   default=4)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--use_class_weights", action="store_true")

    args = parser.parse_args()

    pretrained = not args.no_pretrained
    timm_name, img_size = MODEL_REGISTRY[args.model]

    if args.out_dir is None:
        pretrained_str = "pretrained" if pretrained else "scratch"
        args.out_dir = f"./kvasir_{args.model}_{pretrained_str}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*70}")
    print(f"Kvasir v1 — {args.model} ({timm_name})")
    print(f"{'='*70}")
    print(f"Data root:    {args.data_root}")
    print(f"Output:       {out_dir}")
    print(f"Pretrained:   {pretrained}")
    print(f"Image size:   {img_size}×{img_size}")
    print(f"Split:        {TRAIN_TOTAL} train / {VAL_TOTAL} val / {TEST_TOTAL} test "
          f"({args.split_mode})")
    print(f"Device:       {device}")

    # Datasets
    ds_train = KvasirDataset(args.data_root, split="train",
                             split_mode=args.split_mode, seed=args.seed,
                             img_size=img_size)
    ds_val   = KvasirDataset(args.data_root, split="val",
                             split_mode=args.split_mode, seed=args.seed,
                             img_size=img_size)
    ds_test  = KvasirDataset(args.data_root, split="test",
                             split_mode=args.split_mode, seed=args.seed,
                             img_size=img_size)

    # Stratified subsample if requested
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
        print(f"  → Subsampled to {len(indices)}/{n_total} training samples")

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
                          num_workers=args.num_workers, pin_memory=True,
                          collate_fn=make_collate)
    dl_val   = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True,
                          collate_fn=make_collate)
    dl_test  = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True,
                          collate_fn=make_collate)

    # Model
    model = build_model(args.model, num_classes=NUM_CLASSES, pretrained=pretrained)
    model = model.to(device)

    # Optimizer — AdamW with weight decay is standard for all five architectures
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Cosine LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # Training loop
    best_val_score = -1.0
    best_state     = None
    history        = []

    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(
            model, dl_train, optimizer, class_weights,
            device=device, amp=True, epoch=epoch, total_epochs=args.epochs,
        )
        scheduler.step()

        print(f"[Epoch {epoch:02d}] train_loss={stats['train_loss']:.4f}  "
              f"train_acc={stats['train_acc']:.4f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        val_metrics, val_cm = evaluate(
            model, dl_val, device=device, desc="val", num_classes=NUM_CLASSES)
        print(f"[Epoch {epoch:02d}] VAL  "
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
            "lr":            scheduler.get_last_lr()[0],
        })

        if val_metrics["accuracy"] > best_val_score:
            best_val_score = val_metrics["accuracy"]
            best_state = {
                "model":       model.state_dict(),
                "val_metrics": val_metrics,
                "val_cm":      val_cm.clone(),
                "epoch":       epoch,
            }
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch":            epoch,
                "val_metrics":      val_metrics,
            }, out_dir / "best_model.pt")

    # Test with best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state["model"])

    test_metrics, test_cm = evaluate(
        model, dl_test, device=device, desc="test", num_classes=NUM_CLASSES)

    p   = test_metrics['macro_precision']   * 100
    se  = test_metrics['macro_sensitivity'] * 100
    sp  = test_metrics['macro_specificity'] * 100
    f1  = test_metrics['macro_f1']          * 100
    oa  = test_metrics['accuracy']          * 100
    auc = test_metrics['macro_auc']

    print(f"\n{'='*70}")
    print(f"TEST RESULTS — Kvasir v1 | {args.model}")
    print(f"{'='*70}")
    print(f"{'P(%)':>8} {'Se(%)':>8} {'Sp(%)':>8} {'F1(%)':>8} {'OA(%)':>8} {'AUC':>8}")
    print(f"{'-'*56}")
    print(f"{p:>8.1f} {se:>8.1f} {sp:>8.1f} {f1:>8.1f} {oa:>8.1f} {auc:>8.3f}")
    print(f"{'='*70}")

    # Confusion matrix
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
        cm_float = cm_np.astype(float)
        row_sums = cm_float.sum(axis=1, keepdims=True).clip(min=1)
        cm_pct   = cm_float / row_sums * 100

        im = ax.imshow(cm_pct, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Row %", fontsize=11)

        tick_labels = [c.replace("-", "\n") for c in KVASIR_CLASSES]
        ax.set_xticks(range(NUM_CLASSES))
        ax.set_yticks(range(NUM_CLASSES))
        ax.set_xticklabels(tick_labels, fontsize=8, ha="center")
        ax.set_yticklabels(tick_labels, fontsize=8)

        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                color = "white" if cm_pct[i, j] > 50 else "black"
                ax.text(j, i, f"{int(cm_float[i, j])}\n({cm_pct[i, j]:.1f}%)",
                        ha="center", va="center", fontsize=7, color=color)

        ax.set_xlabel("Predicted label", fontsize=12)
        ax.set_ylabel("True label", fontsize=12)
        ax.set_title(
            f"Kvasir v1 — {args.model} ({'pretrained' if pretrained else 'scratch'})\n"
            f"OA={oa:.1f}%  F1={f1:.1f}%  AUC={auc:.3f}", fontsize=12)

        plt.tight_layout()
        fig.savefig(out_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n✓ Saved confusion matrix → {out_dir / 'confusion_matrix.png'}")
    except ImportError:
        print("Warning: matplotlib not available — skipping confusion matrix plot")

    # Save full checkpoint
    torch.save({
        "model_state_dict":      model.state_dict(),
        "model_name":            args.model,
        "timm_name":             timm_name,
        "num_classes":           NUM_CLASSES,
        "classes":               KVASIR_CLASSES,
        "pretrained":            pretrained,
        "img_size":              img_size,
        "best_val": {
            "epoch":                best_state["epoch"] if best_state else None,
            "val_metrics":          best_state["val_metrics"] if best_state else None,
            "val_confusion_matrix": best_state["val_cm"].tolist() if best_state else None,
        },
        "test_metrics":          test_metrics,
        "test_confusion_matrix": test_cm.tolist(),
    }, out_dir / "checkpoint.pt")

    # Training history text file
    history_path = out_dir / "training_history.txt"
    with open(history_path, "w") as f:
        f.write(f"Kvasir v1 — {args.model} Training History\n")
        f.write(f"{'='*80}\n\n")
        f.write("Configuration:\n")
        f.write(f"  Model:          {args.model} ({timm_name})\n")
        f.write(f"  Pretrained:     {pretrained}\n")
        f.write(f"  Image size:     {img_size}×{img_size}\n")
        f.write(f"  Split:          {TRAIN_TOTAL} / {VAL_TOTAL} / {TEST_TOTAL} ({args.split_mode})\n")
        f.write(f"  Batch size:     {args.batch_size}\n")
        f.write(f"  LR:             {args.lr}\n")
        f.write(f"  Weight decay:   {args.weight_decay}\n")
        f.write(f"  Scheduler:      CosineAnnealingLR (eta_min={args.lr*0.01:.2e})\n")
        f.write(f"  Epochs:         {args.epochs}\n")
        f.write(f"  Class weights:  {args.use_class_weights}\n\n")

        f.write("Training History:\n")
        f.write(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} "
                f"{'Val Loss':>12} {'Val Acc':>10} {'Val F1':>10} "
                f"{'Val AUC':>10} {'LR':>12}\n")
        f.write(f"{'-'*100}\n")
        for h in history:
            f.write(f"{h['epoch']:>6} {h['train_loss']:>12.4f} {h['train_acc']:>10.4f} "
                    f"{h['val_loss']:>12.4f} {h['val_acc']:>10.4f} "
                    f"{h['val_macro_f1']:>10.4f} {h['val_macro_auc']:>10.4f} "
                    f"{h['lr']:>12.2e}\n")

        f.write(f"\n{'='*100}\n")
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
            "dataset":     "kvasir_v1",
            "model":       args.model,
            "timm_name":   timm_name,
            "pretrained":  pretrained,
            "img_size":    img_size,
            "num_classes": NUM_CLASSES,
            "class_names": KVASIR_CLASSES,
            "test_metrics":          test_metrics,
            "test_confusion_matrix": test_cm.tolist(),
            "best_val": {
                "epoch":                best_state["epoch"] if best_state else None,
                "val_metrics":          best_state["val_metrics"] if best_state else None,
                "val_confusion_matrix": best_state["val_cm"].tolist() if best_state else None,
            },
            "training_config": {
                "pretrained":    pretrained,
                "lr":            args.lr,
                "weight_decay":  args.weight_decay,
                "batch_size":    args.batch_size,
                "epochs":        args.epochs,
                "split_mode":    args.split_mode,
                "use_class_weights": args.use_class_weights,
            },
        }, f, indent=2)

    print(f"\n✓ Saved checkpoint       → {out_dir / 'checkpoint.pt'}")
    print(f"✓ Saved best model       → {out_dir / 'best_model.pt'}")
    print(f"✓ Saved training history → {history_path}")
    print(f"✓ Saved metrics          → {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()