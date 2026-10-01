#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-class classification on PAD-UFES-20 (6 classes, ~2298 images)
using timm baseline models with pretrained ImageNet weights.

Supported models:
    deit_base       - DeiT-base (Facebook, ViT-style, ~86M)
    efficientnetv2m - EfficientNetV2-M (Google, ~54M)
    pvtv2b3         - PVTv2-b3 (Wang et al., ~45M)
    davit_base      - DaViT-base (Microsoft, ~87M)
    cvt21           - CvT-21 (Microsoft, ~32M)

Split is done at the PATIENT level to prevent data leakage.
Target image totals matching MedMamba: 1384 / 227 / 687

Usage:
    python pad_ufes_timm_baselines.py --model deit_base --data_root /scratch/cui0011/pad_ufes
    python pad_ufes_timm_baselines.py --model pvtv2b3 --data_root /scratch/cui0011/pad_ufes
    python pad_ufes_timm_baselines.py --model efficientnetv2m --data_root /scratch/cui0011/pad_ufes
"""

import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
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
MODEL_REGISTRY = {
    "deit_base":       ("deit_base_patch16_224", 224),
    "efficientnetv2m": ("tf_efficientnetv2_m",   480),
    "pvtv2b3":         ("pvt_v2_b3",             224),
    "davit_base":      ("davit_base",             224),
    "cvt21":           ("cvt_21",                 224),
}

# -----------------------
# Dataset info
# -----------------------
# BOD has 0 images in this dataset — using 6 classes
CLASSES = ["BCC", "SCC", "ACK", "SEK", "MEL", "NEV"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

TRAIN_TARGET = 1384
VAL_TARGET   = 227
TEST_TARGET  = 687
TOTAL        = TRAIN_TARGET + VAL_TARGET + TEST_TARGET

TRAIN_FRAC = TRAIN_TARGET / TOTAL
VAL_FRAC   = VAL_TARGET   / TOTAL


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
# Patient-level split
# -----------------------
def make_patient_split(df: pd.DataFrame, seed: int = 42):
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

        n_train = max(1, round(n * TRAIN_FRAC))
        n_val   = max(1, round(n * VAL_FRAC))
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

    splits = {"train": split_df(train_pids), "val": split_df(val_pids),
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
          f"{sum(len(s) for s in splits.values()):>10}")
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
    PAD-UFES-20 dataset with patient-level split.

    Expected folder structure:
        data_root/
            metadata.csv
            images/
                imgs_part_*/
                    PAT_*.png
    """

    def __init__(self, data_root: str, split: str = "train",
                 seed: int = 42, img_size: int = 224):
        assert split in ("train", "val", "test")
        self.split     = split
        self.img_size  = img_size
        self.data_root = Path(data_root)
        self.img_dir   = self.data_root / "images"

        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")

        # Build image lookup
        self._img_lookup: dict = {}
        part_dirs = sorted(self.img_dir.glob("imgs_part_*"))
        if not part_dirs:
            part_dirs = [self.img_dir]
        for part_dir in part_dirs:
            for p in part_dir.iterdir():
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                    self._img_lookup[p.name] = p
        print(f"[PAD-UFES] Found {len(self._img_lookup)} images across "
              f"{len(part_dirs)} folder(s)")

        df = pd.read_csv(self.data_root / "metadata.csv")
        df = df[df["diagnostic"].isin(CLASSES)].reset_index(drop=True)

        train_pids, val_pids, test_pids = make_patient_split(df, seed=seed)
        if split == "train":
            print_split_stats(df, train_pids, val_pids, test_pids)

        split_pids = {"train": train_pids, "val": val_pids, "test": test_pids}[split]
        subset = df[df["patient_id"].isin(split_pids)].reset_index(drop=True)

        self.samples: List[Tuple[Path, int]] = []
        missing = []
        for _, row in subset.iterrows():
            img_path = self._img_lookup.get(row["img_id"])
            if img_path is None:
                missing.append(row["img_id"])
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
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_np = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_np = (img_np - mean) / std
        tensor = torch.from_numpy(img_np.transpose(2, 0, 1))
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

    extra_kwargs = {}
    if model_name == "deit_base":
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
    """Unwrap model output to a plain logit tensor."""
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
        description="Train timm baseline models on PAD-UFES-20")

    # Model
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()),
                        help="Model to train: " + ", ".join(MODEL_REGISTRY.keys()))
    parser.add_argument("--no_pretrained", action="store_true",
                        help="Disable pretrained weights (matches MedMamba from-scratch setting)")

    # Data
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root folder containing metadata.csv and images/")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (auto-generated from model name if not set)")
    parser.add_argument("--train_fraction", type=float, default=1.0)

    # Training
    parser.add_argument("--epochs",       type=int,   default=30)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers",  type=int,   default=4)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--use_class_weights", action="store_true")

    args = parser.parse_args()

    pretrained = not args.no_pretrained
    timm_name, img_size = MODEL_REGISTRY[args.model]
    pretrained_str = "pretrained" if pretrained else "scratch"

    if args.out_dir is None:
        args.out_dir = f"./pad_ufes_{args.model}_{pretrained_str}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*70}")
    print(f"PAD-UFES-20 — {args.model} ({timm_name})")
    print(f"{'='*70}")
    print(f"Data root:    {args.data_root}")
    print(f"Output:       {out_dir}")
    print(f"Pretrained:   {pretrained}")
    print(f"Image size:   {img_size}×{img_size}")
    print(f"Classes:      {NUM_CLASSES} {CLASSES}")
    print(f"Split target: {TRAIN_TARGET} train / {VAL_TARGET} val / {TEST_TARGET} test")
    print(f"Split:        patient-level stratified")
    print(f"Device:       {device}")

    # Datasets
    ds_train = PADUFESDataset(args.data_root, split="train",
                              seed=args.seed, img_size=img_size)
    ds_val   = PADUFESDataset(args.data_root, split="val",
                              seed=args.seed, img_size=img_size)
    ds_test  = PADUFESDataset(args.data_root, split="test",
                              seed=args.seed, img_size=img_size)

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
        print(f"\nClass counts (train): "
              f"{ {CLASSES[i]: int(class_counts[i]) for i in range(NUM_CLASSES)} }")

    # DataLoaders
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True,
                          collate_fn=make_collate)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True,
                          collate_fn=make_collate)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True,
                          collate_fn=make_collate)

    # Model
    model = build_model(args.model, num_classes=NUM_CLASSES, pretrained=pretrained)
    model = model.to(device)

    # AdamW + cosine scheduler — standard for all five architectures
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
    print(f"TEST RESULTS — PAD-UFES-20 | {args.model}")
    print(f"{'='*70}")
    print(f"{'P(%)':>8} {'Se(%)':>8} {'Sp(%)':>8} {'F1(%)':>8} {'OA(%)':>8} {'AUC':>8}")
    print(f"{'-'*56}")
    print(f"{p:>8.1f} {se:>8.1f} {sp:>8.1f} {f1:>8.1f} {oa:>8.1f} {auc:>8.3f}")
    print(f"{'='*70}")

    # Confusion matrix terminal print
    col_w = 7
    print(f"\nConfusion Matrix (rows=true, cols=predicted):")
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

        im = ax.imshow(cm_pct, interpolation="nearest", cmap="Blues", vmin=0, vmax=100)
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
        fig.savefig(out_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n✓ Saved confusion matrix → {out_dir / 'confusion_matrix.png'}")
    except ImportError:
        print("Warning: matplotlib not available — skipping confusion matrix plot")

    # Save checkpoint
    torch.save({
        "model_state_dict":      model.state_dict(),
        "model_name":            args.model,
        "timm_name":             timm_name,
        "num_classes":           NUM_CLASSES,
        "classes":               CLASSES,
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
        f.write(f"PAD-UFES-20 — {args.model} Training History\n")
        f.write(f"{'='*80}\n\n")
        f.write("Configuration:\n")
        f.write(f"  Model:          {args.model} ({timm_name})\n")
        f.write(f"  Pretrained:     {pretrained}\n")
        f.write(f"  Image size:     {img_size}×{img_size}\n")
        f.write(f"  Classes:        {CLASSES}\n")
        f.write(f"  Split:          patient-level stratified\n")
        f.write(f"  Split target:   {TRAIN_TARGET} / {VAL_TARGET} / {TEST_TARGET}\n")
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

        header = f"{'Class':<8} {'Prec':>8} {'Sens':>8} {'Spec':>8} {'F1':>8} {'AUC':>8}\n"
        f.write("Per-class Metrics:\n")
        f.write(header)
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
            "timm_name":      timm_name,
            "pretrained":     pretrained,
            "img_size":       img_size,
            "num_classes":    NUM_CLASSES,
            "class_names":    CLASSES,
            "split_strategy": "patient_level_stratified",
            "split_target":   {"train": TRAIN_TARGET, "val": VAL_TARGET,
                               "test": TEST_TARGET},
            "test_metrics":          test_metrics,
            "test_confusion_matrix": test_cm.tolist(),
            "best_val": {
                "epoch":                best_state["epoch"] if best_state else None,
                "val_metrics":          best_state["val_metrics"] if best_state else None,
                "val_confusion_matrix": best_state["val_cm"].tolist() if best_state else None,
            },
            "training_config": {
                "pretrained":        pretrained,
                "lr":                args.lr,
                "weight_decay":      args.weight_decay,
                "batch_size":        args.batch_size,
                "epochs":            args.epochs,
                "use_class_weights": args.use_class_weights,
                "train_fraction":    args.train_fraction,
            },
        }, f, indent=2)

    print(f"\n✓ Saved checkpoint       → {out_dir / 'checkpoint.pt'}")
    print(f"✓ Saved best model       → {out_dir / 'best_model.pt'}")
    print(f"✓ Saved training history → {history_path}")
    print(f"✓ Saved metrics          → {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()