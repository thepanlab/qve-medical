#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-class classification on PAD-UFES-20 (6 classes, 2298 images)
using Qwen3-VL vision encoder.
Supports optional multi-tap feature extraction with depth fusion.

Split is done at the PATIENT level to prevent data leakage:
all images of a given patient appear in exactly one of train/val/test.
Stratification uses each patient's primary class (most frequent lesion).

Target image totals matching MedMamba: 1384 / 227 / 687
(approximate — exact counts depend on patient grouping)

Note: BOD (Bowen's disease) has 0 images in this dataset → 6 classes used.

Usage:
    python pad_ufes_train.py --data_root /scratch/cui0011/pad_ufes
    python pad_ufes_train.py --data_root /scratch/cui0011/pad_ufes --mode full_finetune
    python pad_ufes_train.py --data_root /scratch/cui0011/pad_ufes --scratch
    python pad_ufes_train.py --data_root /scratch/cui0011/pad_ufes \
        --use_multi_tap --tap_layers 6 13 20 --fusion_type token_attention --pooling_type mean
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

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, Qwen3VLConfig

from multi_tap_extractor import MultiTapExtractor
from depth_fusion import FusionAndPooling


# -----------------------
# Dataset info
# -----------------------
# BOD has 0 images in this dataset — using 6 classes
CLASSES = ["BCC", "SCC", "ACK", "SEK", "MEL", "NEV"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

# MedMamba target image totals (approximate after patient-level split)
TRAIN_TARGET = 1384
VAL_TARGET   = 227
TEST_TARGET  = 687
TOTAL        = TRAIN_TARGET + VAL_TARGET + TEST_TARGET  # 2298

TRAIN_FRAC = TRAIN_TARGET / TOTAL  # ~0.602
VAL_FRAC   = VAL_TARGET   / TOTAL  # ~0.099


# -----------------------
# General utils
# -----------------------
def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _dig(model, path: str):
    cur = model
    for attr in path.split("."):
        if not hasattr(cur, attr):
            return None
        cur = getattr(cur, attr)
    return cur


def _pick_vision_backbone(cand) -> Optional[nn.Module]:
    if cand is None:
        return None
    for subname in ["vision_model", "visual", "image_encoder", "backbone", "model"]:
        if hasattr(cand, subname):
            sub = getattr(cand, subname)
            if isinstance(sub, nn.Module):
                return sub
    return cand if isinstance(cand, nn.Module) else None


def find_vision_backbone(model) -> nn.Module:
    entry_paths = [
        "vision_tower", "model.vision_tower", "model.model.vision_tower",
        "visual", "model.visual", "vision_model", "model.vision_model",
        "image_encoder", "model.image_encoder", "vision_backbone", "model.vision_backbone",
        "vision_tower.vision_tower", "model.vision_tower.vision_tower",
    ]
    for p in entry_paths:
        cand = _dig(model, p)
        vb = _pick_vision_backbone(cand)
        if isinstance(vb, nn.Module):
            return vb
    if hasattr(model, "get_vision_tower"):
        try:
            cand = model.get_vision_tower()
            vb = _pick_vision_backbone(cand)
            if isinstance(vb, nn.Module):
                return vb
        except Exception:
            pass
    for name, module in model.named_modules():
        low = (name + " " + module.__class__.__name__).lower()
        if any(k in low for k in ("vision", "visual", "image", "clip", "siglip")) \
                and "projector" not in low:
            if hasattr(module, "forward") and len(list(module.children())) > 0:
                return module
    raise AttributeError("Could not locate a vision backbone in this Qwen*-VL build.")


def count_parameters(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def print_trainable_parameters(model):
    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model, trainable_only=False)
    print(f"Trainable params: {trainable:,} || Total params: {total:,} || "
          f"Trainable%: {100 * trainable / total:.2f}%")


# -----------------------
# Fine-tuning configuration
# -----------------------
def configure_model_training(base_model, vision_backbone, mode="frozen",
                              unfreeze_layers=0, gradient_checkpointing=False):
    print(f"\n{'='*60}")
    print(f"Configuring model in '{mode}' mode")
    print(f"{'='*60}")

    for p in base_model.parameters():
        p.requires_grad = False

    if mode == "frozen":
        print("✓ All parameters frozen (feature extraction only)")
        base_model.eval()
        vision_backbone.eval()

    elif mode == "full_finetune":
        print("✓ Unfreezing entire vision backbone")
        for p in vision_backbone.parameters():
            p.requires_grad = True
        vision_backbone.train()
        if gradient_checkpointing:
            print("✓ Enabling gradient checkpointing")
            if hasattr(base_model, "gradient_checkpointing_enable"):
                base_model.gradient_checkpointing_enable()

    elif mode == "partial_finetune":
        if unfreeze_layers <= 0:
            raise ValueError("For partial_finetune, --unfreeze_layers must be > 0")
        print(f"✓ Unfreezing last {unfreeze_layers} layer(s) of vision backbone")

        all_params = list(vision_backbone.named_parameters())
        layer_names = []
        for name, _ in all_params:
            for pattern in ["blocks.", "layers.", "layer.", "encoder.layer."]:
                if pattern in name:
                    try:
                        layer_num = int(name.split(pattern)[1].split(".")[0])
                        layer_names.append((name, layer_num))
                    except (IndexError, ValueError):
                        continue
                    break

        if layer_names:
            unique_layers = sorted(set(ln[1] for ln in layer_names))
            layers_to_unfreeze = set(unique_layers[-unfreeze_layers:])
            print(f"  Unfreezing layers: {sorted(layers_to_unfreeze)}")
            unfrozen = 0
            for name, param in vision_backbone.named_parameters():
                for ln, layer_num in layer_names:
                    if name == ln and layer_num in layers_to_unfreeze:
                        param.requires_grad = True
                        unfrozen += 1
                        break
            print(f"  Unfrozen {unfrozen} parameter tensors")
        else:
            print("  Could not identify layer structure; unfreezing last params by position")
            total_p = len(all_params)
            start_idx = max(0, total_p - int(total_p * unfreeze_layers / 10))
            for idx, (_, param) in enumerate(all_params):
                if idx >= start_idx:
                    param.requires_grad = True

        vision_backbone.train()
        if gradient_checkpointing:
            if hasattr(base_model, "gradient_checkpointing_enable"):
                base_model.gradient_checkpointing_enable()

    else:
        raise ValueError(f"Unknown mode: {mode}")

    print("\nVision Backbone Parameters:")
    print_trainable_parameters(vision_backbone)
    return base_model, vision_backbone


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

    splits = {
        "train": split_df(train_pids),
        "val":   split_df(val_pids),
        "test":  split_df(test_pids),
    }

    print(f"\n{'='*70}")
    print(f"SPLIT SUMMARY  (target: {TRAIN_TARGET}/{VAL_TARGET}/{TEST_TARGET})")
    print(f"{'='*70}")
    print(f"{'':>8} {'train':>10} {'val':>10} {'test':>10} {'total':>10}")
    print(f"{'-'*50}")
    print(f"{'patients':>8} {len(train_pids):>10} {len(val_pids):>10} "
          f"{len(test_pids):>10} {len(train_pids)+len(val_pids)+len(test_pids):>10}")
    print(f"{'images':>8} {len(splits['train']):>10} {len(splits['val']):>10} "
          f"{len(splits['test']):>10} "
          f"{len(splits['train'])+len(splits['val'])+len(splits['test']):>10}")
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
    PAD-UFES-20 dataset loaded from metadata CSV + image folder.

    Expected folder structure:
        data_root/
            metadata.csv
            images/
                imgs_part_*/
                    PAT_1516_1765_530.png
                    ...

    Split is patient-level: all images of a patient are in one split only.
    """

    def __init__(self, data_root: str, split: str = "train", seed: int = 42):
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.split     = split
        self.data_root = Path(data_root)
        self.img_dir   = self.data_root / "images"

        if not self.img_dir.is_dir():
            raise FileNotFoundError(
                f"Image directory not found: {self.img_dir}\n"
                f"Expected images under {self.img_dir}/imgs_part_*/"
            )

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
        return img, label, str(path)


# -----------------------
# Collate & vision forward
# -----------------------
def make_collate_dynamic(processor, min_pixels: int, max_pixels: int):
    def collate(batch):
        imgs, ys, paths = zip(*batch)
        enc = processor(
            images=list(imgs),
            text=[""] * len(imgs),
            return_tensors="pt",
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
        pixel = enc["pixel_values"]
        grid  = enc.get("image_grid_thw", enc.get("grid_thw", None))
        if not torch.is_tensor(grid):
            grid = torch.tensor(grid)
        y = torch.tensor(ys, dtype=torch.long)
        return pixel, grid, y, paths
    return collate


def forward_vision_positional(vision_backbone, pixel_values, grid_thw,
                               requires_grad=False):
    if requires_grad:
        out = vision_backbone(pixel_values, grid_thw)
    else:
        with torch.no_grad():
            out = vision_backbone(pixel_values, grid_thw)

    feats = None
    if isinstance(out, (tuple, list)):
        feats = out[0]
    else:
        feats = getattr(out, "last_hidden_state", None) or getattr(out, "pooler_output", None)
        if feats is None:
            for v in out.__dict__.values():
                if torch.is_tensor(v):
                    feats = v
                    break

    if feats is None:
        raise RuntimeError("Cannot find tensor features in vision output.")

    feats = feats.float()

    if feats.ndim == 3:
        feats = feats.mean(dim=1)
    elif feats.ndim == 2:
        first_dim  = feats.shape[0]
        batch_size = grid_thw.shape[0]
        if first_dim == batch_size:
            pass
        elif first_dim % batch_size == 0:
            feats = feats.reshape(batch_size, first_dim // batch_size, -1).mean(dim=1)
        else:
            patches_per_image = (grid_thw[:, 1] * grid_thw[:, 2]).tolist()
            total_patches = sum(patches_per_image)
            if first_dim == total_patches:
                agg, start = [], 0
                for n in patches_per_image:
                    n = int(n)
                    agg.append(feats[start:start + n].mean(dim=0, keepdim=True))
                    start += n
                feats = torch.cat(agg, dim=0)
            else:
                agg, start = [], 0
                for n in patches_per_image:
                    actual = min(int(first_dim * n / total_patches), first_dim - start)
                    chunk  = feats[start:start + actual]
                    agg.append(chunk.mean(dim=0, keepdim=True) if chunk.size(0) > 0
                               else torch.zeros(1, feats.size(1),
                                                device=feats.device, dtype=feats.dtype))
                    start += actual
                feats = torch.cat(agg, dim=0)
    else:
        raise ValueError(f"Unexpected feature tensor shape: {feats.shape}")

    return feats


# -----------------------
# Linear head
# -----------------------
class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


# -----------------------
# Train / Eval
# -----------------------
def train_one_epoch(vb, dl, head, optimizer, class_weights,
                    device, amp, epoch, total_epochs, requires_grad,
                    extractor=None, fusion_module=None):
    head.train()
    if fusion_module is not None:
        fusion_module.train()
    vb.train() if requires_grad else vb.eval()

    tot_loss = tot_correct = tot_seen = 0
    dtype = (torch.bfloat16
             if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
             else torch.float16)
    use_scaler = amp and device.startswith("cuda") and dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    pbar = tqdm(dl, desc=f"[Epoch {epoch:02d}/{total_epochs:02d}] train", leave=False)
    for pixel_values, grid_thw, y, _ in pbar:
        pixel_values = pixel_values.to(device, non_blocking=True)
        grid_thw     = grid_thw.to(device, non_blocking=True)
        y            = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp and device.startswith("cuda"),
                                 dtype=dtype):
            if extractor is not None:
                tapped_features, mask = extractor(pixel_values, grid_thw,
                                                  requires_grad=requires_grad)
                feats = fusion_module(tapped_features, mask)
            else:
                feats = forward_vision_positional(vb, pixel_values, grid_thw,
                                                  requires_grad=requires_grad)

            logits = head(feats)
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
def evaluate(vb, dl, head, device, desc, num_classes,
             extractor=None, fusion_module=None):
    head.eval()
    if fusion_module is not None:
        fusion_module.eval()
    vb.eval()

    all_y, all_p, all_probs = [], [], []
    tot_loss = tot_seen = 0

    pbar = tqdm(dl, desc=f"[{desc}]", leave=False)
    for pixel_values, grid_thw, y, _ in pbar:
        pixel_values = pixel_values.to(device, non_blocking=True)
        grid_thw     = grid_thw.to(device, non_blocking=True)
        y            = y.to(device, non_blocking=True)

        if extractor is not None:
            tapped_features, mask = extractor(pixel_values, grid_thw, requires_grad=False)
            feats = fusion_module(tapped_features, mask)
        else:
            feats = forward_vision_positional(vb, pixel_values, grid_thw)

        feats  = feats.to(next(head.parameters()).dtype)
        logits = head(feats)
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
        valid = [a for a in per_class_auc if not np.isnan(a)]
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
    parser = argparse.ArgumentParser(description="Train Qwen3-VL on PAD-UFES-20")

    # Data
    parser.add_argument("--data_root", type=str,
                        default="/scratch/cui0011/pad_ufes",
                        help="Root folder containing metadata.csv and images/")
    parser.add_argument("--out_dir", type=str,
                        default="./pad_ufes_runs",
                        help="Output directory")
    parser.add_argument("--train_fraction", type=float, default=1.0,
                        help="Fraction of training data to use (stratified by class)")

    # Model
    parser.add_argument("--model_id", type=str,
                        default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--scratch", action="store_true",
                        help="Randomly initialise vision backbone weights "
                             "(matches MedMamba from-scratch setting)")

    # Multi-tap feature extraction
    parser.add_argument("--use_multi_tap", action="store_true",
                        help="Use MultiTapExtractor for multi-layer feature extraction")
    parser.add_argument("--tap_layers", nargs="+", type=int, default=[6, 13, 20],
                        help="Which transformer layers to tap (e.g., 6 13 20)")
    parser.add_argument("--project_to_dim", type=int, default=4096,
                        help="Project tapped features to this dimension before fusion")
    parser.add_argument("--fusion_type", type=str, default="token_attention",
                        choices=["scalar", "token_attention", "gated"],
                        help="Depth fusion strategy")
    parser.add_argument("--pooling_type", type=str, default="mean",
                        choices=["mean", "max", "attention", "cls"],
                        help="Pooling strategy after fusion")

    # Resolution
    parser.add_argument("--min_pixels", type=int, default=300 * 300)
    parser.add_argument("--max_pixels", type=int, default=300 * 300)

    # Training
    parser.add_argument("--mode", type=str, default="frozen",
                        choices=["frozen", "partial_finetune", "full_finetune"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--backbone_lr", type=float, default=1e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Fine-tuning options
    parser.add_argument("--unfreeze_layers", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_class_weights", action="store_true")

    args = parser.parse_args()

    if not (0.0 < args.train_fraction <= 1.0):
        raise ValueError(f"--train_fraction must be in (0, 1], got {args.train_fraction}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pretrained_str = "from_scratch" if args.scratch else "pretrained"

    print(f"\n{'='*70}")
    print(f"Training Qwen3-VL on PAD-UFES-20")
    print(f"{'='*70}")
    print(f"Data root:      {args.data_root}")
    print(f"Output:         {out_dir}")
    print(f"Classes:        {NUM_CLASSES} {CLASSES}")
    print(f"Split target:   {TRAIN_TARGET} train / {VAL_TARGET} val / {TEST_TARGET} test")
    print(f"Split strategy: patient-level stratified")
    print(f"Mode:           {args.mode}")
    print(f"Pretrained:     {not args.scratch} ({pretrained_str})")
    print(f"Device:         {device}")
    print(f"Train fraction: {args.train_fraction*100:.1f}%")
    if args.use_multi_tap:
        print(f"Multi-tap:      layers={args.tap_layers}, project_to={args.project_to_dim}, "
              f"fusion={args.fusion_type}, pooling={args.pooling_type}")

    # Datasets
    ds_train = PADUFESDataset(args.data_root, split="train", seed=args.seed)
    ds_val   = PADUFESDataset(args.data_root, split="val",   seed=args.seed)
    ds_test  = PADUFESDataset(args.data_root, split="test",  seed=args.seed)

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
        total = class_counts.sum()
        weights = [total / (NUM_CLASSES * max(1, int(c))) for c in class_counts]
        class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
        print(f"\nClass weights: {class_weights.cpu().tolist()}")
    else:
        class_weights = None
        print(f"\nClass counts (train): { {CLASSES[i]: int(class_counts[i]) for i in range(NUM_CLASSES)} }")

    # Load model
    print(f"\nLoading Qwen3-VL ({pretrained_str})...")
    processor = AutoProcessor.from_pretrained(args.model_id)

    if args.scratch:
        config = Qwen3VLConfig.from_pretrained(args.model_id)
        base_model = Qwen3VLForConditionalGeneration(config)
        base_model = base_model.to(torch.bfloat16).to(device)
        print(f"✓ Architecture from {args.model_id} config, weights randomly initialised")
    else:
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        print(f"✓ Loaded pretrained weights from {args.model_id}")

    vision_backbone = find_vision_backbone(base_model)

    base_model, vision_backbone = configure_model_training(
        base_model, vision_backbone,
        mode=args.mode,
        unfreeze_layers=args.unfreeze_layers,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # Multi-tap extractor + fusion module
    extractor     = None
    fusion_module = None
    if args.use_multi_tap:
        print(f"\n{'='*60}")
        print(f"Initializing Multi-Tap Extractor")
        print(f"{'='*60}")
        extractor = MultiTapExtractor(
            vision_backbone=vision_backbone,
            tap_layers=args.tap_layers,
            project_to_dim=args.project_to_dim,
            use_layernorm=True,
        )
        print(f"✓ Multi-tap extractor ready")

        print(f"\n{'='*60}")
        print(f"Initializing Fusion Module")
        print(f"{'='*60}")
        print(f"  Fusion type:  {args.fusion_type}")
        print(f"  Pooling type: {args.pooling_type}")
        fusion_module = FusionAndPooling(
            hidden_size=args.project_to_dim,
            num_taps=len(args.tap_layers),
            fusion_type=args.fusion_type,
            pooling_type=args.pooling_type,
            use_layernorm=True,
        ).to(device)
        fusion_params = sum(p.numel() for p in fusion_module.parameters())
        print(f"  Fusion parameters: {fusion_params:,}")
        print(f"✓ Fusion module ready")

    collate = make_collate_dynamic(processor, args.min_pixels, args.max_pixels)

    # Infer hidden size
    print("\nInferring feature dimension...")
    dummy_loader = DataLoader(ds_train, batch_size=2, shuffle=False,
                              num_workers=0, collate_fn=collate)
    pv, gt, _, _ = next(iter(dummy_loader))
    pv = pv.to(device)
    gt = gt.to(device)

    if args.use_multi_tap:
        with torch.no_grad():
            tapped_features, mask = extractor(pv, gt, requires_grad=False)
            feats_dummy = fusion_module(tapped_features, mask)
        print(f"✓ Multi-tap + fusion: {len(tapped_features)} taps → fused features")
    else:
        with torch.no_grad():
            feats_dummy = forward_vision_positional(vision_backbone, pv, gt)

    hidden_size = feats_dummy.shape[-1]
    print(f"✓ Hidden size: {hidden_size}")

    requires_grad = args.mode in ("partial_finetune", "full_finetune")

    # DataLoaders
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True,
                          collate_fn=collate)
    dl_val   = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True,
                          collate_fn=collate)
    dl_test  = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True,
                          collate_fn=collate)

    # Head & optimizer
    head = LinearHead(hidden_size, NUM_CLASSES).to(device)
    print(f"\nLinear head: {hidden_size} → {NUM_CLASSES}")

    if requires_grad:
        param_groups = [
            {"params": head.parameters(), "lr": args.lr},
            {"params": [p for p in vision_backbone.parameters() if p.requires_grad],
             "lr": args.backbone_lr},
        ]
        if fusion_module is not None:
            param_groups.append({"params": fusion_module.parameters(), "lr": args.lr})
        optimizer = torch.optim.Adam(param_groups)
    else:
        params = list(head.parameters())
        if fusion_module is not None:
            params += list(fusion_module.parameters())
        optimizer = torch.optim.Adam(params, lr=args.lr)

    # Training loop
    best_val_score = -1.0
    best_state     = None
    history        = []

    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(
            vision_backbone, dl_train, head, optimizer, class_weights,
            device=device, amp=True, epoch=epoch, total_epochs=args.epochs,
            requires_grad=requires_grad,
            extractor=extractor, fusion_module=fusion_module,
        )
        print(f"[Epoch {epoch:02d}] train_loss={stats['train_loss']:.4f}  "
              f"train_acc={stats['train_acc']:.4f}")

        val_metrics, val_cm = evaluate(
            vision_backbone, dl_val, head,
            device=device, desc="val", num_classes=NUM_CLASSES,
            extractor=extractor, fusion_module=fusion_module,
        )
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
        })

        if val_metrics["accuracy"] > best_val_score:
            best_val_score = val_metrics["accuracy"]
            best_state = {
                "head":        head.state_dict(),
                "val_metrics": val_metrics,
                "val_cm":      val_cm.clone(),
                "epoch":       epoch,
            }
            ckpt = {"head_state_dict": head.state_dict(), "epoch": epoch,
                    "val_metrics": val_metrics}
            if requires_grad:
                ckpt["vision_backbone_state_dict"] = vision_backbone.state_dict()
            if fusion_module is not None:
                ckpt["fusion_module_state_dict"] = fusion_module.state_dict()
            torch.save(ckpt, out_dir / "best_model.pt")

    # Test with best checkpoint
    if best_state is not None:
        head.load_state_dict(best_state["head"])

    test_metrics, test_cm = evaluate(
        vision_backbone, dl_test, head,
        device=device, desc="test", num_classes=NUM_CLASSES,
        extractor=extractor, fusion_module=fusion_module,
    )

    p   = test_metrics['macro_precision']   * 100
    se  = test_metrics['macro_sensitivity'] * 100
    sp  = test_metrics['macro_specificity'] * 100
    f1  = test_metrics['macro_f1']          * 100
    oa  = test_metrics['accuracy']          * 100
    auc = test_metrics['macro_auc']

    print(f"\n{'='*70}")
    print(f"TEST RESULTS — PAD-UFES-20 ({pretrained_str})")
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
            f"PAD-UFES-20 — Qwen3-VL ({pretrained_str})\n"
            f"OA={oa:.1f}%  F1={f1:.1f}%  AUC={auc:.3f}", fontsize=12)

        plt.tight_layout()
        cm_path = out_dir / "confusion_matrix.png"
        fig.savefig(cm_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n✓ Saved confusion matrix → {cm_path}")
    except ImportError:
        print("Warning: matplotlib not available — skipping confusion matrix plot")

    # Save checkpoint
    save_dict = {
        "head_state_dict":       head.state_dict(),
        "hidden_size":           hidden_size,
        "classes":               CLASSES,
        "model_id":              args.model_id,
        "pretrained_str":        pretrained_str,
        "scratch":               args.scratch,
        "min_pixels":            args.min_pixels,
        "max_pixels":            args.max_pixels,
        "training_mode":         args.mode,
        "num_classes":           NUM_CLASSES,
        "train_fraction":        args.train_fraction,
        "best_val": {
            "epoch":                best_state["epoch"] if best_state else None,
            "val_metrics":          best_state["val_metrics"] if best_state else None,
            "val_confusion_matrix": best_state["val_cm"].tolist() if best_state else None,
        },
        "test_metrics":          test_metrics,
        "test_confusion_matrix": test_cm.tolist(),
    }
    if fusion_module is not None:
        save_dict["fusion_module_state_dict"] = fusion_module.state_dict()
        save_dict["fusion_type"]    = args.fusion_type
        save_dict["pooling_type"]   = args.pooling_type
        save_dict["tap_layers"]     = args.tap_layers
        save_dict["project_to_dim"] = args.project_to_dim
    if requires_grad:
        save_dict["vision_backbone_state_dict"] = {
            k: v for k, v in vision_backbone.state_dict().items()
            if any(p.requires_grad for p in vision_backbone.parameters())
        }
    torch.save(save_dict, out_dir / "checkpoint.pt")

    # Training history text file
    history_path = out_dir / "training_history.txt"
    with open(history_path, "w") as f:
        f.write(f"PAD-UFES-20 Classification Training History ({pretrained_str})\n")
        f.write(f"{'='*80}\n\n")
        f.write("Configuration:\n")
        f.write(f"  Model:          {args.model_id}\n")
        f.write(f"  Pretrained:     {not args.scratch} ({pretrained_str})\n")
        f.write(f"  Training mode:  {args.mode}\n")
        f.write(f"  Classes:        {CLASSES}\n")
        f.write(f"  Split:          patient-level stratified\n")
        f.write(f"  Split target:   {TRAIN_TARGET} / {VAL_TARGET} / {TEST_TARGET}\n")
        f.write(f"  Train fraction: {args.train_fraction*100:.1f}%\n")
        f.write(f"  Batch size:     {args.batch_size}\n")
        f.write(f"  Head LR:        {args.lr}\n")
        if requires_grad:
            f.write(f"  Backbone LR:    {args.backbone_lr}\n")
        f.write(f"  Class weights:  {args.use_class_weights}\n")
        f.write(f"  Epochs:         {args.epochs}\n")
        if args.use_multi_tap:
            f.write(f"\nMulti-Tap Extractor:\n")
            f.write(f"  Tap layers:     {args.tap_layers}\n")
            f.write(f"  Project to dim: {args.project_to_dim}\n")
            f.write(f"  Fusion type:    {args.fusion_type}\n")
            f.write(f"  Pooling type:   {args.pooling_type}\n")
        f.write("\n")

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
            "num_classes":    NUM_CLASSES,
            "class_names":    CLASSES,
            "pretrained_str": pretrained_str,
            "scratch":        args.scratch,
            "split_strategy": "patient_level_stratified",
            "split_target":   {"train": TRAIN_TARGET, "val": VAL_TARGET,
                               "test": TEST_TARGET},
            "train_fraction": args.train_fraction,
            "test_metrics":   test_metrics,
            "test_confusion_matrix": test_cm.tolist(),
            "best_val": {
                "epoch":                best_state["epoch"] if best_state else None,
                "val_metrics":          best_state["val_metrics"] if best_state else None,
                "val_confusion_matrix": best_state["val_cm"].tolist() if best_state else None,
            },
            "training_config": {
                "mode":                   args.mode,
                "model_id":               args.model_id,
                "head_lr":                args.lr,
                "backbone_lr":            args.backbone_lr if requires_grad else None,
                "unfreeze_layers":        args.unfreeze_layers if args.mode == "partial_finetune" else None,
                "gradient_checkpointing": args.gradient_checkpointing,
                "use_class_weights":      args.use_class_weights,
                "batch_size":             args.batch_size,
                "min_pixels":             args.min_pixels,
                "max_pixels":             args.max_pixels,
                "train_fraction":         args.train_fraction,
                "scratch":                args.scratch,
                "use_multi_tap":          args.use_multi_tap,
                "tap_layers":             args.tap_layers if args.use_multi_tap else None,
                "project_to_dim":         args.project_to_dim if args.use_multi_tap else None,
                "fusion_type":            args.fusion_type if args.use_multi_tap else None,
                "pooling_type":           args.pooling_type if args.use_multi_tap else None,
            },
        }, f, indent=2)

    print(f"\n✓ Saved checkpoint       → {out_dir / 'checkpoint.pt'}")
    print(f"✓ Saved best model       → {out_dir / 'best_model.pt'}")
    print(f"✓ Saved training history → {history_path}")
    print(f"✓ Saved metrics          → {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()