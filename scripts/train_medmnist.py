#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Multi-class classification on MedMNIST using Qwen3-VL vision encoder.

Usage:
    python medmnist_train.py --medmnist pathmnist
    python medmnist_train.py --medmnist tissuemnist --mode frozen --epochs 20
"""

from html import parser
import os

from requests import head
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm.auto import tqdm

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from multi_tap_extractor import MultiTapExtractor  
from depth_fusion import FusionAndPooling, TokenWiseDepthFusion

from cross_attn_mil import CrossAttnMILHead, MILPrototypeHead, CrossAttentionReasoning

# -----------------------
# Dataset info
# -----------------------
MEDMNIST_INFO = {
    "pathmnist": {
        "num_classes": 9,
        "class_names": [
            "adipose", "background", "debris", "lymphocytes", "mucus",
            "smooth muscle", "normal colon mucosa", "cancer-associated stroma",
            "colorectal adenocarcinoma epithelium"
        ]
    },
    "tissuemnist": {
        "num_classes": 8,
        "class_names": [
            "Collecting Duct, Connecting Tubule", "Distal Convoluted Tubule",
            "Glomerular endothelial cells", "Interstitial endothelial cells",
            "Leukocytes", "Podocytes", "Proximal Tubule Segments", "Thick Ascending Limb"
        ]
    },
    "organamnist": {"num_classes": 11, "class_names": None},
    "organcmnist": {"num_classes": 11, "class_names": None},
    "organsmnist": {"num_classes": 11, "class_names": None},
    "dermamnist": {"num_classes": 7, "class_names": None},
    "bloodmnist": {"num_classes": 8, "class_names": None},
    "retinamnist": {"num_classes": 5, "class_names": None},
    "breastmnist": {"num_classes": 2, "class_names": ["benign", "malignant"]},
    "pneumoniamnist": {"num_classes": 2, "class_names": ["normal", "pneumonia"]},
    "octmnist": {"num_classes": 4, "class_names": ["CNV", "DME", "DRUSEN", "NORMAL"]},
}


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


def _pick_vision_backbone(cand) -> Optional[torch.nn.Module]:
    if cand is None:
        return None
    for subname in ["vision_model", "visual", "image_encoder", "backbone", "model"]:
        if hasattr(cand, subname):
            sub = getattr(cand, subname)
            if isinstance(sub, torch.nn.Module):
                return sub
    return cand if isinstance(cand, torch.nn.Module) else None


def find_vision_backbone(model) -> torch.nn.Module:
    entry_paths = [
        "vision_tower", "model.vision_tower", "model.model.vision_tower",
        "visual", "model.visual", "vision_model", "model.vision_model",
        "image_encoder", "model.image_encoder", "vision_backbone", "model.vision_backbone",
        "vision_tower.vision_tower", "model.vision_tower.vision_tower",
    ]
    for p in entry_paths:
        cand = _dig(model, p)
        vb = _pick_vision_backbone(cand)
        if isinstance(vb, torch.nn.Module):
            return vb
    if hasattr(model, "get_vision_tower"):
        try:
            cand = model.get_vision_tower()
            vb = _pick_vision_backbone(cand)
            if isinstance(vb, torch.nn.Module):
                return vb
        except Exception:
            pass
    for name, module in model.named_modules():
        low = (name + " " + module.__class__.__name__).lower()
        if any(k in low for k in ("vision", "visual", "image", "clip", "siglip")) and "projector" not in low:
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
    print(f"Trainable params: {trainable:,} || Total params: {total:,} || Trainable%: {100 * trainable / total:.2f}%")


# -----------------------
# Fine-tuning configuration
# -----------------------
def configure_model_training(
    base_model,
    vision_backbone,
    mode: str = "frozen",
    unfreeze_layers: int = 0,
    gradient_checkpointing: bool = False
):
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
        print("✓ Unfreezing entire vision backbone for full fine-tuning")
        for p in vision_backbone.parameters():
            p.requires_grad = True
        vision_backbone.train()

        if gradient_checkpointing:
            print("✓ Enabling gradient checkpointing")
            if hasattr(base_model, 'gradient_checkpointing_enable'):
                base_model.gradient_checkpointing_enable()
            elif hasattr(vision_backbone, 'gradient_checkpointing_enable'):
                vision_backbone.gradient_checkpointing_enable()

    elif mode == "partial_finetune":
        if unfreeze_layers <= 0:
            raise ValueError("For partial_finetune, unfreeze_layers must be > 0")

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
            layers_to_unfreeze = unique_layers[-unfreeze_layers:]
            print(f"  Found layers: {unique_layers}")
            print(f"  Unfreezing layers: {layers_to_unfreeze}")

            unfrozen_count = 0
            for name, param in vision_backbone.named_parameters():
                for ln, layer_num in layer_names:
                    if name == ln and layer_num in layers_to_unfreeze:
                        param.requires_grad = True
                        unfrozen_count += 1
                        break
            print(f"  Unfrozen {unfrozen_count} parameters")
        else:
            print("  Could not identify layer structure, unfreezing last parameters by position")
            total_params = len(all_params)
            start_idx = max(0, total_params - int(total_params * unfreeze_layers / 10))
            for idx, (name, param) in enumerate(all_params):
                if idx >= start_idx:
                    param.requires_grad = True

        vision_backbone.train()

        if gradient_checkpointing:
            print("✓ Enabling gradient checkpointing")
            if hasattr(base_model, 'gradient_checkpointing_enable'):
                base_model.gradient_checkpointing_enable()
            elif hasattr(vision_backbone, 'gradient_checkpointing_enable'):
                vision_backbone.gradient_checkpointing_enable()

    else:
        raise ValueError(f"Unknown mode: {mode}")

    print(f"\nVision Backbone Parameters:")
    print_trainable_parameters(vision_backbone)

    return base_model, vision_backbone


# -----------------------
# Collate & vision forward
# -----------------------
# def _infer_grid_thw_from_pixels(pixel_values: torch.Tensor) -> torch.Tensor:
#     B, C, H, W = pixel_values.shape
#     for patch in (14, 16):
#         if H % patch == 0 and W % patch == 0:
#             gh, gw = H // patch, W // patch
#             return torch.tensor([[1, gh, gw]] * B, dtype=torch.int32, device=pixel_values.device)
#     gh, gw = max(1, round(H / 14)), max(1, round(W / 14))
#     return torch.tensor([[1, gh, gw]] * B, dtype=torch.int32, device=pixel_values.device)


def make_collate_dynamic(processor, min_pixels: int, max_pixels: int):
    """Dynamic resolution collate using min_pixels and max_pixels."""
    def collate(batch):
        imgs, ys, paths = zip(*batch)
        
        try:
            enc = processor(
                images=list(imgs),
                text=[""] * len(imgs),
                return_tensors="pt",
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )
        except TypeError as e:
            print(f"[ERROR] Processor doesn't support min_pixels/max_pixels: {e}")
            raise RuntimeError(
                "Qwen3-VL processor requires min_pixels and max_pixels parameters."
            )
        
        pixel = enc["pixel_values"]
        grid = enc.get("image_grid_thw", enc.get("grid_thw", None))
        
        # if grid is None:
        #     grid = _infer_grid_thw_from_pixels(pixel)
        if not torch.is_tensor(grid):
            grid = torch.tensor(grid)
        
        y = torch.tensor(ys, dtype=torch.long)
        return pixel, grid, y, paths
    
    return collate


def forward_vision_positional(vision_backbone, pixel_values: torch.Tensor,
                              grid_thw: torch.Tensor,
                              requires_grad: bool = False) -> torch.Tensor:
    """Forward pass with proper per-image feature aggregation."""
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
        batch_size = feats.shape[0]
        if batch_size == grid_thw.shape[0]:
            feats = feats.mean(dim=1)
        else:
            print(f"[WARN] Batch size mismatch: feats={batch_size}, grid={grid_thw.shape[0]}")
            feats = feats.mean(dim=1)
    
    elif feats.ndim == 2:
        first_dim = feats.shape[0]
        batch_size = grid_thw.shape[0]
        
        if first_dim == batch_size:
            pass
        elif first_dim % batch_size == 0:
            patches_per_image = first_dim // batch_size
            feats_reshaped = feats.reshape(batch_size, patches_per_image, -1)
            feats = feats_reshaped.mean(dim=1)
        else:
            print(f"[WARN] Uneven patches: {first_dim} patches for {batch_size} images")
            patches_per_image = (grid_thw[:, 1] * grid_thw[:, 2]).tolist()
            total_patches_from_grid = sum(patches_per_image)
            
            if first_dim == total_patches_from_grid:
                aggregated_feats = []
                start_idx = 0
                for num_patches in patches_per_image:
                    num_patches = int(num_patches)
                    end_idx = start_idx + num_patches
                    img_feats = feats[start_idx:end_idx]
                    img_feats_avg = img_feats.mean(dim=0, keepdim=True)
                    aggregated_feats.append(img_feats_avg)
                    start_idx = end_idx
                feats = torch.cat(aggregated_feats, dim=0)
            else:
                aggregated_feats = []
                start_idx = 0
                for grid_patches in patches_per_image:
                    proportion = grid_patches / total_patches_from_grid
                    actual_tokens = int(first_dim * proportion)
                    if start_idx + actual_tokens > first_dim:
                        actual_tokens = first_dim - start_idx
                    end_idx = start_idx + actual_tokens
                    img_feats = feats[start_idx:end_idx]
                    if img_feats.size(0) > 0:
                        img_feats_avg = img_feats.mean(dim=0, keepdim=True)
                    else:
                        img_feats_avg = torch.zeros(1, feats.size(1), device=feats.device, dtype=feats.dtype)
                    aggregated_feats.append(img_feats_avg)
                    start_idx = end_idx
                feats = torch.cat(aggregated_feats, dim=0)
    else:
        raise ValueError(f"Unexpected feature tensor shape: {feats.shape}")
    
    return feats


def get_logits_from_head(head, feats, mask=None):
    """Unified forward for both linear and MIL heads."""
    if isinstance(head, (CrossAttnMILHead, MILPrototypeHead)):
        # MIL head returns (logits, attention)
        logits, _ = head(feats, mask)
    else:
        # Linear head returns logits directly
        logits = head(feats)
    return logits

# -----------------------
# Train / Eval
# -----------------------
def train_one_epoch(vb, dl_train, head, optimizer, class_weights,
                    device="cuda", amp=True,
                    epoch=1, total_epochs=1, requires_grad=False, extractor=None, fusion_module=None):
    head.train()

    if fusion_module is not None:
        fusion_module.train()

    if requires_grad:
        vb.train()
    else:
        vb.eval()

    tot_loss = 0.0
    tot_correct = 0
    tot_seen = 0

    use_scaler = amp and device.startswith("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if use_scaler and dtype == torch.bfloat16:
        use_scaler = False
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    pbar = tqdm(dl_train, desc=f"[Epoch {epoch:02d}/{total_epochs:02d}] train", leave=False)

    for pixel_values, grid_thw, y, _paths in pbar:
        pixel_values = pixel_values.to(device, non_blocking=True)
        grid_thw = grid_thw.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=amp and device.startswith("cuda"), dtype=dtype):
            if extractor is not None:
                tapped_features, mask = extractor(pixel_values, grid_thw, requires_grad=requires_grad)
                feats = fusion_module(tapped_features, mask)
            else:
                feats = forward_vision_positional(vb, pixel_values, grid_thw, requires_grad=requires_grad)
                mask = None  # No mask for single-tap
            
            logits = get_logits_from_head(head, feats, mask)
            loss = F.cross_entropy(logits, y, weight=class_weights)

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = y.size(0)
        tot_loss += loss.item() * batch_size
        preds = logits.argmax(dim=1)
        tot_correct += (preds == y).sum().item()
        tot_seen += batch_size

        pbar.set_postfix({
            "loss": f"{tot_loss / max(1, tot_seen):.4f}",
            "acc": f"{tot_correct / max(1, tot_seen):.4f}"
        })

    return {
        "train_loss": tot_loss / max(1, tot_seen),
        "train_acc": tot_correct / max(1, tot_seen)
    }


@torch.no_grad()
def evaluate(vb, dl, head, device="cuda", desc="val", num_classes: int = None, extractor=None, fusion_module=None):
    head.eval()
    if fusion_module is not None:
        fusion_module.eval()
    vb.eval()
    all_y, all_p, all_probs = [], [], []
    tot_loss = 0.0
    tot_seen = 0

    if num_classes is None:
        num_classes = head.fc.out_features

    pbar = tqdm(dl, desc=f"[{desc}]", leave=False)

    for pixel_values, grid_thw, y, _paths in pbar:
        pixel_values = pixel_values.to(device, non_blocking=True)
        grid_thw = grid_thw.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        if extractor is not None:
            tapped_features, mask = extractor(pixel_values, grid_thw, requires_grad=False)
            feats = fusion_module(tapped_features, mask)
        else:
            feats = forward_vision_positional(vb, pixel_values, grid_thw, requires_grad=False)
            mask = None  # No mask for single-tap
        # feats = forward_vision_positional(vb, pixel_values, grid_thw, requires_grad=False)
        feats = feats.to(next(head.parameters()).dtype)
        logits = get_logits_from_head(head, feats, mask)
        loss = F.cross_entropy(logits, y)

        batch_size = y.size(0)
        tot_loss += loss.item() * batch_size
        tot_seen += batch_size
        preds = logits.argmax(dim=1)
        probs = F.softmax(logits, dim=1)

        all_y.append(y.detach().cpu())
        all_p.append(preds.detach().cpu())
        all_probs.append(probs.detach().cpu())

        pbar.set_postfix({"loss": f"{tot_loss / max(1, tot_seen):.4f}"})

    y_true = torch.cat(all_y, dim=0)
    y_pred = torch.cat(all_p, dim=0)
    y_probs = torch.cat(all_probs, dim=0)
    
    support = int(y_true.numel())
    correct = int((y_true == y_pred).sum().item())
    acc = correct / max(1, support)

    cm = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_prec = float(sum(precisions) / num_classes)
    macro_rec = float(sum(recalls) / num_classes)
    macro_f1 = float(sum(f1s) / num_classes)

    try:
        from sklearn.metrics import roc_auc_score
        y_true_np = y_true.numpy()
        y_probs_np = y_probs.numpy()
        
        per_class_auc = []
        for c in range(num_classes):
            y_binary = (y_true_np == c).astype(int)
            if y_binary.sum() > 0 and (1 - y_binary).sum() > 0:
                auc_c = roc_auc_score(y_binary, y_probs_np[:, c])
                per_class_auc.append(auc_c)
            else:
                per_class_auc.append(float('nan'))
        
        valid_aucs = [auc for auc in per_class_auc if not np.isnan(auc)]
        macro_auc = float(np.mean(valid_aucs)) if valid_aucs else float('nan')
        
        try:
            auc_ovr = roc_auc_score(y_true_np, y_probs_np, multi_class='ovr', average='weighted')
        except:
            auc_ovr = float('nan')
            
    except ImportError:
        print("Warning: sklearn not available, AUC scores will not be calculated")
        per_class_auc = [float('nan')] * num_classes
        macro_auc = float('nan')
        auc_ovr = float('nan')

    metrics = {
        "accuracy": acc,
        "loss": tot_loss / max(1, support),
        "support": support,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "macro_auc": macro_auc,
        "auc_ovr_weighted": auc_ovr,
        "per_class_f1": f1s,
        "per_class_auc": per_class_auc,
    }
    return metrics, cm


# -----------------------
# Dataset
# -----------------------
class MedMNISTNPZDataset(Dataset):
    """Generic MedMNIST dataset from NPZ file."""
    def __init__(self, npz_path: str, split: str = "train"):
        assert split in ["train", "val", "test"]
        self.npz_path = npz_path
        self.split = split

        data = np.load(npz_path)

        img_key = f"{split}_images"
        lbl_key = f"{split}_labels"
        if img_key not in data or lbl_key not in data:
            raise KeyError(f"NPZ file missing keys '{img_key}' / '{lbl_key}'")

        images = data[img_key]
        labels = data[lbl_key]

        labels = np.array(labels).squeeze()
        if labels.ndim != 1:
            raise ValueError(f"Labels must be 1D after squeeze, got shape {labels.shape}")

        self.images = images
        self.labels = labels.astype(np.int64)
        self.num_classes = int(self.labels.max() + 1)

        print(f"[MedMNIST] Loaded split='{split}' from {Path(npz_path).name}")
        print(f"  Samples: {len(self.labels)} | Classes: {self.num_classes}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        y = int(self.labels[idx])

        if img.ndim == 2:
            img_hwc = img[:, :, np.newaxis]
        elif img.ndim == 3:
            if img.shape[-1] in (1, 3):
                img_hwc = img
            elif img.shape[0] in (1, 3):
                img_hwc = np.transpose(img, (1, 2, 0))
            else:
                raise ValueError(f"Cannot infer channel dimension from shape {img.shape}")
        else:
            raise ValueError(f"Expected 2D or 3D image, got shape {img.shape}")

        if img_hwc.dtype != np.uint8:
            min_v, max_v = img_hwc.min(), img_hwc.max()
            if max_v <= 1.0:
                img_hwc = (img_hwc * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img_hwc = img_hwc.clip(0, 255).astype(np.uint8)

        if img_hwc.shape[-1] == 1:
            img_hwc = img_hwc[:, :, 0]
            pil_img = Image.fromarray(img_hwc).convert("RGB")
        else:
            pil_img = Image.fromarray(img_hwc)

        pseudo_path = f"{self.split}_{idx}"
        return pil_img, y, pseudo_path


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
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Train Qwen3-VL on MedMNIST")
    
    # Simple dataset argument
    parser.add_argument("--medmnist", type=str, required=True,
                        choices=list(MEDMNIST_INFO.keys()),
                        help="MedMNIST dataset name (e.g., pathmnist, tissuemnist)")
    parser.add_argument("--data_root", type=str, default="/scratch/cui0011/medmnist",
                        help="Directory containing NPZ files")
    parser.add_argument("--out_dir", type=str, default="/scratch/cui0011/medmnist/multi_tap_try",
                        help="Output directory (auto-generated if not provided)")
    
    # Model
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--use_multi_tap", action="store_true",
                        help="Use MultiTapExtractor for multi-layer feature extraction")
    parser.add_argument("--tap_layers", nargs = "+" ,type=int, default= [6, 13, 20],
                        help="Which layers to tap (e.g., 6 13 20 for early/mid/late)")
    parser.add_argument("--project_to_dim", type=int, default=4096, 
                        help="Project tapped features to this dimension (if using multi-tap)")
    parser.add_argument("--fusion_type", type=str, default="token_attention",
                    choices=["scalar", "token_attention", "gated"],
                    help="Depth fusion strategy")
    parser.add_argument("--pooling_type", type=str, default="mean",
                    choices=["mean", "max", "attention", "cls"],
                    help="Pooling strategy")
    parser.add_argument("--use_cross_attention", action="store_true",
                    help="Use cross-attention reasoning blocks")
    parser.add_argument("--num_cross_attn_blocks", type=int, default=1,
                        help="Number of cross-attention blocks (1-2 recommended)")
    parser.add_argument("--cross_attn_heads", type=int, default=8,
                        help="Number of attention heads in cross-attention")

    # MIL configuration  
    parser.add_argument("--use_mil", action="store_true",
                        help="Use MIL prototype head instead of linear head")
    parser.add_argument("--num_prototypes", type=int, default=2,
                        help="Number of prototypes per class for MIL")
    parser.add_argument("--prototype_aggregation", type=str, default="max",
                        choices=["max", "mean", "mlp"],
                        help="How to aggregate prototypes in MIL")

    # Resolution
    parser.add_argument("--min_pixels", type=int, default=224*224)
    parser.add_argument("--max_pixels", type=int, default=224*224)
    
    # Training
    parser.add_argument("--mode", type=str, default="frozen",
                        choices=["frozen", "partial_finetune", "full_finetune"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--backbone_lr", type=float, default=2e-5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    
    # Fine-tuning
    parser.add_argument("--unfreeze_layers", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--use_class_weights", action="store_true")

    args = parser.parse_args()

    # Setup paths
    dataset_name = args.medmnist
    npz_path = Path(args.data_root) / f"{dataset_name}_224.npz"
    
    if args.out_dir is None:
        out_dir = Path(args.data_root) / f"{dataset_name}_{args.mode}_run"
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get dataset info
    dataset_info = MEDMNIST_INFO[dataset_name]
    num_classes = dataset_info["num_classes"]
    class_names = dataset_info["class_names"]
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*70}")
    print(f"Training Qwen3-VL on {dataset_name.upper()}")
    print(f"{'='*70}")
    print(f"NPZ: {npz_path}")
    print(f"Output: {out_dir}")
    print(f"Classes: {num_classes}")
    print(f"Mode: {args.mode}")
    print(f"Device: {device}")

    # Datasets
    ds_train = MedMNISTNPZDataset(str(npz_path), split="train")
    ds_val = MedMNISTNPZDataset(str(npz_path), split="val")
    ds_test = MedMNISTNPZDataset(str(npz_path), split="test")

    # Class weights
    class_counts = np.bincount(ds_train.labels, minlength=num_classes)
    total = class_counts.sum()
    
    if args.use_class_weights:
        class_weights = []
        for c in range(num_classes):
            cnt = max(1, int(class_counts[c]))
            w_c = total / (num_classes * cnt)
            class_weights.append(w_c)
        class_weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
        print(f"\nClass weights: {class_weights.cpu().tolist()}")
    else:
        class_weights = None
        print(f"\nClass counts: {class_counts.tolist()}")

    # Load model
    print("\nLoading Qwen3-VL...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    vision_backbone = find_vision_backbone(base_model)
    print(f"✓ Loaded {args.model_id}")

    # Configure training
    base_model, vision_backbone = configure_model_training(
        base_model, vision_backbone,
        mode=args.mode,
        unfreeze_layers=args.unfreeze_layers,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    extractor = None
    fusion_module = None
    if args.use_multi_tap:
        print(f"\n{'='*60}")
        print(f"Initializing Multi-Tap Extractor")
        print(f"{'='*60}")
        extractor = MultiTapExtractor(
            vision_backbone=vision_backbone,
            tap_layers=args.tap_layers,
            project_to_dim=args.project_to_dim if args.use_multi_tap else None,
            use_layernorm=True
        )
        print(f"✓ Multi-tap extractor ready")
        
        # Create fusion module
        print(f"\n{'='*60}")
        print(f"Initializing Depth Fusion Module")
        print(f"{'='*60}")
        print(f"  Fusion type: {args.fusion_type}")
        
        if args.use_mil:
            # MIL does its own pooling, so only use fusion (no pooling)
            from depth_fusion import TokenWiseDepthFusion
            
            fusion_module = TokenWiseDepthFusion(
                hidden_size=args.project_to_dim,
                num_taps=len(args.tap_layers),
                fusion_type=args.fusion_type,
                use_layernorm=True
            ).to(device)
            print(f"  MIL mode: fusion only (no pooling)")
        else:
            # Linear head needs pooled features
            print(f"  Pooling type: {args.pooling_type}")
            
            fusion_module = FusionAndPooling(
                hidden_size=args.project_to_dim,
                num_taps=len(args.tap_layers),
                fusion_type=args.fusion_type,
                pooling_type=args.pooling_type,
                use_layernorm=True
            ).to(device)
        
        # Count fusion parameters
        fusion_params = sum(p.numel() for p in fusion_module.parameters())
        print(f"  Fusion parameters: {fusion_params:,}")
        print(f"✓ Fusion module ready")

    collate = make_collate_dynamic(processor, args.min_pixels, args.max_pixels)

    # Infer hidden size
    print("\nInferring feature dimension...")
    dummy_loader = DataLoader(ds_train, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate)
    pixel_dummy, grid_dummy, _, _ = next(iter(dummy_loader))
    pixel_dummy = pixel_dummy.to(device)
    grid_dummy = grid_dummy.to(device)
    # with torch.no_grad():
    #     feats = forward_vision_positional(vision_backbone, pixel_dummy, grid_dummy, requires_grad=False)
    # hidden_size = feats.shape[-1]
    # print(f"✓ Hidden size: {hidden_size}")
    if args.use_multi_tap:
        with torch.no_grad():
            tapped_features, mask = extractor(pixel_dummy, grid_dummy, requires_grad=False)
            # Apply fusion
            feats = fusion_module(tapped_features, mask)
        hidden_size = feats.shape[-1]
        print(f"✓ Multi-tap + fusion mode: {len(tapped_features)} taps → fused features")
    else:
        with torch.no_grad():
            feats = forward_vision_positional(vision_backbone, pixel_dummy, grid_dummy, requires_grad=False)
        hidden_size = feats.shape[-1]

    print(f"✓ Hidden size: {hidden_size}")  

    requires_grad = args.mode in ["partial_finetune", "full_finetune"]

    # DataLoaders
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                         num_workers=args.num_workers, pin_memory=True, collate_fn=collate)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                       num_workers=args.num_workers, pin_memory=True, collate_fn=collate)
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, collate_fn=collate)

    # Head
    head = LinearHead(in_dim=hidden_size, num_classes=num_classes).to(device)

    if args.use_mil:
        # MIL head (with optional cross-attention)
        print(f"\n{'='*60}")
        print(f"Initializing MIL Prototype Head")
        print(f"{'='*60}")
        print(f"  Classes: {num_classes}")
        print(f"  Prototypes per class: {args.num_prototypes}")
        print(f"  Aggregation: {args.prototype_aggregation}")
        if args.use_cross_attention:
            print(f"  Cross-attention blocks: {args.num_cross_attn_blocks}")
            print(f"  Cross-attention heads: {args.cross_attn_heads}")
        
        head = CrossAttnMILHead(
            hidden_size=hidden_size,
            num_classes=num_classes,
            use_cross_attention=args.use_cross_attention,
            num_cross_attn_blocks=args.num_cross_attn_blocks,
            num_cross_attn_heads=args.cross_attn_heads,
            cross_attn_dropout=0.1,
            num_prototypes=args.num_prototypes,
            prototype_aggregation=args.prototype_aggregation,
            mil_dropout=0.1,
        ).to(device)
        
        mil_params = sum(p.numel() for p in head.parameters())
        print(f"  Head parameters: {mil_params:,}")
        print(f"✓ MIL head ready")

    else:
        # Simple linear head (original)
        print(f"\n{'='*60}")
        print(f"Initializing Linear Head")
        print(f"{'='*60}")
        head = LinearHead(in_dim=hidden_size, num_classes=num_classes).to(device)
        print(f"✓ Linear head ready")


    # Optimizer
    if requires_grad:
        param_groups = [
            {'params': head.parameters(), 'lr': args.lr},
            {'params': [p for p in vision_backbone.parameters() if p.requires_grad],
            'lr': args.backbone_lr}
        ]
        if fusion_module is not None:
            param_groups.append({'params': fusion_module.parameters(), 'lr': args.lr})
        optimizer = torch.optim.Adam(param_groups)
    else:
        params = list(head.parameters())
        if fusion_module is not None:
            params += list(fusion_module.parameters())
        optimizer = torch.optim.Adam(params, lr=args.lr)

    # Train
    best_val_score = -1.0
    best_state = None
    history = []

    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(
            vision_backbone, dl_train, head, optimizer, class_weights,
            device=device, amp=True, epoch=epoch, total_epochs=args.epochs,
            requires_grad=requires_grad, extractor=extractor, fusion_module=fusion_module
        )
        print(f"[Epoch {epoch:02d}] train_loss={stats['train_loss']:.4f} "
              f"train_acc={stats['train_acc']:.4f}")

        val_metrics, val_cm = evaluate(
            vision_backbone, dl_val, head,
            device=device, desc="val", num_classes=num_classes, extractor=extractor, fusion_module=fusion_module
        )
        print(f"[Epoch {epoch:02d}] VAL "
              f"acc={val_metrics['accuracy']:.4f} "
              f"f1={val_metrics['macro_f1']:.4f} "
              f"auc={val_metrics['macro_auc']:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": stats['train_loss'],
            "train_acc": stats['train_acc'],
            "val_loss": val_metrics['loss'],
            "val_acc": val_metrics['accuracy'],
            "val_macro_f1": val_metrics['macro_f1'],
            "val_macro_auc": val_metrics['macro_auc'],
        })

        if val_metrics["accuracy"] > best_val_score:
            best_val_score = val_metrics["accuracy"]
            best_state = {
                "head": head.state_dict(),
                "val_metrics": val_metrics,
                "val_cm": val_cm.clone(),
                "epoch": epoch,
            }
            checkpoint_dict = {
                "head_state_dict": head.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
            }
            if requires_grad:
                checkpoint_dict["vision_backbone_state_dict"] = vision_backbone.state_dict()
            torch.save(checkpoint_dict, str(out_dir / "best_model.pt"))

    # Load best and test
    if best_state is not None:
        head.load_state_dict(best_state["head"])

    test_metrics, test_cm = evaluate(
        vision_backbone, dl_test, head,
        device=device, desc="test", num_classes=num_classes, extractor=extractor, fusion_module=fusion_module
    )
    
    print(f"\n{'='*70}")
    print(f"TEST RESULTS - {dataset_name.upper()}")
    print(f"{'='*70}")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Macro F1:  {test_metrics['macro_f1']:.4f}")
    print(f"Macro AUC: {test_metrics['macro_auc']:.4f}")

    # Save checkpoint
    save_dict = {
        "head_state_dict": head.state_dict(),
        "hidden_size": hidden_size,
        "classes": class_names,
        "model_id": args.model_id,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "training_mode": args.mode,
        "num_classes": num_classes,
        "dataset": dataset_name,
        "best_val": {
            "epoch": best_state["epoch"] if best_state is not None else None,
            "val_metrics": best_state["val_metrics"] if best_state is not None else None,
            "val_confusion_matrix": best_state["val_cm"].tolist() if best_state is not None else None,
        },
        "test_metrics": test_metrics,
        "test_confusion_matrix": test_cm.tolist(),
    }

    if fusion_module is not None:
        save_dict["fusion_module_state_dict"] = fusion_module.state_dict()
        save_dict["fusion_type"] = args.fusion_type
        save_dict["pooling_type"] = args.pooling_type

    if args.use_mil:
        save_dict["use_mil"] = True
        save_dict["num_prototypes"] = args.num_prototypes
        save_dict["prototype_aggregation"] = args.prototype_aggregation
    if args.use_cross_attention:
        save_dict["use_cross_attention"] = True
        save_dict["num_cross_attn_blocks"] = args.num_cross_attn_blocks

    if requires_grad:
        save_dict["vision_backbone_state_dict"] = {
            k: v for k, v in vision_backbone.state_dict().items()
            if any(p.requires_grad for p in vision_backbone.parameters())
        }
    
    torch.save(save_dict, str(out_dir / "checkpoint.pt"))
    
    # Save training history to text file
    history_path = out_dir / "training_history.txt"
    with open(history_path, "w") as f:
        f.write(f"{dataset_name.upper()} Classification Training History\n")
        f.write(f"{'='*80}\n\n")
        f.write("Configuration:\n")
        f.write(f"  Dataset: {dataset_name}\n")
        f.write(f"  Model: {args.model_id}\n")
        f.write(f"  Training mode: {args.mode}\n")
        f.write(f"  Resolution: Dynamic (min={args.min_pixels:,}, max={args.max_pixels:,})\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Head learning rate: {args.lr}\n")
        if requires_grad:
            f.write(f"  Backbone learning rate: {args.backbone_lr}\n")
        if args.mode == "partial_finetune":
            f.write(f"  Unfrozen layers: {args.unfreeze_layers}\n")
        f.write(f"  Gradient checkpointing: {args.gradient_checkpointing}\n")
        f.write(f"  Use class weights: {args.use_class_weights}\n")
        f.write(f"  Epochs: {args.epochs}\n\n")
        if args.use_multi_tap:
            f.write("Multi-Tap Extractor:\n")
            f.write(f"  Tap layers: {args.tap_layers}\n")
            f.write(f"  Project to dim: {args.project_to_dim}\n")
            f.write(f"  Fusion type: {args.fusion_type}\n")
            f.write(f"  Pooling type: {args.pooling_type}\n\n")
        if args.use_mil:
            f.write("MIL Head Configuration:\n")
            f.write(f"  Prototypes per class: {args.num_prototypes}\n")
            f.write(f"  Prototype aggregation: {args.prototype_aggregation}\n")
            if args.use_cross_attention:
                f.write(f"  Cross-attention blocks: {args.num_cross_attn_blocks}\n")
                f.write(f"  Cross-attention heads: {args.cross_attn_heads}\n")
            f.write("\n")
        
        f.write("Training History:\n")
        f.write(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'Val Loss':>12} {'Val Acc':>10} {'Val F1':>10} {'Val AUC':>10}\n")
        f.write(f"{'-'*90}\n")
        for h in history:
            f.write(f"{h['epoch']:>6} {h['train_loss']:>12.4f} {h['train_acc']:>10.4f} "
                   f"{h['val_loss']:>12.4f} {h['val_acc']:>10.4f} {h['val_macro_f1']:>10.4f} {h['val_macro_auc']:>10.4f}\n")
        
        f.write(f"\n{'='*90}\n")
        f.write(f"Best Validation Accuracy: {best_val_score:.4f} (Epoch {best_state['epoch'] if best_state else 'N/A'})\n")
        f.write(f"\nTest Results:\n")
        f.write(f"  Accuracy:        {test_metrics['accuracy']:.4f}\n")
        f.write(f"  Macro Precision: {test_metrics['macro_precision']:.4f}\n")
        f.write(f"  Macro Recall:    {test_metrics['macro_recall']:.4f}\n")
        f.write(f"  Macro F1:        {test_metrics['macro_f1']:.4f}\n")
        f.write(f"  Macro AUC:       {test_metrics['macro_auc']:.4f}\n")
        f.write(f"  AUC (OvR):       {test_metrics['auc_ovr_weighted']:.4f}\n")
        f.write(f"  Loss:            {test_metrics['loss']:.4f}\n")
        f.write(f"  Support:         {test_metrics['support']}\n")
        
        f.write(f"\nPer-class F1 Scores:\n")
        for i, f1 in enumerate(test_metrics['per_class_f1']):
            f.write(f"  {class_names[i]:40s}: {f1:.4f}\n")
        
        f.write(f"\nPer-class AUC Scores:\n")
        for i, auc in enumerate(test_metrics['per_class_auc']):
            if not np.isnan(auc):
                f.write(f"  {class_names[i]:40s}: {auc:.4f}\n")
            else:
                f.write(f"  {class_names[i]:40s}: N/A\n")
    
    # Save JSON metrics
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "dataset": dataset_name,
            "num_classes": num_classes,
            "class_names": class_names,
            "test_metrics": test_metrics,
            "test_confusion_matrix": test_cm.tolist(),
            "best_val": {
                "epoch": best_state["epoch"] if best_state is not None else None,
                "val_metrics": best_state["val_metrics"] if best_state is not None else None,
                "val_confusion_matrix": best_state["val_cm"].tolist() if best_state is not None else None,
            },
            "training_config": {
                "mode": args.mode,
                "model_id": args.model_id,
                "backbone_lr": args.backbone_lr if requires_grad else None,
                "head_lr": args.lr,
                "unfreeze_layers": args.unfreeze_layers if args.mode == "partial_finetune" else None,
                "gradient_checkpointing": args.gradient_checkpointing,
                "use_class_weights": args.use_class_weights,
                "batch_size": args.batch_size,
                "min_pixels": args.min_pixels,
                "max_pixels": args.max_pixels,
            }
        }, f, indent=2)

    print(f"\n✓ Saved checkpoint to {out_dir/'checkpoint.pt'}")
    print(f"✓ Saved best model to {out_dir/'best_model.pt'}")
    print(f"✓ Saved training history to {history_path}")
    print(f"✓ Saved metrics to {out_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
