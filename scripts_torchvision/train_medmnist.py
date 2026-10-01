#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Baseline CNN and ViT training on MedMNIST datasets.

Supports: ResNet50, DenseNet121, ConvNeXt-Tiny, Swin-Tiny, ViT-Base, EfficientNet-B0

Usage:
    python medmnist_baselines.py --medmnist pathmnist --model resnet50
    python medmnist_baselines.py --medmnist tissuemnist --model convnext_tiny --epochs 20
    python medmnist_baselines.py --medmnist octmnist --model swin_tiny --image_size 224
    python medmnist_baselines.py --medmnist pathmnist --model resnet50 --train_fraction 0.1
"""

import os
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
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm


# -----------------------
# Dataset info (same as original)
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
# Model registry
# -----------------------
AVAILABLE_MODELS = {
    "resnet50": "ResNet-50",
    "resnet101": "ResNet-101",
    "densenet121": "DenseNet-121",
    "densenet169": "DenseNet-169",
    "convnext_tiny": "ConvNeXt-Tiny",
    "convnext_small": "ConvNeXt-Small",
    "convnext_base": "ConvNeXt-Base",
    "swin_tiny": "Swin-Tiny",
    "swin_small": "Swin-Small",
    "swin_base": "Swin-Base",
    "vit_base": "ViT-Base/16",
    "vit_large": "ViT-Large/16",
    "efficientnet_b0": "EfficientNet-B0",
    "efficientnet_b3": "EfficientNet-B3",
    "inception_v3": "Inception-v3",
}


def load_model(model_name: str, num_classes: int, pretrained: bool = True):
    """Load model with appropriate classification head."""
    from torchvision import models

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(AVAILABLE_MODELS.keys())}")

    print(f"\nLoading {AVAILABLE_MODELS[model_name]}...")

    weights = "DEFAULT" if pretrained else None

    if model_name == "resnet50":
        model = models.resnet50(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Linear(feature_dim, num_classes)

    elif model_name == "resnet101":
        model = models.resnet101(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Linear(feature_dim, num_classes)

    elif model_name == "densenet121":
        model = models.densenet121(weights=weights)
        feature_dim = model.classifier.in_features
        model.classifier = nn.Linear(feature_dim, num_classes)

    elif model_name == "densenet169":
        model = models.densenet169(weights=weights)
        feature_dim = model.classifier.in_features
        model.classifier = nn.Linear(feature_dim, num_classes)

    elif model_name == "convnext_tiny":
        model = models.convnext_tiny(weights=weights)
        feature_dim = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(feature_dim, num_classes)

    elif model_name == "convnext_small":
        model = models.convnext_small(weights=weights)
        feature_dim = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(feature_dim, num_classes)

    elif model_name == "convnext_base":
        model = models.convnext_base(weights=weights)
        feature_dim = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(feature_dim, num_classes)

    elif model_name == "swin_tiny":
        model = models.swin_t(weights=weights)
        feature_dim = model.head.in_features
        model.head = nn.Linear(feature_dim, num_classes)

    elif model_name == "swin_small":
        model = models.swin_s(weights=weights)
        feature_dim = model.head.in_features
        model.head = nn.Linear(feature_dim, num_classes)

    elif model_name == "swin_base":
        model = models.swin_b(weights=weights)
        feature_dim = model.head.in_features
        model.head = nn.Linear(feature_dim, num_classes)

    elif model_name == "vit_base":
        model = models.vit_b_16(weights=weights)
        feature_dim = model.heads.head.in_features
        model.heads.head = nn.Linear(feature_dim, num_classes)

    elif model_name == "vit_large":
        model = models.vit_l_16(weights=weights)
        feature_dim = model.heads.head.in_features
        model.heads.head = nn.Linear(feature_dim, num_classes)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        feature_dim = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(feature_dim, num_classes)

    elif model_name == "efficientnet_b3":
        model = models.efficientnet_b3(weights=weights)
        feature_dim = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(feature_dim, num_classes)

    elif model_name == "inception_v3":
        model = models.inception_v3(weights=weights)
        feature_dim = model.fc.in_features
        model.fc = nn.Linear(feature_dim, num_classes)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        # model.aux_logits = False

    else:
        raise ValueError(f"Model {model_name} not implemented")

    print(f"✓ Loaded {AVAILABLE_MODELS[model_name]}")
    print(f"  Feature dimension: {feature_dim}")

    return model, feature_dim


def count_parameters(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def print_trainable_parameters(model):
    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model, trainable_only=False)
    print(f"Trainable params: {trainable:,} || Total params: {total:,} || Trainable%: {100 * trainable / total:.2f}%")


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
class MedMNISTNPZDataset(Dataset):
    """Generic MedMNIST dataset from NPZ file."""
    def __init__(self, npz_path: str, split: str = "train", transform=None):
        assert split in ["train", "val", "test"]
        self.npz_path = npz_path
        self.split = split
        self.transform = transform

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
            if img_hwc.max() <= 1.0:
                img_hwc = (img_hwc * 255.0).clip(0, 255).astype(np.uint8)
            else:
                img_hwc = img_hwc.clip(0, 255).astype(np.uint8)

        if img_hwc.shape[-1] == 1:
            img_hwc = img_hwc[:, :, 0]
            pil_img = Image.fromarray(img_hwc).convert("RGB")
        else:
            pil_img = Image.fromarray(img_hwc)

        if self.transform:
            pil_img = self.transform(pil_img)

        pseudo_path = f"{self.split}_{idx}"
        return pil_img, y, pseudo_path


# -----------------------
# Training & Evaluation
# -----------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device="cuda",
                    amp=True, epoch=1, total_epochs=1, scheduler=None):
    model.train()

    tot_loss = 0.0
    tot_correct = 0
    tot_seen = 0

    use_scaler = amp and device.startswith("cuda")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    if use_scaler and dtype == torch.bfloat16:
        use_scaler = False
    scaler = torch.amp.GradScaler('cuda', enabled=use_scaler)

    pbar = tqdm(dataloader, desc=f"[Epoch {epoch:02d}/{total_epochs:02d}] train", leave=False)

    for images, labels, _paths in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=amp and device.startswith("cuda"), dtype=dtype):
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0] 
            loss = criterion(outputs, labels)

        if use_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batch_size = labels.size(0)
        tot_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        tot_correct += (preds == labels).sum().item()
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
def evaluate(model, dataloader, device="cuda", desc="val", num_classes=None):
    model.eval()

    all_y, all_p, all_probs = [], [], []
    tot_loss = 0.0
    tot_seen = 0

    if num_classes is None:
        if hasattr(model, 'num_classes'):
            num_classes = model.num_classes
        elif hasattr(model, 'head') and hasattr(model.head, 'out_features'):
            num_classes = model.head.out_features
        elif hasattr(model, 'fc') and hasattr(model.fc, 'out_features'):
            num_classes = model.fc.out_features
        else:
            num_classes = 10

    pbar = tqdm(dataloader, desc=f"[{desc}]", leave=False)

    for images, labels, _paths in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)

        batch_size = labels.size(0)
        tot_loss += loss.item() * batch_size
        tot_seen += batch_size

        preds = outputs.argmax(dim=1)
        probs = F.softmax(outputs, dim=1)

        all_y.append(labels.detach().cpu())
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
        except Exception:
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
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Train baseline models on MedMNIST")

    # Dataset
    parser.add_argument("--medmnist", type=str, required=True,
                        choices=list(MEDMNIST_INFO.keys()),
                        help="MedMNIST dataset name")
    parser.add_argument("--data_root", type=str, default="/scratch/cui0011/medmnist",
                        help="Directory containing NPZ files")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output directory (auto-generated if not provided)")
    parser.add_argument("--train_fraction", type=float, default=1.0,
                        help="Fraction of training data to use per class, e.g. 0.1 for 10%% (stratified).")

    # Model
    parser.add_argument("--model", type=str, required=True,
                        choices=list(AVAILABLE_MODELS.keys()),
                        help="Model architecture")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pretrained weights")
    parser.add_argument("--no_pretrained", dest="pretrained", action="store_false",
                        help="Train from scratch")

    # Image processing
    parser.add_argument("--image_size", type=int, default=224,
                        help="Input image size")
    parser.add_argument("--augmentation", action="store_true",
                        help="Use data augmentation for training")

    # Training
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--warmup_epochs", type=int, default=0)
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # Optimizer
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adam", "adamw", "sgd"])
    parser.add_argument("--scheduler", type=str, default="none",
                        choices=["cosine", "step", "none"])

    args = parser.parse_args()

    # Validate train_fraction
    if not (0.0 < args.train_fraction <= 1.0):
        raise ValueError(f"--train_fraction must be in (0, 1], got {args.train_fraction}")

    # Setup paths
    dataset_name = args.medmnist
    npz_path = Path(args.data_root) / f"{dataset_name}_224.npz"

    if args.out_dir is None:
        pretrain_str = "pretrained" if args.pretrained else "scratch"
        out_dir = Path(args.data_root) / f"{dataset_name}_{args.model}_{pretrain_str}"
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
    print(f"Training {AVAILABLE_MODELS[args.model]} on {dataset_name.upper()}")
    print(f"{'='*70}")
    print(f"NPZ: {npz_path}")
    print(f"Output: {out_dir}")
    print(f"Classes: {num_classes}")
    print(f"Pretrained: {args.pretrained}")
    print(f"Image size: {args.image_size}")
    print(f"Device: {device}")
    print(f"Train fraction: {args.train_fraction*100:.1f}%")

    # Transforms
    if args.augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Datasets
    ds_train = MedMNISTNPZDataset(str(npz_path), split="train", transform=train_transform)
    ds_val   = MedMNISTNPZDataset(str(npz_path), split="val",   transform=test_transform)
    ds_test  = MedMNISTNPZDataset(str(npz_path), split="test",  transform=test_transform)

    # -------------------------------------------------------
    # Subset training data (stratified) if fraction < 1.0
    # -------------------------------------------------------
    if args.train_fraction < 1.0:
        n_total = len(ds_train)
        rng = np.random.default_rng(args.seed)
        indices = []
        for c in range(num_classes):
            class_indices = np.where(ds_train.labels == c)[0]
            n_keep = max(1, int(len(class_indices) * args.train_fraction))
            chosen = rng.choice(class_indices, size=n_keep, replace=False).tolist()
            indices.extend(chosen)
        indices = sorted(indices)
        ds_train = torch.utils.data.Subset(ds_train, indices)
        ds_train.labels = np.array([ds_train.dataset.labels[i] for i in indices])
        print(f"  → Subsampled to {len(indices)}/{n_total} training samples "
              f"({args.train_fraction*100:.1f}%, stratified per class)")

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
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        print(f"\nClass counts (train): {class_counts.tolist()}")
        criterion = nn.CrossEntropyLoss()

    # DataLoaders
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)
    dl_test  = DataLoader(ds_test,  batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers, pin_memory=True)

    # Load model
    model, feature_dim = load_model(args.model, num_classes, pretrained=args.pretrained)
    model = model.to(device)
    print_trainable_parameters(model)

    # Optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    # Scheduler
    total_steps  = len(dl_train) * args.epochs
    warmup_steps = len(dl_train) * args.warmup_epochs

    if args.scheduler == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    elif args.scheduler == "step":
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=args.epochs // 3, gamma=0.1)
    else:
        scheduler = None

    # Training loop
    best_val_score = -1.0
    best_state     = None
    history        = []

    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model, dl_train, optimizer, criterion,
            device=device, amp=True, epoch=epoch, total_epochs=args.epochs,
            scheduler=scheduler if args.scheduler == "cosine" else None,
        )

        if args.scheduler == "step" and scheduler is not None:
            scheduler.step()

        print(f"[Epoch {epoch:02d}] train_loss={train_stats['train_loss']:.4f} "
              f"train_acc={train_stats['train_acc']:.4f}")

        val_metrics, val_cm = evaluate(model, dl_val, device=device, desc="val", num_classes=num_classes)
        print(f"[Epoch {epoch:02d}] VAL "
              f"acc={val_metrics['accuracy']:.4f} "
              f"f1={val_metrics['macro_f1']:.4f} "
              f"auc={val_metrics['macro_auc']:.4f}")

        history.append({
            "epoch"        : epoch,
            "train_loss"   : train_stats['train_loss'],
            "train_acc"    : train_stats['train_acc'],
            "val_loss"     : val_metrics['loss'],
            "val_acc"      : val_metrics['accuracy'],
            "val_macro_f1" : val_metrics['macro_f1'],
            "val_macro_auc": val_metrics['macro_auc'],
        })

        if val_metrics["accuracy"] > best_val_score:
            best_val_score = val_metrics["accuracy"]
            best_state = {
                "model"      : model.state_dict(),
                "val_metrics": val_metrics,
                "val_cm"     : val_cm.clone(),
                "epoch"      : epoch,
            }
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch"           : epoch,
                "val_metrics"     : val_metrics,
            }, str(out_dir / "best_model.pt"))

    # Load best and test
    if best_state is not None:
        model.load_state_dict(best_state["model"])

    test_metrics, test_cm = evaluate(model, dl_test, device=device, desc="test", num_classes=num_classes)

    print(f"\n{'='*70}")
    print(f"TEST RESULTS - {dataset_name.upper()} - {AVAILABLE_MODELS[args.model]}")
    print(f"{'='*70}")
    print(f"Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Macro F1:  {test_metrics['macro_f1']:.4f}")
    print(f"Macro AUC: {test_metrics['macro_auc']:.4f}")

    # Save checkpoint
    save_dict = {
        "model_state_dict" : model.state_dict(),
        "model_name"       : args.model,
        "model_full_name"  : AVAILABLE_MODELS[args.model],
        "feature_dim"      : feature_dim,
        "classes"          : class_names,
        "num_classes"      : num_classes,
        "dataset"          : dataset_name,
        "pretrained"       : args.pretrained,
        "image_size"       : args.image_size,
        "train_fraction"   : args.train_fraction,
        "best_val": {
            "epoch"               : best_state["epoch"] if best_state is not None else None,
            "val_metrics"         : best_state["val_metrics"] if best_state is not None else None,
            "val_confusion_matrix": best_state["val_cm"].tolist() if best_state is not None else None,
        },
        "test_metrics"          : test_metrics,
        "test_confusion_matrix" : test_cm.tolist(),
    }
    torch.save(save_dict, str(out_dir / "checkpoint.pt"))

    # Save training history
    history_path = out_dir / "training_history.txt"
    with open(history_path, "w") as f:
        f.write(f"{dataset_name.upper()} - {AVAILABLE_MODELS[args.model]} Training History\n")
        f.write(f"{'='*80}\n\n")
        f.write("Configuration:\n")
        f.write(f"  Dataset: {dataset_name}\n")
        f.write(f"  Model: {AVAILABLE_MODELS[args.model]}\n")
        f.write(f"  Pretrained: {args.pretrained}\n")
        f.write(f"  Train fraction: {args.train_fraction*100:.1f}%\n")
        f.write(f"  Image size: {args.image_size}\n")
        f.write(f"  Batch size: {args.batch_size}\n")
        f.write(f"  Learning rate: {args.lr}\n")
        f.write(f"  Weight decay: {args.weight_decay}\n")
        f.write(f"  Optimizer: {args.optimizer}\n")
        f.write(f"  Scheduler: {args.scheduler}\n")
        f.write(f"  Warmup epochs: {args.warmup_epochs}\n")
        f.write(f"  Augmentation: {args.augmentation}\n")
        f.write(f"  Class weights: {args.use_class_weights}\n")
        f.write(f"  Epochs: {args.epochs}\n")
        f.write(f"  Feature dimension: {feature_dim}\n")
        f.write(f"  Total parameters: {count_parameters(model, trainable_only=False):,}\n")
        f.write(f"  Trainable parameters: {count_parameters(model, trainable_only=True):,}\n\n")

        f.write("Training History:\n")
        f.write(f"{'Epoch':>6} {'Train Loss':>12} {'Train Acc':>10} {'Val Loss':>12} "
                f"{'Val Acc':>10} {'Val F1':>10} {'Val AUC':>10}\n")
        f.write(f"{'-'*90}\n")
        for h in history:
            f.write(f"{h['epoch']:>6} {h['train_loss']:>12.4f} {h['train_acc']:>10.4f} "
                    f"{h['val_loss']:>12.4f} {h['val_acc']:>10.4f} "
                    f"{h['val_macro_f1']:>10.4f} {h['val_macro_auc']:>10.4f}\n")

        f.write(f"\n{'='*90}\n")
        f.write(f"Best Validation Accuracy: {best_val_score:.4f} "
                f"(Epoch {best_state['epoch'] if best_state else 'N/A'})\n")
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
            "dataset"              : dataset_name,
            "model"                : args.model,
            "model_full_name"      : AVAILABLE_MODELS[args.model],
            "num_classes"          : num_classes,
            "class_names"          : class_names,
            "train_fraction"       : args.train_fraction,
            "test_metrics"         : test_metrics,
            "test_confusion_matrix": test_cm.tolist(),
            "best_val": {
                "epoch"               : best_state["epoch"] if best_state is not None else None,
                "val_metrics"         : best_state["val_metrics"] if best_state is not None else None,
                "val_confusion_matrix": best_state["val_cm"].tolist() if best_state is not None else None,
            },
            "training_config": {
                "pretrained"       : args.pretrained,
                "image_size"       : args.image_size,
                "batch_size"       : args.batch_size,
                "lr"               : args.lr,
                "weight_decay"     : args.weight_decay,
                "optimizer"        : args.optimizer,
                "scheduler"        : args.scheduler,
                "warmup_epochs"    : args.warmup_epochs,
                "augmentation"     : args.augmentation,
                "use_class_weights": args.use_class_weights,
                "epochs"           : args.epochs,
                "train_fraction"   : args.train_fraction,
            }
        }, f, indent=2)

    print(f"\n✓ Saved checkpoint to {out_dir / 'checkpoint.pt'}")
    print(f"✓ Saved best model to {out_dir / 'best_model.pt'}")
    print(f"✓ Saved training history to {history_path}")
    print(f"✓ Saved metrics to {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()