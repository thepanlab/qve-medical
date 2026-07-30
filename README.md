# QVE-Medical: Web-Scale VLM Vision Encoders for Medical Image Classification

Code for **"Web-Scale Vision-Language Model Pretraining Transfers to Medical Imaging Classification Without Language"**

Haoyang Cui, Chen Wang, Qinggong Tang, Chongle Pan

This repository evaluates the vision encoder of Qwen3-VL (**QVE**), decoupled from its language model, as a task-agnostic backbone for medical image classification. QVE is benchmarked across four modality groups — optical coherence tomography (OCT), gastrointestinal endoscopy (Kvasir), smartphone dermoscopy (PAD-UFES-20), and MedMNIST v2 — and paired with a multi-scale, token-wise depth-fusion architecture that aggregates early/mid/late encoder layers.

## Repository structure

```
scripts/
├── multi_tap_extractor.py   # Hooks into QVE transformer blocks to extract multi-depth patch tokens
├── depth_fusion.py           # Token-wise depth fusion (scalar / token_attention / gated) + pooling (mean / max / attention / cls)
├── train_kvasir.py           # Training + eval on Kvasir v1 (8-class GI endoscopy)
├── train_padufes.py          # Training + eval on PAD-UFES-20 (6-class skin lesion, patient-level split)
└── train_medmnist.py         # Training + eval on MedMNIST v2 subtasks
```

## Setup

```bash
git clone https://github.com/<your-username>/qve-medical.git
cd qve-medical
pip install torch torchvision transformers pillow numpy pandas tqdm accelerate
```

QVE is loaded directly from Hugging Face (`Qwen/Qwen3-VL-8B-Instruct`) via `transformers`, so no separate model download step is needed — the first run will fetch and cache the weights.

## Datasets

| Dataset | Task | Link |
|---|---|---|
| **Kvasir v1** | 8-class GI endoscopy (4,000 images) | https://datasets.simula.no/kvasir/ |
| **PAD-UFES-20** | 6-class smartphone skin-lesion (2,298 images) | https://data.mendeley.com/datasets/zr7vgbcyr2/1 |
| **MedMNIST v2** | 12 subtasks used in this study | https://medmnist.com/ |
| **OCT Liver** (ours) | Binary normal vs. tumor, 4-channel (intensity, retardation, optic axis, DOPU) OCT/PS-OCT, 5 subjects | https://zenodo.org/records/21694718 |

`train_medmnist.py` expects pre-processed 224×224 `.npz` files (the format distributed by the MedMNIST project) at `<data_root>/<dataset_name>_224.npz`. `train_kvasir.py` and `train_padufes.py` expect the datasets extracted in their original folder structure, pointed to via `--data_root`.

## Usage

**Kvasir v1**
```bash
python scripts/train_kvasir.py --data_root /path/to/kvasir-dataset
python scripts/train_kvasir.py --data_root /path/to/kvasir-dataset \
    --use_multi_tap --tap_layers 8 16 24 --fusion_type token_attention --pooling_type attention
```

**PAD-UFES-20**
```bash
python scripts/train_padufes.py --data_root /path/to/pad_ufes
python scripts/train_padufes.py --data_root /path/to/pad_ufes \
    --use_multi_tap --tap_layers 6 13 20 --fusion_type token_attention --pooling_type mean
```

**MedMNIST**
```bash
python scripts/train_medmnist.py --medmnist pathmnist --data_root /path/to/medmnist
python scripts/train_medmnist.py --medmnist tissuemnist --mode frozen --epochs 20 --data_root /path/to/medmnist
```

Common flags across all three scripts:
- `--mode {frozen, partial_finetune, full_finetune}` — how much of the QVE backbone is trained
- `--use_multi_tap` — enable multi-scale feature fusion instead of the single-tap (last-layer) baseline
- `--tap_layers` — which encoder layers to tap (defaults differ slightly per script; see each script's `--help`)
- `--fusion_type {scalar, token_attention, gated}` and `--pooling_type {mean, max, attention, cls}`
- `--epochs`, `--batch_size`, `--lr`, `--backbone_lr`, `--seed`

Run any script with `-h` for the full list of arguments.

## Citation

A citation entry will be added once the paper is published/available on a preprint server.

## License

*To be determined.* Please contact the authors before reuse until a license is added.
