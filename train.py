"""
Depth Anything V2 Fine-tuning Script (Relative Depth)
=====================================================
Minimal, single-GPU training script for Lightning.ai / Colab.

Usage:
    python train.py --config finetune_config.yaml

Workspace layout expected (all files inside the cloned repo):
    /teamspace/studios/this_studio/Depth-Anything-V2/
    ├── depth_anything_v2/          # model code (part of repo)
    ├── data/nyu_small/train/rgb/
    ├── data/nyu_small/train/depth/
    ├── data/nyu_small/test/rgb/
    ├── data/nyu_small/test/depth/
    ├── checkpoints/depth_anything_v2_vits.pth
    ├── finetune_config.yaml
    └── train.py
"""

import argparse
import logging
import os
import sys
import time
import random
from datetime import datetime
import yaml

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from PIL import Image

# ---------------------------------------------------------------------------
# Add the repo root to sys.path so we can import depth_anything_v2
# Since train.py lives inside the cloned Depth-Anything-V2/ repo,
# the script's own directory IS the repo root.
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from depth_anything_v2.dpt import DepthAnythingV2
from lora import apply_lora, get_lora_params, count_lora_params, save_lora, lora_summary

# ========================================================================== #
#                              LOGGING SETUP                                 #
# ========================================================================== #

def setup_logging(log_dir="logs"):
    """Configure logging to write to both console and a timestamped log file."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")

    logger = logging.getLogger("depth_ft")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Logging to: {log_file}")
    return logger


log = setup_logging()

# ========================================================================== #
#                             MODEL CONFIGS                                  #
# ========================================================================== #

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}

# ========================================================================== #
#                              DATASET                                       #
# ========================================================================== #

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


class NYUDepthDataset(Dataset):
    """
    Loads paired RGB + depth images from:
        root/rgb/   (*.jpg or *.png)
        root/depth/ (*.png)

    For relative depth fine-tuning, depth maps are normalised to [0, 1].
    """

    def __init__(self, root, image_size=518, augment=False, file_list=None):
        """
        Args:
            root:       Path containing rgb/ and depth/ subdirectories.
            image_size: Resize target (square). Must be divisible by 14 for ViT.
            augment:    If True, apply random horizontal flip.
            file_list:  Optional explicit list of filenames (basenames) to use.
                        If None, all files in rgb/ are used.
        """
        self.rgb_dir   = os.path.join(root, "rgb")
        self.depth_dir = os.path.join(root, "depth")
        self.image_size = image_size
        self.augment = augment

        if file_list is not None:
            self.filenames = sorted(file_list)
        else:
            self.filenames = sorted(os.listdir(self.rgb_dir))

        # Build a map from rgb basenames → depth basenames
        depth_files = set(os.listdir(self.depth_dir))
        self.pairs = []
        for fname in self.filenames:
            stem = os.path.splitext(fname)[0]
            # Try matching depth file with same stem but .png extension
            depth_candidates = [f"{stem}.png", f"{stem}_depth.png"]
            depth_name = None
            for c in depth_candidates:
                if c in depth_files:
                    depth_name = c
                    break
            if depth_name is None:
                # Fallback: if rgb is *_colors.png, depth is *_depth.png
                if "_colors" in stem:
                    d = stem.replace("_colors", "_depth") + ".png"
                    if d in depth_files:
                        depth_name = d
            if depth_name is not None:
                self.pairs.append((fname, depth_name))

        if len(self.pairs) == 0:
            # Print diagnostics to help debug
            rgb_samples = self.filenames[:5] if self.filenames else ["(empty)"]
            depth_samples = sorted(list(depth_files))[:5] if depth_files else ["(empty)"]
            raise RuntimeError(
                f"No valid RGB-Depth pairs found in {root}.\n"
                f"  rgb/ has {len(self.filenames)} files, sample: {rgb_samples}\n"
                f"  depth/ has {len(depth_files)} files, sample: {depth_samples}\n"
                f"  file_list provided: {file_list is not None} ({len(file_list) if file_list else 0} items)"
            )

        log.info(f"  Dataset: {root}  →  {len(self.pairs)} pairs (augment={augment})")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rgb_name, depth_name = self.pairs[idx]

        # --- Load RGB ---
        rgb_path = os.path.join(self.rgb_dir, rgb_name)
        image = cv2.imread(rgb_path)
        if image is None:
            raise FileNotFoundError(f"Could not read: {rgb_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # --- Load Depth ---
        depth_path = os.path.join(self.depth_dir, depth_name)
        depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f"Could not read: {depth_path}")
        depth = depth.astype(np.float32)
        if depth.ndim == 3:
            depth = depth[:, :, 0]

        # --- Resize ---
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
        depth = cv2.resize(depth, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)

        # --- Augmentation: random horizontal flip ---
        if self.augment and random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            depth = np.flip(depth, axis=1).copy()

        # --- Normalise RGB (ImageNet stats) ---
        image = image.astype(np.float32) / 255.0
        image = (image - np.array(IMAGENET_MEAN)) / np.array(IMAGENET_STD)
        image = np.transpose(image, (2, 0, 1))  # HWC → CHW

        # --- Normalise depth to [0, 1] ---
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)

        # --- Valid mask (non-zero depth) ---
        valid_mask = (depth > 1e-6).astype(np.float32)

        return {
            "image":      torch.from_numpy(image).float(),
            "depth":      torch.from_numpy(depth).float(),
            "valid_mask": torch.from_numpy(valid_mask).float(),
        }


# ========================================================================== #
#                               LOSSES                                       #
# ========================================================================== #

class SiLogLoss(nn.Module):
    """
    Scale-Invariant Logarithmic Loss.
    Standard loss for monocular depth estimation (Eigen et al.).
    """
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask=None):
        if valid_mask is not None:
            pred = pred[valid_mask]
            target = target[valid_mask]

        if pred.numel() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Clamp to avoid log(0)
        pred   = torch.clamp(pred, min=1e-6)
        target = torch.clamp(target, min=1e-6)

        log_diff = torch.log(pred) - torch.log(target)
        silog = torch.sqrt(
            torch.mean(log_diff ** 2) - self.lambd * (torch.mean(log_diff) ** 2)
        )
        return silog


class GradientMatchingLoss(nn.Module):
    """
    Penalises differences in spatial gradients (edges) between
    predicted and ground-truth depth. Encourages sharp boundaries.
    """
    def forward(self, pred, target, valid_mask=None):
        # pred, target: (B, H, W)
        pred   = pred.unsqueeze(1)   # (B, 1, H, W)
        target = target.unsqueeze(1)

        # Sobel-like gradients
        pred_dx   = pred[:, :, :, :-1] - pred[:, :, :, 1:]
        pred_dy   = pred[:, :, :-1, :] - pred[:, :, 1:, :]
        target_dx = target[:, :, :, :-1] - target[:, :, :, 1:]
        target_dy = target[:, :, :-1, :] - target[:, :, 1:, :]

        loss_dx = torch.mean(torch.abs(pred_dx - target_dx))
        loss_dy = torch.mean(torch.abs(pred_dy - target_dy))

        return loss_dx + loss_dy


# ========================================================================== #
#                          EVALUATION METRICS                                #
# ========================================================================== #

@torch.no_grad()
def compute_depth_metrics(pred, gt, valid_mask=None):
    """
    Compute standard depth estimation metrics.
    Returns dict with: abs_rel, rmse, delta1, delta2, delta3
    """
    if valid_mask is not None:
        pred = pred[valid_mask]
        gt   = gt[valid_mask]

    pred = torch.clamp(pred, min=1e-6)
    gt   = torch.clamp(gt, min=1e-6)

    if pred.numel() < 10:
        return None

    # Absolute Relative Error
    abs_rel = torch.mean(torch.abs(pred - gt) / gt).item()

    # RMSE
    rmse = torch.sqrt(torch.mean((pred - gt) ** 2)).item()

    # Threshold accuracy (δ < 1.25^n)
    ratio = torch.max(pred / gt, gt / pred)
    d1 = (ratio < 1.25).float().mean().item()
    d2 = (ratio < 1.25 ** 2).float().mean().item()
    d3 = (ratio < 1.25 ** 3).float().mean().item()

    return {"abs_rel": abs_rel, "rmse": rmse, "d1": d1, "d2": d2, "d3": d3}


def compute_ssim(pred, gt, valid_mask=None, window_size=11):
    """
    Compute Structural Similarity Index (SSIM) between predicted and GT depth.
    Simplified version operating on 2D tensors.
    """
    if valid_mask is not None:
        pred = pred * valid_mask.float()
        gt   = gt * valid_mask.float()

    pred = pred.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    gt   = gt.unsqueeze(0).unsqueeze(0)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Use average pooling as a simple window
    pad = window_size // 2
    mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=pad)
    mu_gt   = F.avg_pool2d(gt,   window_size, stride=1, padding=pad)

    mu_pred_sq = mu_pred ** 2
    mu_gt_sq   = mu_gt ** 2
    mu_cross   = mu_pred * mu_gt

    sigma_pred_sq = F.avg_pool2d(pred ** 2, window_size, stride=1, padding=pad) - mu_pred_sq
    sigma_gt_sq   = F.avg_pool2d(gt ** 2,   window_size, stride=1, padding=pad) - mu_gt_sq
    sigma_cross   = F.avg_pool2d(pred * gt, window_size, stride=1, padding=pad) - mu_cross

    ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / \
               ((mu_pred_sq + mu_gt_sq + C1) * (sigma_pred_sq + sigma_gt_sq + C2))

    return ssim_map.mean().item()


@torch.no_grad()
def save_visual_samples(model, dataset, device, save_dir, epoch, num_samples=4):
    """
    Save side-by-side comparison images: [RGB | GT Depth | Predicted Depth].
    Saves to save_dir/visuals/epoch_XX/
    """
    model.eval()
    vis_dir = os.path.join(save_dir, "visuals", f"epoch_{epoch+1:02d}")
    os.makedirs(vis_dir, exist_ok=True)

    indices = list(range(min(num_samples, len(dataset))))

    for idx in indices:
        sample = dataset[idx]
        img       = sample["image"].unsqueeze(0).to(device)
        depth_gt  = sample["depth"].numpy()

        # Predict
        pred = model(img)
        if pred.shape[-2:] != (depth_gt.shape[0], depth_gt.shape[1]):
            pred = F.interpolate(
                pred.unsqueeze(1), depth_gt.shape[-2:],
                mode="bilinear", align_corners=True
            ).squeeze(1)
        pred = pred[0].cpu().numpy()

        # Normalise pred to [0, 1]
        p_min, p_max = pred.min(), pred.max()
        if p_max - p_min > 1e-6:
            pred = (pred - p_min) / (p_max - p_min)

        # Reconstruct RGB from normalised tensor for display
        rgb = sample["image"].numpy().transpose(1, 2, 0)  # CHW → HWC
        rgb = rgb * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)

        # Convert depth maps to colour maps for better visualisation
        gt_vis   = (depth_gt * 255).astype(np.uint8)
        pred_vis = (pred * 255).astype(np.uint8)
        gt_color   = cv2.applyColorMap(gt_vis, cv2.COLORMAP_INFERNO)
        pred_color = cv2.applyColorMap(pred_vis, cv2.COLORMAP_INFERNO)

        # Resize all to same height
        h = 256
        w = int(rgb.shape[1] * h / rgb.shape[0])
        rgb_resized  = cv2.resize(rgb, (w, h))
        gt_resized   = cv2.resize(gt_color, (w, h))
        pred_resized = cv2.resize(pred_color, (w, h))

        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(rgb_resized,  "RGB",       (5, 20), font, 0.6, (255,255,255), 2)
        cv2.putText(gt_resized,   "GT Depth",  (5, 20), font, 0.6, (255,255,255), 2)
        cv2.putText(pred_resized, "Predicted", (5, 20), font, 0.6, (255,255,255), 2)

        # Concatenate side by side
        canvas = np.concatenate([
            cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2BGR),
            gt_resized,
            pred_resized
        ], axis=1)

        cv2.imwrite(os.path.join(vis_dir, f"sample_{idx:03d}.png"), canvas)

    log.info(f"    📷 Visual samples saved to: {vis_dir}")


# ========================================================================== #
#                             TRAINING LOOP                                  #
# ========================================================================== #

def train_one_epoch(model, loader, optimizer, criterion, grad_loss_fn,
                    use_grad_loss, device, epoch, total_epochs):
    model.train()
    running_loss = 0.0
    num_batches  = 0

    for i, batch in enumerate(loader):
        img        = batch["image"].to(device)
        depth_gt   = batch["depth"].to(device)
        valid_mask = batch["valid_mask"].to(device).bool()

        # Forward
        pred = model(img)  # (B, H, W)

        # Resize prediction to match ground truth if needed
        if pred.shape[-2:] != depth_gt.shape[-2:]:
            pred = F.interpolate(
                pred.unsqueeze(1), depth_gt.shape[-2:],
                mode="bilinear", align_corners=True
            ).squeeze(1)

        # Normalise prediction to [0, 1] per sample for relative depth
        # Must avoid in-place ops to keep autograd graph intact
        pred_norm = torch.zeros_like(pred)
        for b in range(pred.shape[0]):
            p_min, p_max = pred[b].min(), pred[b].max()
            if p_max - p_min > 1e-6:
                pred_norm[b] = (pred[b] - p_min) / (p_max - p_min)
            else:
                pred_norm[b] = pred[b]
        pred = pred_norm

        # Loss
        loss = criterion(pred, depth_gt, valid_mask)
        if use_grad_loss:
            loss = loss + 0.5 * grad_loss_fn(pred, depth_gt, valid_mask)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        num_batches  += 1

        # Track gradient norm
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        if (i + 1) % 100 == 0 or (i + 1) == len(loader):
            avg = running_loss / num_batches
            log.info(f"  [Epoch {epoch+1}/{total_epochs}]  Batch {i+1}/{len(loader)}  Loss: {loss.item():.4f}  Avg: {avg:.4f}  GradNorm: {grad_norm:.2f}")

    return running_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, grad_loss_fn, use_grad_loss, device):
    model.eval()
    all_metrics = {"abs_rel": 0, "rmse": 0, "d1": 0, "d2": 0, "d3": 0, "ssim": 0}
    val_loss_total = 0.0
    n = 0

    for batch in loader:
        img        = batch["image"].to(device)
        depth_gt   = batch["depth"].to(device)
        valid_mask = batch["valid_mask"].to(device).bool()

        pred = model(img)

        if pred.shape[-2:] != depth_gt.shape[-2:]:
            pred = F.interpolate(
                pred.unsqueeze(1), depth_gt.shape[-2:],
                mode="bilinear", align_corners=True
            ).squeeze(1)

        # Normalise per sample
        for b in range(pred.shape[0]):
            p_min, p_max = pred[b].min(), pred[b].max()
            if p_max - p_min > 1e-6:
                pred[b] = (pred[b] - p_min) / (p_max - p_min)

            m = compute_depth_metrics(pred[b], depth_gt[b], valid_mask[b])
            if m is not None:
                for k in ["abs_rel", "rmse", "d1", "d2", "d3"]:
                    all_metrics[k] += m[k]
                all_metrics["ssim"] += compute_ssim(pred[b], depth_gt[b], valid_mask[b])
                n += 1

        # Compute val loss (same as train)
        loss = criterion(pred, depth_gt, valid_mask)
        if use_grad_loss and grad_loss_fn is not None:
            loss = loss + 0.5 * grad_loss_fn(pred, depth_gt, valid_mask)
        val_loss_total += loss.item()

    if n == 0:
        return all_metrics, 0.0

    for k in all_metrics:
        all_metrics[k] /= n

    val_loss_avg = val_loss_total / n
    return all_metrics, val_loss_avg


# ========================================================================== #
#                                 MAIN                                       #
# ========================================================================== #

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Depth Anything V2 Fine-tuning (Relative Depth)")
    parser.add_argument("--config", type=str, default="finetune_config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--val-skip", type=int, default=0,
                        help="Number of initial test files (sorted) to skip; remainder used as val")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- Config values ----
    encoder_type = cfg["model"]["encoder_type"]
    checkpoint   = cfg["model"]["checkpoint"]
    batch_size   = cfg["training"]["batch_size"]
    lr           = cfg["training"]["learning_rate"]
    num_epochs   = cfg["training"]["num_epochs"]
    warmup_epochs = cfg["training"].get("warmup_epochs", 1)
    freeze_encoder_epochs = cfg["training"].get("freeze_encoder_epochs", 0)
    train_path   = cfg["data"]["train_path"]
    test_path    = cfg["data"].get("val_path", cfg["data"].get("test_path", "data/nyu_depth/test"))
    image_size   = cfg["data"]["image_size"]
    use_grad_loss     = cfg["loss"].get("use_gradient_matching", True)
    use_si_loss       = cfg["loss"].get("use_scale_invariant", True)

    # LoRA config (optional)
    lora_cfg = cfg.get("lora", {})
    use_lora       = lora_cfg.get("enabled", False)
    lora_rank      = lora_cfg.get("rank", 8)
    lora_alpha     = lora_cfg.get("alpha", 16.0)
    lora_dropout   = lora_cfg.get("dropout", 0.05)
    lora_targets   = lora_cfg.get("target_modules", ["qkv", "proj"])
    lora_lr_mult   = lora_cfg.get("lr_multiplier", 1.0)  # relative to base LR

    log.info(f"\n{'='*60}")
    log.info(f"  Encoder:     {encoder_type}")
    log.info(f"  Checkpoint:  {checkpoint}")
    log.info(f"  Image size:  {image_size}")
    log.info(f"  Batch size:  {batch_size}")
    log.info(f"  LR:          {lr}")
    log.info(f"  Epochs:      {num_epochs}")
    log.info(f"  Train path:  {train_path}")
    log.info(f"  Val source:  {test_path} (skip first {args.val_skip} files)")
    log.info(f"  Losses:      SiLog={use_si_loss}, GradMatch={use_grad_loss}")
    if freeze_encoder_epochs > 0:
        log.info(f"  Freeze:      Encoder frozen for first {freeze_encoder_epochs} epochs")
    if use_lora:
        log.info(f"  LoRA:        rank={lora_rank}, alpha={lora_alpha}, targets={lora_targets}")
    log.info(f"{'='*60}\n")

    # ---- Datasets ----
    # Train: use all files in train/
    train_dataset = NYUDepthDataset(train_path, image_size=image_size, augment=True)

    # Validation: use test/ files after skipping the first N (sorted by name)
    test_rgb_dir = os.path.join(test_path, "rgb")
    all_test_files = sorted(os.listdir(test_rgb_dir))
    val_files = all_test_files[args.val_skip:] if args.val_skip > 0 else all_test_files
    log.info(f"  Test set total: {len(all_test_files)} files")
    log.info(f"  Using {len(val_files)} files for validation (skipping first {args.val_skip})")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)

    val_loader = None
    if len(val_files) > 0:
        val_dataset = NYUDepthDataset(test_path, image_size=image_size, augment=False, file_list=val_files)
        val_loader  = DataLoader(val_dataset, batch_size=1, shuffle=False,
                                 num_workers=2, pin_memory=True)
    else:
        log.info("  ⚠ No validation files — training-only mode")

    # ---- Model ----
    log.info(f"\nLoading model: {encoder_type} ...")
    model_cfg = MODEL_CONFIGS[encoder_type]
    model = DepthAnythingV2(**model_cfg)

    # Load pre-trained weights
    if os.path.isfile(checkpoint):
        log.info(f"  Loading pre-trained weights from: {checkpoint}")
        state_dict = torch.load(checkpoint, map_location="cpu")
        # Load with strict=False to handle any mismatched keys
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            log.info(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            log.info(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        log.info("  ✓ Weights loaded")
    else:
        log.info(f"  ⚠ Checkpoint not found: {checkpoint}")
        log.info(f"  Training from scratch (not recommended)")

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    log.info(f"  Total params:     {total_params:.1f}M")
    log.info(f"  Trainable params: {trainable:.1f}M")

    # ---- Apply LoRA (if enabled) ----
    if use_lora:
        log.info(f"\n  Applying LoRA (rank={lora_rank}, alpha={lora_alpha}) ...")
        num_lora_layers = apply_lora(
            model.pretrained,  # only the DINOv2 encoder
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            target_modules=lora_targets,
        )
        lora_total, lora_trainable = count_lora_params(model)
        log.info(f"  LoRA layers injected: {num_lora_layers}")
        log.info(f"  LoRA params: {lora_total:,} ({lora_total * 4 / 1024:.0f} KB on disk)")
        lora_summary(model)

    # ---- Encoder freeze/unfreeze helpers ----
    wd = cfg["optimizer"].get("weight_decay", 0.0001)

    def freeze_encoder():
        for n, p in model.named_parameters():
            if "pretrained" in n and "lora_" not in n:
                p.requires_grad = False
        frozen = sum(1 for n, p in model.named_parameters()
                     if "pretrained" in n and not p.requires_grad)
        log.info(f"  🔒 Encoder FROZEN ({frozen} param groups)")
        if use_lora:
            lora_count = sum(1 for n, p in model.named_parameters()
                            if "lora_" in n and p.requires_grad)
            log.info(f"  🔑 LoRA adapters remain TRAINABLE ({lora_count} param groups)")

    def unfreeze_encoder():
        for n, p in model.named_parameters():
            if "pretrained" in n:
                p.requires_grad = True
        log.info(f"  🔓 Encoder UNFROZEN — full model training")

    def build_optimizer():
        # Separate params into groups with different LR
        lora_params = []
        enc_params = []
        dec_params = []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "lora_" in n:
                lora_params.append(p)
            elif "pretrained" in n:
                enc_params.append(p)
            else:
                dec_params.append(p)

        param_groups = []
        if lora_params:
            param_groups.append({"params": lora_params, "lr": lr * lora_lr_mult, "label": "lora"})
        if enc_params:
            param_groups.append({"params": enc_params, "lr": lr, "label": "encoder"})
        param_groups.append({"params": dec_params, "lr": lr * 10.0, "label": "decoder"})

        trainable_count = sum(p.numel() for g in param_groups for p in g["params"]) / 1e6
        log.info(f"  Trainable params: {trainable_count:.1f}M")
        for g in param_groups:
            log.info(f"    {g.get('label', '?')}: {sum(p.numel() for p in g['params'])/1e6:.2f}M @ lr={g['lr']:.2e}")
        return AdamW(param_groups, lr=lr, betas=(0.9, 0.999), weight_decay=wd)

    # ---- Initial freeze state ----
    if freeze_encoder_epochs > 0:
        freeze_encoder()
    optimizer = build_optimizer()
    encoder_unfrozen = (freeze_encoder_epochs == 0)

    log.info(f"  Optimizer: AdamW (wd={wd})")

    # ---- Loss functions ----
    criterion    = SiLogLoss()
    grad_loss_fn = GradientMatchingLoss() if use_grad_loss else None

    # ---- Training ----
    save_dir = os.path.join(os.path.dirname(checkpoint) if checkpoint else ".", "finetuned")
    os.makedirs(save_dir, exist_ok=True)

    best_abs_rel = float("inf")
    best_d1      = 0.0

    log.info(f"\n{'='*60}")
    log.info(f"  Starting training for {num_epochs} epochs")
    log.info(f"  Checkpoints will be saved to: {save_dir}")
    log.info(f"{'='*60}\n")

    total_iters = num_epochs * len(train_loader)

    for epoch in range(num_epochs):
        t0 = time.time()

        # ---- Unfreeze encoder after freeze period ----
        if not encoder_unfrozen and epoch >= freeze_encoder_epochs:
            unfreeze_encoder()
            optimizer = build_optimizer()  # Rebuild with encoder params
            encoder_unfrozen = True

        # ---- LR schedule: polynomial decay with warmup ----
        for i_group, param_group in enumerate(optimizer.param_groups):
            # Last group is always decoder; first group is encoder (if unfrozen)
            is_decoder = (i_group == len(optimizer.param_groups) - 1)
            base_lr = lr * 10.0 if is_decoder else lr
            if epoch < warmup_epochs:
                current_lr = base_lr * (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / max(num_epochs - warmup_epochs, 1)
                current_lr = base_lr * (1.0 - progress) ** 0.9
            param_group["lr"] = current_lr

        lr_str = " / ".join(f"{pg['lr']:.2e}" for pg in optimizer.param_groups)
        if not encoder_unfrozen and use_lora:
            phase = "lora+decoder"
        elif not encoder_unfrozen:
            phase = "decoder-only"
        else:
            phase = "full model"
        log.info(f"\n--- Epoch {epoch+1}/{num_epochs}  LR: {lr_str}  [{phase}] ---")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, grad_loss_fn,
            use_grad_loss, device, epoch, num_epochs
        )

        # Validate
        val_loss = 0.0
        if val_loader is not None:
            val_metrics, val_loss = validate(model, val_loader, criterion, grad_loss_fn, use_grad_loss, device)
        else:
            val_metrics = {"abs_rel": 0, "rmse": 0, "d1": 0, "d2": 0, "d3": 0, "ssim": 0}

        # Save visual samples every 5 epochs (and first + last)
        if val_loader is not None and (epoch == 0 or (epoch + 1) % 5 == 0 or epoch == num_epochs - 1):
            save_visual_samples(model, val_loader.dataset, device, save_dir, epoch)

        elapsed = time.time() - t0
        log.info(f"\n  Epoch {epoch+1} Summary ({elapsed:.0f}s):")
        log.info(f"    Train Loss:  {train_loss:.4f}")
        log.info(f"    Val Loss:    {val_loss:.4f}")
        log.info(f"    Val AbsRel:  {val_metrics['abs_rel']:.4f}")
        log.info(f"    Val RMSE:    {val_metrics['rmse']:.4f}")
        log.info(f"    Val SSIM:    {val_metrics['ssim']:.4f}")
        log.info(f"    Val δ1:      {val_metrics['d1']:.4f}")
        log.info(f"    Val δ2:      {val_metrics['d2']:.4f}")
        log.info(f"    Val δ3:      {val_metrics['d3']:.4f}")

        # Overfit check
        if train_loss < val_loss * 0.5:
            log.info(f"    ⚠ Possible overfitting: train_loss ({train_loss:.4f}) << val_loss ({val_loss:.4f})")

        # Save history to CSV for post-analysis
        history_file = os.path.join(save_dir, "training_history.csv")
        write_header = not os.path.exists(history_file)
        with open(history_file, "a") as f:
            if write_header:
                f.write("epoch,train_loss,val_loss,abs_rel,rmse,ssim,d1,d2,d3,phase\n")
            f.write(f"{epoch+1},{train_loss:.6f},{val_loss:.6f},"
                    f"{val_metrics['abs_rel']:.6f},{val_metrics['rmse']:.6f},{val_metrics['ssim']:.6f},"
                    f"{val_metrics['d1']:.6f},{val_metrics['d2']:.6f},{val_metrics['d3']:.6f},{phase}\n")

        # ---- Save checkpoints ----
        # Save latest
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_metrics": val_metrics,
        }, os.path.join(save_dir, "latest.pth"))

        # Save best (by abs_rel)
        improved = False
        if val_metrics["abs_rel"] < best_abs_rel:
            best_abs_rel = val_metrics["abs_rel"]
            improved = True
        if val_metrics["d1"] > best_d1:
            best_d1 = val_metrics["d1"]
            improved = True

        if improved:
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            # Save LoRA weights separately (tiny file)
            if use_lora:
                lora_path = os.path.join(save_dir, "best_lora.pth")
                n_saved = save_lora(model, lora_path)
                log.info(f"    ★ New best model saved! (AbsRel={best_abs_rel:.4f}, δ1={best_d1:.4f})")
                log.info(f"    ★ LoRA weights saved: {lora_path} ({n_saved} tensors)")
            else:
                log.info(f"    ★ New best model saved! (AbsRel={best_abs_rel:.4f}, δ1={best_d1:.4f})")

    log.info(f"\n{'='*60}")
    log.info(f"  Training complete!")
    log.info(f"  Best AbsRel: {best_abs_rel:.4f}")
    log.info(f"  Best δ1:     {best_d1:.4f}")
    log.info(f"  Checkpoints: {save_dir}")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()
