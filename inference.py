"""
Depth Anything V2 — Inference Script
=====================================
Run a fine-tuned (or pre-trained) model on images and visualize depth predictions.

Usage:
    # Single image
    python inference.py --checkpoint checkpoints/finetuned/best_model.pth --input test.jpg

    # Folder of images
    python inference.py --checkpoint checkpoints/finetuned/best_model.pth --input data/nyu_depth/test/rgb/

    # With ground truth comparison
    python inference.py --checkpoint checkpoints/finetuned/best_model.pth \
                        --input data/nyu_depth/test/rgb/ \
                        --gt data/nyu_depth/test/depth/

    # Save raw depth as .npy
    python inference.py --checkpoint checkpoints/finetuned/best_model.pth --input test.jpg --save-raw

Workspace layout:
    /teamspace/studios/this_studio/Depth-Anything-V2/
    ├── depth_anything_v2/
    ├── checkpoints/finetuned/best_model.pth
    └── inference.py
"""

import argparse
import os
import sys
import time
import glob

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Add repo root to sys.path
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from depth_anything_v2.dpt import DepthAnythingV2

# ========================================================================== #
#                             MODEL CONFIGS                                  #
# ========================================================================== #

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64,  'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


# ========================================================================== #
#                            DEPTH PREDICTION                                #
# ========================================================================== #

@torch.no_grad()
def predict_depth(model, image_bgr, input_size=518, device='cuda'):
    """
    Run depth prediction on a single BGR image (as loaded by cv2).
    Returns: depth_map (H, W) numpy array normalised to [0, 1].
    """
    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Resize to model input size (must be multiple of 14)
    img = cv2.resize(image_rgb, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    img = (img - IMAGENET_MEAN) / IMAGENET_STD
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW

    tensor = torch.from_numpy(img).unsqueeze(0).float().to(device)

    # Forward pass
    pred = model(tensor)  # (1, H', W')

    # Resize back to original resolution
    pred = F.interpolate(
        pred.unsqueeze(1), (h, w),
        mode="bilinear", align_corners=True
    )[0, 0]

    depth = pred.cpu().numpy()

    # Normalise to [0, 1]
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        depth = (depth - d_min) / (d_max - d_min)
    else:
        depth = np.zeros_like(depth)

    return depth


# ========================================================================== #
#                           VISUALIZATION                                    #
# ========================================================================== #

def depth_to_colormap(depth, colormap=cv2.COLORMAP_INFERNO):
    """Convert a [0, 1] depth map to a colormapped BGR image."""
    depth_u8 = (depth * 255).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, colormap)


def create_comparison(rgb_bgr, pred_depth, gt_depth=None, max_height=480):
    """
    Create a side-by-side comparison image.
    If gt_depth is provided: [RGB | GT Depth | Predicted Depth]
    Otherwise:               [RGB | Predicted Depth]
    """
    h, w = rgb_bgr.shape[:2]
    scale = max_height / h
    new_w = int(w * scale)
    new_h = max_height

    # Resize RGB
    rgb_resized = cv2.resize(rgb_bgr, (new_w, new_h))

    # Predicted depth colormap
    pred_color = depth_to_colormap(pred_depth)
    pred_resized = cv2.resize(pred_color, (new_w, new_h))

    # Labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(rgb_resized,  "RGB",       (8, 28), font, 0.8, (255,255,255), 2)
    cv2.putText(pred_resized, "Predicted", (8, 28), font, 0.8, (255,255,255), 2)

    panels = [rgb_resized, pred_resized]

    # GT depth if available
    if gt_depth is not None:
        gt_color = depth_to_colormap(gt_depth)
        gt_resized = cv2.resize(gt_color, (new_w, new_h))
        cv2.putText(gt_resized, "GT Depth", (8, 28), font, 0.8, (255,255,255), 2)
        panels = [rgb_resized, gt_resized, pred_resized]

    return np.concatenate(panels, axis=1)


def create_overlay(rgb_bgr, pred_depth, alpha=0.5):
    """Blend depth heatmap onto the RGB image."""
    pred_color = depth_to_colormap(pred_depth)
    pred_resized = cv2.resize(pred_color, (rgb_bgr.shape[1], rgb_bgr.shape[0]))
    overlay = cv2.addWeighted(rgb_bgr, 1 - alpha, pred_resized, alpha, 0)
    return overlay


# ========================================================================== #
#                               MAIN                                         #
# ========================================================================== #

def collect_images(input_path):
    """Collect image file paths from a file or directory."""
    if os.path.isfile(input_path):
        return [input_path]
    elif os.path.isdir(input_path):
        files = []
        for ext in IMAGE_EXTENSIONS:
            files.extend(glob.glob(os.path.join(input_path, f"*{ext}")))
            files.extend(glob.glob(os.path.join(input_path, f"*{ext.upper()}")))
        return sorted(set(files))
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")


def find_gt_depth(image_path, gt_dir):
    """Try to find a matching ground truth depth map."""
    if gt_dir is None:
        return None

    stem = os.path.splitext(os.path.basename(image_path))[0]

    # Try common naming patterns
    candidates = [
        os.path.join(gt_dir, f"{stem}.png"),
        os.path.join(gt_dir, f"{stem}_depth.png"),
    ]
    if "_colors" in stem:
        candidates.append(os.path.join(gt_dir, stem.replace("_colors", "_depth") + ".png"))

    for c in candidates:
        if os.path.isfile(c):
            depth = cv2.imread(c, cv2.IMREAD_UNCHANGED).astype(np.float32)
            if depth.ndim == 3:
                depth = depth[:, :, 0]
            d_min, d_max = depth.min(), depth.max()
            if d_max - d_min > 1e-6:
                depth = (depth - d_min) / (d_max - d_min)
            return depth

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Depth Anything V2 — Inference & Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference.py --checkpoint checkpoints/finetuned/best_model.pth --input test.jpg
  python inference.py --checkpoint checkpoints/finetuned/best_model.pth --input data/nyu_depth/test/rgb/
  python inference.py --checkpoint checkpoints/finetuned/best_model.pth --input data/nyu_depth/test/rgb/ --gt data/nyu_depth/test/depth/
        """
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--encoder", type=str, default="vits",
                        choices=["vits", "vitb", "vitl", "vitg"],
                        help="Encoder type (must match checkpoint)")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to an image or directory of images")
    parser.add_argument("--gt", type=str, default=None,
                        help="Optional: path to ground truth depth directory")
    parser.add_argument("--output", type=str, default="inference_output",
                        help="Output directory for results")
    parser.add_argument("--input-size", type=int, default=518,
                        help="Model input resolution (default: 518)")
    parser.add_argument("--mode", type=str, default="comparison",
                        choices=["comparison", "overlay", "raw", "all"],
                        help="Output mode: comparison, overlay, raw (.npy), or all")
    parser.add_argument("--save-raw", action="store_true",
                        help="Also save raw depth as .npy files")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Limit number of images to process")
    args = parser.parse_args()

    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- Load model ----
    print(f"\nLoading model: {args.encoder} ...")
    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])

    print(f"  Loading checkpoint: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location="cpu")

    # Handle checkpoint format (could be full checkpoint dict or just state_dict)
    if "model" in state_dict:
        state_dict = state_dict["model"]
        # Remove "module." prefix if saved from DDP
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys")
    print("  ✓ Model loaded")

    model = model.to(device)
    model.eval()

    # ---- Collect images ----
    image_paths = collect_images(args.input)
    if args.max_images:
        image_paths = image_paths[:args.max_images]
    print(f"\n  Found {len(image_paths)} images")

    # ---- Output directory ----
    os.makedirs(args.output, exist_ok=True)
    if args.mode == "all" or args.save_raw:
        os.makedirs(os.path.join(args.output, "raw"), exist_ok=True)

    # ---- Run inference ----
    print(f"\n{'='*60}")
    print(f"  Running inference ({args.mode} mode)")
    print(f"  Output: {args.output}")
    print(f"{'='*60}\n")

    total_time = 0
    for i, img_path in enumerate(image_paths):
        basename = os.path.splitext(os.path.basename(img_path))[0]

        # Load image
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print(f"  [{i+1}/{len(image_paths)}] ⚠ Could not read: {img_path}")
            continue

        # Predict
        t0 = time.time()
        pred_depth = predict_depth(model, image_bgr, args.input_size, device)
        dt = time.time() - t0
        total_time += dt

        # Find GT if available
        gt_depth = find_gt_depth(img_path, args.gt)

        # Save outputs based on mode
        if args.mode in ("comparison", "all"):
            canvas = create_comparison(image_bgr, pred_depth, gt_depth)
            out_path = os.path.join(args.output, f"{basename}_comparison.png")
            cv2.imwrite(out_path, canvas)

        if args.mode in ("overlay", "all"):
            overlay = create_overlay(image_bgr, pred_depth)
            out_path = os.path.join(args.output, f"{basename}_overlay.png")
            cv2.imwrite(out_path, overlay)

        if args.mode == "raw" or args.save_raw or args.mode == "all":
            npy_path = os.path.join(args.output, "raw", f"{basename}_depth.npy")
            np.save(npy_path, pred_depth)

            # Also save colormap version for quick viewing
            color_path = os.path.join(args.output, f"{basename}_depth.png")
            cv2.imwrite(color_path, depth_to_colormap(pred_depth))

        # Progress
        gt_str = " +GT" if gt_depth is not None else ""
        print(f"  [{i+1}/{len(image_paths)}] {basename}  ({dt*1000:.0f}ms){gt_str}")

    # ---- Summary ----
    avg_ms = (total_time / len(image_paths) * 1000) if image_paths else 0
    print(f"\n{'='*60}")
    print(f"  Inference complete!")
    print(f"  Processed: {len(image_paths)} images")
    print(f"  Avg time:  {avg_ms:.0f}ms per image")
    print(f"  Output:    {os.path.abspath(args.output)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
