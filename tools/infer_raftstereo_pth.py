import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.raft_stereo import RAFTStereo  # noqa: E402
from core.utils.utils import InputPadder  # noqa: E402
from tools.export_raftstereo_torchscript import (  # noqa: E402
    fuse_downsample_batch_norm,
    load_checkpoint,
    model_args,
)


def load_image(path, width=None, height=None):
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"failed to read image: {path}")

    original_height, original_width = image_bgr.shape[:2]
    if width is not None and height is not None:
        image_bgr = cv2.resize(image_bgr, (width, height), interpolation=cv2.INTER_LINEAR)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float()[None]
    return tensor, (original_height, original_width), image_bgr.shape[:2]


def colorize_disparity(disparity):
    finite = np.isfinite(disparity)
    if not finite.any():
        return np.zeros((*disparity.shape, 3), dtype=np.uint8)

    values = disparity[finite]
    lo, hi = np.percentile(values, [2, 98])
    normalized = np.clip((disparity - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    gray = (normalized * 255.0).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_JET)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_ckpt", required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--out", default="outputs/pth_disp_vis.png")
    parser.add_argument("--save_raw", default="outputs/pth_disp_raw.npy")
    parser.add_argument("--save_positive", default="outputs/pth_disp_positive.npy")
    parser.add_argument("--save_fullres_positive", default=None)
    parser.add_argument("--context_norm", default="batch", choices=["group", "batch", "instance", "none"])
    parser.add_argument("--mixed_precision", action="store_true")
    args = parser.parse_args()

    if (args.width is None) != (args.height is None):
        raise ValueError("--width and --height must be specified together")
    if args.width is not None and (args.width % 32 != 0 or args.height % 32 != 0):
        raise ValueError("--width and --height must be divisible by 32")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    left, original_shape, infer_shape = load_image(args.left, args.width, args.height)
    right, right_original_shape, right_infer_shape = load_image(args.right, args.width, args.height)
    if original_shape != right_original_shape or infer_shape != right_infer_shape:
        raise ValueError("left and right images must have the same size")

    margs = model_args()
    margs.context_norm = args.context_norm
    margs.mixed_precision = args.mixed_precision
    model = RAFTStereo(margs)
    model.load_state_dict(load_checkpoint(args.restore_ckpt, device), strict=True)
    model.to(device).eval()
    fuse_downsample_batch_norm(model)

    left = left.to(device)
    right = right.to(device)
    padder = InputPadder(left.shape, divis_by=32)
    left, right = padder.pad(left, right)

    with torch.no_grad():
        _, flow_up = model(left, right, iters=args.iters, test_mode=True)
        flow_up = padder.unpad(flow_up)

    raw_disparity = flow_up[0, 0].detach().cpu().numpy()
    positive_disparity = -raw_disparity

    out_path = Path(args.out)
    raw_path = Path(args.save_raw)
    positive_path = Path(args.save_positive)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    positive_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(raw_path, raw_disparity)
    np.save(positive_path, positive_disparity)
    cv2.imwrite(str(out_path), colorize_disparity(positive_disparity))

    if args.save_fullres_positive:
        fullres_path = Path(args.save_fullres_positive)
        fullres_path.parent.mkdir(parents=True, exist_ok=True)
        original_height, original_width = original_shape
        infer_height, infer_width = positive_disparity.shape
        scale_x = original_width / infer_width
        fullres = cv2.resize(
            positive_disparity,
            (original_width, original_height),
            interpolation=cv2.INTER_LINEAR,
        ) * scale_x
        np.save(fullres_path, fullres)
        print(f"saved fullres positive disparity: {fullres_path}")

    print(f"saved visualization: {out_path}")
    print(f"saved raw disparity: {raw_path}")
    print(f"saved positive disparity: {positive_path}")
    print(f"original image shape: {original_shape}")
    print(f"inference image shape: {positive_disparity.shape}")
    print(
        "positive disparity:",
        f"min={float(np.nanmin(positive_disparity)):.6f}",
        f"max={float(np.nanmax(positive_disparity)):.6f}",
    )


if __name__ == "__main__":
    main()
