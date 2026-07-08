import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.raft_stereo import RAFTStereo  # noqa: E402


class RAFTStereoExportWrapper(nn.Module):
    def __init__(self, model, iters):
        super().__init__()
        self.model = model
        self.iters = iters

    def forward(self, left, right):
        _, disparity = self.model(left, right, iters=self.iters, test_mode=True)
        return disparity


def model_args(corr_implementation="reg"):
    return SimpleNamespace(
        hidden_dims=[128, 128, 128],
        corr_implementation=corr_implementation,
        shared_backbone=False,
        corr_levels=4,
        corr_radius=4,
        n_downsample=2,
        context_norm="batch",
        slow_fast_gru=False,
        n_gru_layers=3,
        mixed_precision=False,
    )


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device)
    if any(k.startswith("module.") for k in checkpoint.keys()):
        checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}
    return checkpoint


def fuse_downsample_batch_norm(model):
    for module in model.modules():
        downsample = getattr(module, "downsample", None)
        if not isinstance(downsample, nn.Sequential) or len(downsample) < 2:
            continue
        if isinstance(downsample[0], nn.Conv2d) and isinstance(downsample[1], nn.BatchNorm2d):
            downsample[0] = fuse_conv_bn_eval(downsample[0], downsample[1])
            downsample[1] = nn.Identity()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_ckpt", default="raftstereo-sceneflow.pth")
    parser.add_argument("--output", default="exports/raftstereo_trace.pt")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()

    if args.height % 32 != 0 or args.width % 32 != 0:
        raise ValueError("height and width must be divisible by 32")

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = RAFTStereo(model_args())
    model.load_state_dict(load_checkpoint(args.restore_ckpt, device), strict=True)
    model.to(device).eval()
    fuse_downsample_batch_norm(model)

    wrapped = RAFTStereoExportWrapper(model, args.iters).to(device).eval()
    left = torch.zeros(1, 3, args.height, args.width, dtype=torch.float32, device=device)
    right = torch.zeros(1, 3, args.height, args.width, dtype=torch.float32, device=device)

    with torch.no_grad():
        traced = torch.jit.trace(wrapped, (left, right), strict=False)
        if args.freeze:
            traced = torch.jit.freeze(traced)
        output = traced(left, right)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output_path))

    print(f"saved: {output_path}")
    print(f"input: left/right [1,3,{args.height},{args.width}], float32 RGB 0..255")
    print(f"output: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
