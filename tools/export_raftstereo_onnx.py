import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.export_raftstereo_torchscript import (  # noqa: E402
    RAFTStereoExportWrapper,
    fuse_downsample_batch_norm,
    load_checkpoint,
    model_args,
)
from core.raft_stereo import RAFTStereo  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_ckpt", default="raftstereo-sceneflow.pth")
    parser.add_argument("--output", default="exports/raftstereo.onnx")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--opset", type=int, default=17)
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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        output = wrapped(left, right)
        torch.onnx.export(
            wrapped,
            (left, right),
            str(output_path),
            input_names=["left", "right"],
            output_names=["disparity"],
            opset_version=args.opset,
            do_constant_folding=True,
        )

    print(f"saved: {output_path}")
    print(f"input: left/right [1,3,{args.height},{args.width}], float32 RGB 0..255")
    print(f"output: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
