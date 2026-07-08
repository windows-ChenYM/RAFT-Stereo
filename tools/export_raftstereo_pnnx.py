import argparse
import sys
from pathlib import Path

import pnnx
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


def export_torchscript(restore_ckpt, output, height, width, iters, device_name):
    device = torch.device(device_name if device_name == "cpu" or torch.cuda.is_available() else "cpu")
    model = RAFTStereo(model_args())
    model.load_state_dict(load_checkpoint(restore_ckpt, device), strict=True)
    model.to(device).eval()
    fuse_downsample_batch_norm(model)

    wrapped = RAFTStereoExportWrapper(model, iters).to(device).eval()
    left = torch.zeros(1, 3, height, width, dtype=torch.float32, device=device)
    right = torch.zeros(1, 3, height, width, dtype=torch.float32, device=device)

    with torch.no_grad():
        traced = torch.jit.trace(wrapped, (left, right), strict=False)
        output_tensor = traced(left, right)

    output.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(output))
    return tuple(output_tensor.shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_ckpt", default="raftstereo-sceneflow.pth")
    parser.add_argument("--output_prefix", default="exports/raftstereo_768x1024_i1")
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--pnnx_device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--fp16", action="store_true", help="export fp16 ncnn weights")
    args = parser.parse_args()

    if args.height % 32 != 0 or args.width % 32 != 0:
        raise ValueError("height and width must be divisible by 32")

    prefix = Path(args.output_prefix)
    pt_path = prefix.with_suffix(".pt")
    ncnn_param = prefix.with_suffix(".ncnn.param")
    ncnn_bin = prefix.with_suffix(".ncnn.bin")

    output_shape = export_torchscript(
        args.restore_ckpt,
        pt_path,
        args.height,
        args.width,
        args.iters,
        args.device,
    )

    pnnx.convert(
        pt_path.as_posix(),
        input_shapes=[[1, 3, args.height, args.width], [1, 3, args.height, args.width]],
        input_types=["f32", "f32"],
        device=args.pnnx_device or args.device,
        ncnnparam=ncnn_param.as_posix(),
        ncnnbin=ncnn_bin.as_posix(),
        fp16=args.fp16,
    )

    print(f"saved torchscript: {pt_path}")
    print(f"saved ncnn param: {ncnn_param}")
    print(f"saved ncnn bin: {ncnn_bin}")
    print(f"input: left/right [1,3,{args.height},{args.width}], float32 RGB 0..255")
    print(f"output: {output_shape}")


if __name__ == "__main__":
    main()
