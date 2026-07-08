import argparse
from pathlib import Path

import cv2
import ncnn
import numpy as np


def parse_ncnn_param(param_path):
    input_names = []
    output_name = None

    with open(param_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines[2:]:
        parts = line.split()
        if len(parts) < 5:
            continue

        layer_type = parts[0]
        bottom_count = int(parts[2])
        top_count = int(parts[3])
        top_start = 4 + bottom_count
        tops = parts[top_start:top_start + top_count]

        if layer_type == "Input":
            input_names.extend(tops)
        elif tops:
            output_name = tops[-1]

    return input_names, output_name


def load_image_chw(path, width, height):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"failed to read image: {path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image = image.transpose(2, 0, 1)
    return np.ascontiguousarray(image)


def mat_to_numpy(mat, height, width):
    array = np.array(mat)
    if array.size == height * width:
        return array.reshape(height, width)

    squeezed = np.squeeze(array)
    if squeezed.ndim == 2:
        return squeezed
    if squeezed.size == height * width:
        return squeezed.reshape(height, width)

    raise ValueError(f"unexpected output shape from ncnn: {array.shape}")


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
    parser.add_argument("--param", required=True, help="path to .ncnn.param")
    parser.add_argument("--bin", required=True, help="path to .ncnn.bin")
    parser.add_argument("--left", required=True, help="left rectified image")
    parser.add_argument("--right", required=True, help="right rectified image")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--out", default="outputs/ncnn_disp_vis.png")
    parser.add_argument("--save_raw", default="outputs/ncnn_disp_raw.npy")
    parser.add_argument("--save_positive", default="outputs/ncnn_disp_positive.npy")
    parser.add_argument("--left_blob", default=None)
    parser.add_argument("--right_blob", default=None)
    parser.add_argument("--output_blob", default=None)
    parser.add_argument("--keep_batch", action="store_true")
    parser.add_argument("--vulkan", action="store_true")
    args = parser.parse_args()

    param_path = Path(args.param)
    bin_path = Path(args.bin)
    input_names, output_name = parse_ncnn_param(param_path)

    left_blob = args.left_blob or (input_names[0] if len(input_names) > 0 else "left")
    right_blob = args.right_blob or (input_names[1] if len(input_names) > 1 else "right")
    output_blob = args.output_blob or output_name
    if output_blob is None:
        raise ValueError("failed to infer output blob name; pass --output_blob manually")

    left = load_image_chw(args.left, args.width, args.height)
    right = load_image_chw(args.right, args.width, args.height)
    if args.keep_batch:
        left = left[None]
        right = right[None]

    net = ncnn.Net()
    net.opt.use_vulkan_compute = args.vulkan
    param_ret = net.load_param(str(param_path))
    model_ret = net.load_model(str(bin_path))
    if param_ret != 0:
        raise RuntimeError(f"ncnn load_param failed with code {param_ret}")
    if model_ret != 0:
        raise RuntimeError(f"ncnn load_model failed with code {model_ret}")

    extractor = net.create_extractor()
    print(f"input blobs: {left_blob}, {right_blob}")
    print(f"output blob: {output_blob}")
    print(f"left shape: {left.shape}, right shape: {right.shape}")
    extractor.input(left_blob, ncnn.Mat(left).clone())
    extractor.input(right_blob, ncnn.Mat(right).clone())

    ret, output = extractor.extract(output_blob)
    if ret != 0:
        raise RuntimeError(f"ncnn extract failed with code {ret}")

    raw_disparity = mat_to_numpy(output, args.height, args.width)
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

    print(f"saved visualization: {out_path}")
    print(f"saved raw disparity: {raw_path}")
    print(f"saved positive disparity: {positive_path}")
    print(
        "positive disparity:",
        f"shape={positive_disparity.shape}",
        f"min={float(np.nanmin(positive_disparity)):.6f}",
        f"max={float(np.nanmax(positive_disparity)):.6f}",
    )


if __name__ == "__main__":
    main()
