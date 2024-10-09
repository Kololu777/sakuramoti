import os
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from sakuramoti.features.dino.vision_transformer import VisionTransformer, load_dino_model


def extract_feature(model: VisionTransformer, frame: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Extract one frame feature everytime.
    Args:
        model: DINO model
        frame: one frame image
     Returns:
         feature: feature of the frame.
         h: Divided height of the frame into patches.
         w: Divided width of the frame into patches.
    """
    with torch.no_grad():
        feature = model.get_intermediate_layers(frame, n=1)[0][:, 1:, :]  # We discard the [CLS] token
        dim = feature.shape[-1]
        h, w = int(frame.shape[-2] / model.patch_embed.patch_size), int(frame.shape[-1] / model.patch_embed.patch_size)
        feature = feature.reshape(h, w, dim)
    return feature, h, w


def read_frame(frame_dir: str, scale_size: list[int] | None = None) -> tuple[torch.Tensor, int, int]:
    """
    read a single frame & preprocess
    """
    if scale_size is None:
        scale_size = [480]

    img = cv2.imread(frame_dir)
    ori_h, ori_w, _ = img.shape
    if len(scale_size) == 1:
        if ori_h > ori_w:
            tw = scale_size[0]
            th = (tw * ori_h) / ori_w
            th = int((th // 64) * 64)
        else:
            th = scale_size[0]
            tw = (th * ori_w) / ori_h
            tw = int((tw // 64) * 64)
    else:
        th, tw = scale_size
    img = cv2.resize(img, (tw, th))
    img = img.astype(np.float32)
    img = img / 255.0
    img = img[:, :, ::-1]
    img = np.transpose(img.copy(), (2, 0, 1))
    img = torch.from_numpy(img).float()
    img = color_normalize(img)
    return img, ori_h, ori_w


def color_normalize(x, mean: list[float] | None = None, std: list[float] | None = None):
    """Normalize the color of the image."""
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.228, 0.224, 0.225]
    for t, m, s in zip(x, mean, std, strict=True):
        t.sub_(m).div_(s)
    return x


def main():
    parser = argparse.ArgumentParser("Evaluation with video object segmentation on DAVIS 2017")
    parser.add_argument(
        "--arch",
        default="vit_small",
        type=str,
        choices=["vit_tiny", "vit_small", "vit_base"],
        help="Architecture (support only ViT atm).",
    )
    parser.add_argument("--patch_size", default=16, type=int, help="Patch resolution of the model.")
    parser.add_argument("--data_dir", type=str, default="", help="dataset dir")
    args = parser.parse_args()

    model = load_dino_model(args.arch, patch_size=args.patch_size).cuda().eval()

    color_dir = Path(args.data_dir) / "color"
    frame_list = sorted(color_dir.glob("*.png"))

    save_dir = Path(args.data_dir) / "features" / "dino"
    os.makedirs(save_dir, exist_ok=True)

    for frame_path in tqdm(frame_list):
        frame, ori_h, ori_w = read_frame(frame_path)
        frame = frame.unsqueeze(0).cuda()
        frame_feat, h, w = extract_feature(model, frame)
        frame_feat = frame_feat.cpu().numpy()
        frame_name = frame_path.stem
        np.save(save_dir / f"{frame_name}.npy", frame_feat)


if __name__ == "__main__":
    main()
