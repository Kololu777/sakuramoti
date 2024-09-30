import os
import re
from pathlib import Path

import numpy as np
from sakuramoti.io import load_image
from sakuramoti.transformation import InputPadder
from sakuramoti.flow_model.raft.raft import RAFT


def run_exhaustive_flow(path: str, data_dir: str):
    flow_out_dir = os.path.join(data_dir, "raft_exhaustive")
    os.makedirs(flow_out_dir, exist_ok=True)

    device = "cuda"
    model = RAFT().to(device=device).eval()

    pattern = re.compile(r".*\.(png|jpg)$", re.IGNORECASE)
    images = [str(file) for file in Path(path).iterdir() if pattern.match(str(file))]
    images = sorted(images)

    num_imgs = len(images)

    for i in range(num_imgs):
        flow_low_prev = None
        for j in range(i + 1, num_imgs):
            imfile1 = images[i]
            imfile2 = images[j]

            image1 = load_image(imfile1, device)
            image2 = load_image(imfile2, device)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model.pred(image1, image2, iters=20, flow_init=flow_low_prev)
            flow_up = padder.unpad(flow_up)
            flow_up_np = flow_up.squeeze().permute(1, 2, 0).cpu().numpy()
            save_file = os.path.join(flow_out_dir, f"{os.path.basename(imfile1)}_{os.path.basename(imfile2)}.npy")
            np.save(save_file, flow_up_np)
            flow_low_prev = flow_low


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/raft-things.pth", help="restore checkpoint")
    parser.add_argument("--small", action="store_true", help="use small model")
    parser.add_argument("--mixed_precision", action="store_true", help="use mixed precision")
    parser.add_argument("--alternate_corr", action="store_true", help="use efficent correlation implementation")
    parser.add_argument("--data_dir", type=str, default="", help="dataset dir")
    args = parser.parse_args()

    run_exhaustive_flow(args)
