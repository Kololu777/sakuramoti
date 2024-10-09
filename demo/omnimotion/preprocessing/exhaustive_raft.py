import os
import re
from pathlib import Path

import numpy as np
from sakuramoti.io import load_image
from sakuramoti.transformation import InputPadder
from sakuramoti.flow_model.raft.raft import RAFT


def run_exhaustive_flow(video_frame_path: str, save_data_dir: str, model_conf: dict):
    flow_out_dir = os.path.join(save_data_dir, "raft_exhaustive")
    os.makedirs(flow_out_dir, exist_ok=True)

    device = "cuda"
    model = RAFT(**model_conf).to(device=device).eval()

    pattern = re.compile(r".*\.(png|jpg)$", re.IGNORECASE)
    images = [str(file) for file in Path(video_frame_path).iterdir() if pattern.match(str(file))]
    images = sorted(images)

    num_imgs = len(images)

    def processing_raft(i, j, flow_low_prev):
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
        return flow_low_prev

    for i in range(num_imgs):
        flow_low_prev = None
        for j in range(i + 1, num_imgs):
            flow_low_prev = processing_raft(i, j, flow_low_prev)

    for i in range(num_imgs - 1, 0, -1):
        flow_low_prev = None
        for j in range(i - 1, -1, -1):
            flow_low_prev = processing_raft(i, j, flow_low_prev)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_frame_path", type=str, default="", help="video frame path")
    parser.add_argument("--save_data_dir", type=str, default="", help="save data dir")
    parser.add_argument("--model", type=str, default="base", help="model size")
    parser.add_argument("--pretrained", type=str, default="things", help="pretrained model")
    parser.add_argument("--mixed_precision", action="store_true", help="use mixed precision")
    args = parser.parse_args()

    model_conf = {"raft_model": args.model, "pretrained": args.pretrained, "mixed_precision": args.mixed_precision}

    run_exhaustive_flow(args.video_frame_path, args.save_data_dir, model_conf)
