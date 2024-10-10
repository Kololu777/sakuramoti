import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import imageio
import torch.nn.functional as F
from sakuramoti.geometry.grid import generate_grid, normalize_coords


def opj_and_mkdir(dir: str, *args) -> str:
    out_dir: str = os.path.join(dir, *args)

    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def get_feature(features: list[torch.Tensor], idx: int, grid_normed: torch.Tensor):
    feature_i = features[idx].permute(2, 0, 1)[None]  # [1, H_p, W_p, Feat]
    # feature_i_sampled shape: [H, W, Feat].
    feature_i_sampled = F.grid_sample(feature_i, grid_normed[None], align_corners=True)[0].permute(1, 2, 0)
    return feature_i_sampled


if __name__ == "__main__":
    device = "cuda"
    feature_name = "dino"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="", help="dataset dir")
    parser.add_argument("--cycle_th", type=float, default=3.0, help="threshold for cycle consistency error")
    parser.add_argument("--similarity_th", type=float, default=0.5, help="threshold for cosine similarity")
    args = parser.parse_args()

    # Get image files
    img_files = sorted((Path(args.data_dir) / "color").glob("*.png"))
    num_imgs = len(img_files)

    # Prepare output directories
    out_flow_stats_file = os.path.join(args.data_dir, "flow_stats.json")
    out_dir = opj_and_mkdir(args.data_dir, "raft_masks")

    count_out_dir = opj_and_mkdir(args.data_dir, "count_maps")

    # check h and w size.
    h, w, _ = imageio.v3.imread(img_files[0]).shape
    grid = generate_grid(h, w, device=device, homogeneous=False).permute(2, 0, 1)[None]

    grid_normed = normalize_coords(grid.squeeze().permute(1, 2, 0), h, w, no_shift=True)  # [h, w, 2]

    features = [
        torch.from_numpy(np.load(Path(args.data_dir) / "features" / feature_name / (Path(img_file).stem + ".npy")))
        .float()
        .to(device)
        for img_file in img_files
    ]
    # Confusion Matrix
    # Exhaustive
    flow_stats = {}
    count_maps = np.zeros((num_imgs, h, w), np.uint16)
    for i in range(num_imgs):
        imgname_i = os.path.basename(img_files[i])
        feature_i_sampled = get_feature(features, i, grid_normed)
        for j in range(num_imgs):
            if i == j:
                continue
            frame_interval = abs(i - j)

            imgname_j = os.path.basename(img_files[j])

            # step(1)
            # Load Estimated Optical Flow by RAFT model.
            # computed discrepancy i -> j and j -> i.
            # i ->
            flow_forward = np.load(Path(args.data_dir) / "raft_exhaustive" / f"{imgname_i}_{imgname_j}.npy")
            flow_forward = torch.from_numpy(flow_forward).float().permute(2, 0, 1)[None].cuda()
            coord2 = flow_forward + grid  # absolute coordinate.
            coord2_normed = normalize_coords(coord2, h, w, no_shift=True)
            # j -> i
            flow_backward = np.load(Path(args.data_dir) / "raft_exhaustive" / f"{imgname_j}_{imgname_i}.npy")
            flow_backward = torch.from_numpy(flow_backward).float().permute(2, 0, 1)[None].cuda()
            feature_21_sampled = F.grid_sample(flow_backward, coord2_normed[None], align_corners=True)[0].permute(
                1, 2, 0
            )
            # discrepancy i -> j and j -> i
            map_i = flow_forward + feature_21_sampled
            fb_discrepancy = torch.norm(map_i.squeeze(), dim=0)
            mask_cycle = fb_discrepancy < args.cycle_th

            # step(2)
            # Load Foundation model Feature.
            # cosine similarity for i and j.
            feature_j_sampled = get_feature(features, j, grid_normed)
            sim_ij = F.cosine_similarity(feature_i_sampled, feature_j_sampled, dim=-1)
            feature_mask = sim_ij > args.similarity_th

            mask_cycle = mask_cycle * feature_mask if frame_interval >= 3 else mask_cycle

            # only keep correspondences for occluded pixels if the correspondences are
            # inconsistent in the first cycle but consistent in the second cycle
            # and if the two frames are adjacent enough (interval < 3)
            if frame_interval < 3:
                coord_21 = grid + map_i  # [1, 2, h, w]
                coord_21_normed = normalize_coords(coord_21.squeeze().permute(1, 2, 0), h, w)  # [h, w, 2]

                flow_22 = F.grid_sample(flow_forward, coord_21_normed[None], align_corners=True)
                fbf_discrepancy = torch.norm((coord_21 + flow_22 - flow_forward - grid).squeeze(), dim=0)

                mask_in_range = (coord2_normed.min(dim=-1)[0] >= -1) * (coord2_normed.max(dim=-1)[0] <= 1)
                mask_occluded = (fbf_discrepancy < args.cycle_th) * (fb_discrepancy > args.cycle_th * 1.5)
                mask_occluded *= mask_in_range

            else:
                mask_occluded = torch.zeros_like(mask_cycle)

            out_mask = torch.stack([mask_cycle, mask_occluded, torch.zeros_like(mask_cycle)], dim=-1).cpu().numpy()
            imageio.imwrite(f"{out_dir}/{imgname_i}_{imgname_j}.png", (255 * out_mask.astype(np.uint8)))

            if imgname_i not in flow_stats.keys():
                flow_stats[imgname_i] = {}
            flow_stats[imgname_i][imgname_j] = np.sum(out_mask).item()
            count_maps[i] += out_mask.sum(axis=-1).astype(np.uint16)

    with open(out_flow_stats_file, "w") as f:
        json.dump(flow_stats, f)

    for i in range(num_imgs):
        save_name = Path(count_out_dir) / img_files[i].with_suffix(".png").name
        imageio.imwrite(save_name, count_maps[i])
