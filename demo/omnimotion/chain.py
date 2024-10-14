import os
import json
from pathlib import Path
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import imageio
import torch.nn.functional as F
from sakuramoti.geometry.grid import generate_grid, normalize_coords


@dataclass
class ChainArgs:
    feature_name = "dino"
    scene_dir: str
    device = "cuda"


def join_and_make_dir(path: str, *args) -> str:
    join_path = os.path.join(path, *args)
    os.makedirs(join_path, exist_ok=True)
    return join_path


def read_image(img_file: str):
    img = cv2.imread(img_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def run(args: ChainArgs):
    def get_flow_and_mask(img_files_idx: str, img_files_jdx: str, is_return_direct_mask=False):
        # imgname_i = os.path.basename(img_files[img_files_idx])
        # imgname_j = os.path.basename(img_files[img_files_jdx])
        # optical flow
        flow_file = os.path.join(args.scene_dir, "raft_exhaustive", f"{img_files_idx}_{img_files_jdx}.npy")
        flow = np.load(flow_file)
        flow = torch.from_numpy(flow).float()[None].cuda()  # [b, h, w, 2]

        # masks
        mask_file = flow_file.replace("raft_exhaustive", "raft_masks").replace(".npy", "_mask.png")
        masks = read_image(mask_file)
        cycle_mask = masks[..., 0] > 0
        occlusion_mask = masks[..., 1] > 0

        # direact mask
        if is_return_direct_mask:
            direct_mask = cycle_mask | occlusion_mask
            direct_mask = torch.from_numpy(direct_mask)[None]
            return flow, cycle_mask, occlusion_mask, direct_mask
        else:
            return flow, cycle_mask, occlusion_mask

    img_files = sorted((Path(args.scene_dir) / "color").glob("*.png"))

    save_ouput = {
        "out_dir": join_and_make_dir(args.scene_dir, "raft_exhaustive"),
        "out_mask_dir": join_and_make_dir(args.scene_dir, "raft_masks"),
        "count_out_dir": join_and_make_dir(args.scene_dir, "count_maps"),
        "out_flow_stats_file": join_and_make_dir(args.scene_dir, "flow_stats.json"),
    }

    images = [torch.from_numpy(imageio.imread(img_file)) for img_file in img_files]
    features = [
        torch.from_numpy(
            np.load(os.path.join(args.scene_dir, "features", args.feature_name, os.path.basename(img_file) + ".npy"))
        )
        .float()
        .to(args.device)
        for img_file in img_files
    ]

    h, w = images[0].shape[:2]
    num_images = len(img_files)
    grid = generate_grid(images.shape[0], images.shape[1], args.device, normalize=True)
    count_maps = np.zeros((num_images, h, w), dtype=np.uint16)
    flow_stats = {}

    def process(i_range=(0, num_images - 1, 1), j_range=(1, num_images, 1)):
        for i in range(i_range[0], i_range[1], i_range[2]):
            start_flow, start_cycle_mask, _ = get_flow_and_mask(
                img_files_idx=img_files[i].stem, img_files_jdx=img_files[i + 1].stem, is_return_direct_mask=False
            )
            feature_i = features[i].permute(2, 0, 1)[None]
            feature_i = F.interpolate(feature_i, size=start_flow.shape[1:3], mode="bilinear")
            # flow
            accumulated_flow = start_flow
            # mask
            accumulated_cycle_mask = start_cycle_mask

            for j in range(i + j_range[0], j_range[1], j_range[2]):
                direct_flow, direct_cycle_mask, direct_occlusion_mask, direct_masks = get_flow_and_mask(i, j, True)

                accumulated_flow[direct_masks] = direct_flow[direct_masks]

                curr_coords = grid + accumulated_flow
                curr_coords_normed = normalize_coords(curr_coords, h, w)

                feature_j = features[j].permute(2, 0, 1)[None]
                feature_j_sampled = F.grid_sample(feature_j, curr_coords_normed, align_corners=True)
                feature_sim = torch.cosine_similarity(feature_i, feature_j_sampled, dim=1).squeeze(0).cpu().numpy()
                image_j_sampled = F.grid_sample(images[j], curr_coords_normed, align_corners=True).squeeze()
                rgb_sim = torch.norm(images[i] - image_j_sampled, dim=1).squeeze(0).cpu().numpy()

                accumulated_cycle_mask *= (feature_sim > 0.5) * (rgb_sim < 0.3)
                accumulated_cycle_mask[direct_cycle_mask] = True
                accumulated_cycle_mask[direct_occlusion_mask] = False

                # save
                # np.save(os.path.join(out_dir, '{}_{}.npy'.format(imgname_i, imgname_j)), accumulated_flow[0].cpu().numpy())
                out_mask = np.concatenate(
                    [255 * accumulated_cycle_mask[..., None].astype(np.uint8), direct_masks[..., 1:]], axis=-1
                )
                count_maps[i] += (out_mask / 255).sum(axis=-1).astype(np.uint16)
                if img_files[i].stem not in flow_stats.keys():
                    flow_stats[img_files[i].stem] = {}
                flow_stats[img_files[i].stem][img_files[j].stem] = int(np.sum(out_mask / 255))
                if j == num_images - 1:
                    continue

                curr_flow, curr_cycle_mask, _ = get_flow_and_mask(
                    img_files[j].stem, img_files[j + 1].stem, is_return_direct_mask=False
                )

                flow_curr_sampled = F.grid_sample(curr_flow, curr_coords_normed, align_corners=True)
                curr_cycle_mask_sampled = F.grid_sample(curr_cycle_mask, curr_coords_normed, align_corners=True)

                # Update
                accumulated_flow += flow_curr_sampled
                accumulated_cycle_mask *= curr_cycle_mask_sampled

    process()
    process(i_range=(num_images - 1, 0, -1), j_range=(-1, -1, -1))

    with open(save_ouput["out_flow_stats_file"], "w") as fp:
        json.dump(flow_stats, fp)

    for i in range(num_images):
        img_name = os.path.basename(img_files[i])
        imageio.imwrite(os.path.join(save_ouput["count_out_dir"], img_name.replace(".jpg", ".png")), count_maps[i])


# join_and_make_dir("./", "raft_exhaustive")
