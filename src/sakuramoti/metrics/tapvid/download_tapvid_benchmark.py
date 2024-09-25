import os
import zipfile

import torch


def download_davis_val(save_dir: str):
    # DAVIS val set.
    url_base = "https://storage.googleapis.com/dm-tapnet/"
    zip_name = "tapvid_davis.zip"
    zip_save_file_path = os.path.join(save_dir, zip_name)
    torch.hub.download_url_to_file(url_base + zip_name, zip_save_file_path)
    with zipfile.ZipFile(zip_save_file_path, "r") as zp:
        zp.extractall(save_dir)
    os.remove(zip_save_file_path)


def download_kinetics_val(save_dir: str):
    # Kinetics val set.
    url_base = "https://storage.googleapis.com/dm-tapnet/"
    zip_name = "tapvid_kinetics.zip"
    zip_save_file_path = os.path.join(save_dir, zip_name)
    torch.hub.download_url_to_file(url_base + zip_name, zip_save_file_path)
    with zipfile.ZipFile(zip_save_file_path, "r") as zp:
        zp.extractall(save_dir)
    os.remove(zip_save_file_path)


def download_rgb_stacking_val(save_dir: str):
    # Deepmind Robotics videos.
    url_base = "https://storage.googleapis.com/dm-tapnet/"
    zip_name = "tapvid_rgb_stacking.zip"
    zip_save_file_path = os.path.join(save_dir, zip_name)
    torch.hub.download_url_to_file(url_base + zip_name, zip_save_file_path)
    with zipfile.ZipFile(zip_save_file_path, "r") as zp:
        zp.extractall(save_dir)
    os.remove(zip_save_file_path)


def download_tapvid_benchmark(
    save_dir: str,
    is_download_davis: bool = True,
    is_download_kinetics: bool = True,
    is_download_rgb_stacking: bool = True,
):
    if is_download_davis:
        download_davis_val(save_dir)

    if is_download_kinetics:
        download_kinetics_val(save_dir)

    if is_download_rgb_stacking:
        download_rgb_stacking_val(save_dir)


if __name__ == "__main__":
    download_tapvid_benchmark(
        "/mnt/wd_sn770_oss/imc2024_code/sakuramoti/data", is_download_davis=False
    )
