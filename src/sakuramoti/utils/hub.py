from __future__ import annotations

import os
import re
import sys
import shutil
import zipfile
from pathlib import Path

import torch
from torch.hub import get_dir, download_url_to_file
from torch.serialization import MAP_LOCATION

HASH_REGEX = re.compile(r"-([a-f0-9]*)\.")
TORCH_WEIGHT_SUFFIX = [".pt", ".pth"]


def load_state_dict_from_zip_url(
    url,
    target_file_name: str,
    model_dir: str | None = None,
    map_location: MAP_LOCATION = None,
    progress: bool = True,
    check_hash: bool = False,
    file_name: str | None = None,
    weights_only: bool = False,
) -> dict[str, any]:
    r"""This function, which is similar to torch.hub.load_state_dict_from_url but allows loading
    a specific model from multiple files within a ZIP archive.

    Args:
        url: URL of the object to download
        target_file_name: Specify the filename of model weight file within the ZIP.
        model_dir: directory in which to save the object
        map_location: a function or a dict specifying how to remap storage locations (see torch.load)
        progress: whether or not to display a progress bar to stderr.
            Default: True
        check_hash: If True, the filename part of the URL should follow the naming convention
            ``filename-<sha256>.ext`` where ``<sha256>`` is the first eight or more
            digits of the SHA256 hash of the contents of the file. The hash is used to
            ensure unique names and to verify the contents of the file.
            Default: False
        file_name: name for the downloaded file. Filename from ``url`` will be used if not set.
        weights_only: If True, only weights will be loaded and no complex pickled objects.
            Recommended for untrusted sources. See :func:`~torch.load` for more details.
    """

    if model_dir is None:
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")

    filename = os.path.basename(url)
    if file_name is not None:
        filename = file_name
    target_file = os.path.join(model_dir, target_file_name)
    if not os.path.exists(target_file):
        zip_file = os.path.join(model_dir, filename)
        sys.stderr.write(f'Downloading: "{url}" to {zip_file}\n')
        hash_prefix = None
        if check_hash:
            r = HASH_REGEX.search(filename)  # r is Optional[Match[str]]
            hash_prefix = r.group(1) if r else None
        download_url_to_file(url, zip_file, hash_prefix, progress=progress)

        # mv
        with zipfile.ZipFile(file=zip_file) as f:
            # extract
            f.extractall(model_dir)
            infolist = f.infolist()

        for info in infolist:
            file_path = os.path.join(model_dir, info.filename)
            filename_in_zip_file = os.path.basename(file_path)
            mv_filename = os.path.join(model_dir, filename_in_zip_file)
            if (
                not os.path.exists(os.path.join(mv_filename))
                and Path(filename_in_zip_file).suffix in TORCH_WEIGHT_SUFFIX
            ):
                sys.stderr.write(f'mv: "{file_path}" to {mv_filename}\n')
                shutil.move(file_path, model_dir)
        os.remove(zip_file)
        # sys.stderr.write(f'rm -rf: "{str(Path(file_path)).parent}"\n')
        # shutil.rmtree(str(Path(file_path).parent))
    return torch.load(target_file, map_location=map_location, weights_only=weights_only)
