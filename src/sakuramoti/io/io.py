import numpy as np
import torch
from PIL import Image
from torch import Tensor


def load_image(path_file: str, device: str) -> Tensor:
    img = np.array(Image.open(path_file)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    img = 2 * (img / 255.0) - 1.0
    return img[None].to(device)
