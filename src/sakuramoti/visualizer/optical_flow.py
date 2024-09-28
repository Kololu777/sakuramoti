# Based on https://github.com/tomrunia/OpticalFlow_Visualization
# This code is Pytorch version.

import torch


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = torch.zeros((3, ncols))
    col = 0

    # RY
    colorwheel[0, 0:RY] = 255
    colorwheel[1, 0:RY] = torch.floor(255 * torch.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[0, col : col + YG] = 255 - torch.floor(255 * torch.arange(0, YG) / YG)
    colorwheel[1, col : col + YG] = 255
    col = col + YG
    # GC
    colorwheel[1, col : col + GC] = 255
    colorwheel[2, col : col + GC] = torch.floor(255 * torch.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[1, col : col + CB] = 255 - torch.floor(255 * torch.arange(CB) / CB)
    colorwheel[2, col : col + CB] = 255
    col = col + CB
    # BM
    colorwheel[2, col : col + BM] = 255
    colorwheel[0, col : col + BM] = torch.floor(255 * torch.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[2, col : col + MR] = 255 - torch.floor(255 * torch.arange(MR) / MR)
    colorwheel[0, col : col + MR] = 255
    return colorwheel


def flow_to_image(flow_uv: torch.Tensor, flow_value_max: float | None = None, epsilon: float = 1e-5):
    if flow_value_max is not None:
        flow_uv = torch.clip(flow_uv, 0, flow_value_max)
    u = flow_uv[0, :, :]
    v = flow_uv[1, :, :]
    rad = torch.sqrt(torch.square(u) + torch.square(v))
    max_rad = torch.max(rad)

    u = u / (max_rad + epsilon)
    v = v / (max_rad + epsilon)

    return flow_uv_to_colors(u, v)


def flow_uv_to_colors(u: torch.Tensor, v: torch.Tensor):
    flow_image = torch.zeros((3, u.shape[0], u.shape[1]), dtype=torch.uint8)
    colorwheel = make_colorwheel().to(device=u.device)

    ncols = colorwheel.shape[1]
    rad = torch.sqrt(torch.square(u) + torch.square(v))
    a = torch.atan2(-v, -u) / torch.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = torch.floor(fk).to(torch.int64)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0

    col0 = (colorwheel[:, k0] / 255).to(dtype=u.dtype)
    col1 = (colorwheel[:, k1] / 255).to(dtype=u.dtype)
    col = (1 - f.unsqueeze(0)) * col0 + f.unsqueeze(0) * col1

    idx = (rad <= 1).to(dtype=torch.bool)
    rad = rad * idx

    col = 1 - rad * (1 - col * idx)
    col = col + (col * ~idx * 0.75)

    flow_image = torch.floor(255 * col)
    return flow_image
