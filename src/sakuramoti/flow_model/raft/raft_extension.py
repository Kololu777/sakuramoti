import torch.nn.functional as F
from torch import Tensor
from sakuramoti.geometry.grid import generate_grid, normalize_coords
from sakuramoti.flow_model.raft.raft import RAFT


class RAFTD:
    """Directly computing RAFT flow between query and target frames.
    Reference:
    [1] OmniMotion: https://arxiv.org/pdf/2306.05422
    """

    def __init__(self, raft_model: RAFT, iters: int = 20):
        self.raft_model = raft_model
        self.iters = iters

    def compute(self, i_frame: Tensor, j_frame: Tensor, flow_init: Tensor | None = None) -> tuple[Tensor, Tensor]:
        return self.raft_model.pred(i_frame, j_frame, iters=self.iters, flow_init=flow_init)


class RAFTC:
    """Chain flow using RAFT.
    Reference:
    [1] OmniMotion: https://arxiv.org/pdf/2306.05422
    """

    def __init__(self, raft_model: RAFT, iters: int = 20):
        self.raft_model = raft_model
        self.iters = iters
        self.__step_flow_list = []

    def __add_step_flow(self, flow_current: Tensor):
        self.__step_flow_list.append(flow_current)

    def compute(self, video_frames: list[Tensor], is_step_flow_list: bool = False):
        """
        Args:
            video_frames: list of video frames. Shape: (1, C, H, W)
            is_step_flow_list: if True, return step flow list.
        """
        start_idx = 0
        end_idx = len(video_frames)
        h, w = video_frames[0].shape[1:]
        device = self.raft_model.device
        grid = generate_grid(h, w, device)[None]
        flow_low, flow_up = self.raft_model.pred(
            video_frames[start_idx], video_frames[start_idx + 1], iters=self.iters, flow_init=None
        )
        accumulated_flow = flow_up
        curr_coords = grid + accumulated_flow
        curr_coords_normed = normalize_coords(curr_coords, h, w)
        if is_step_flow_list:
            self.__add_step_flow(accumulated_flow)

        for j in range(1, end_idx, 1):
            flow_low, flow_up = self.raft_model.pred(
                video_frames[j], video_frames[j + 1], iters=self.iters, flow_init=flow_low
            )

            flow_current = F.grid_sample(flow_up, curr_coords_normed, align_corners=True)
            accumulated_flow += flow_current
            if is_step_flow_list:
                self.__add_step_flow(accumulated_flow)

        if is_step_flow_list:
            return self.__step_flow_list

        return accumulated_flow
