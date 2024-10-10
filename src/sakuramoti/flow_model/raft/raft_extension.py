from torch import Tensor
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
