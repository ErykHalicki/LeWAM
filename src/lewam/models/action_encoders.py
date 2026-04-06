import torch
import torch.nn as nn

from lewam.models.common import make_mlp


class StateEncoder(nn.Module):
    """
    MLP: (B, state_dim) → (B, model_dim)

    Output is added to the timestep embedding for adaLN conditioning in the DiT.
    """
    def __init__(self, state_dim: int, model_dim: int):
        super().__init__()
        self.mlp = make_mlp(state_dim, model_dim, model_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.mlp(state)


class ActionEncoder(nn.Module):
    """
    Per-timestep MLP: (B, action_horizon, action_dim) → (B, action_horizon, model_dim)

    Applied to noisy action samples during flow matching.
    """
    def __init__(self, action_dim: int, model_dim: int):
        super().__init__()
        self.mlp = make_mlp(action_dim, model_dim, model_dim)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.mlp(actions)

class ActionPreprocessor(nn.Module):
    """
    Normalizes and unnormalizes relative actions and state using precomputed stats.

    Stats file stores q1..q99, mean, std for both rel_action and state.
    The norm_strategy selects which percentiles to clip to before z-scoring:
        "q1_q99"  -> clip to 1st/99th percentile (default)
        "q5_q95"  -> clip to 5th/95th percentile
        "q10_q90" -> clip to 10th/90th percentile
        "none"    -> no clipping, just z-score
    """

    def __init__(self, stats: dict, norm_strategy: str = "q1_q99"):
        super().__init__()
        self.norm_strategy = norm_strategy

        if norm_strategy != "none":
            lo_key, hi_key = norm_strategy.split("_")
        else:
            lo_key, hi_key = "q1", "q99"

        for key in ("rel_action", "state"):
            self.register_buffer(f"{key}_lo", stats[key][lo_key].float())
            self.register_buffer(f"{key}_hi", stats[key][hi_key].float())
            self.register_buffer(f"{key}_mean", stats[key]["mean"].float())
            self.register_buffer(f"{key}_std", stats[key]["std"].float())

    def normalize_rel_action(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_strategy != "none":
            x = x.clamp(self.rel_action_lo, self.rel_action_hi)
        return (x - self.rel_action_mean) / self.rel_action_std

    def unnormalize_rel_action(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.rel_action_std + self.rel_action_mean

    def normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_strategy != "none":
            x = x.clamp(self.state_lo, self.state_hi)
        return (x - self.state_mean) / self.state_std

    def unnormalize_state(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.state_std + self.state_mean
