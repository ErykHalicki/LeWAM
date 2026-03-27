import torch
import torch.nn as nn
import torch.nn.functional as F


class SIGReg(nn.Module):
    """
    Sketch Isotropic Gaussian Regularizer from LeWorldModel (single-GPU).
    Encourages embeddings to be isotropically Gaussian via the Epps-Pulley statistic.
    Reference: https://github.com/lucas-maes/le-wm/blob/main/module.py
    """
    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """proj: (T, B, D)"""
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        return (err @ self.weights).mean() * proj.size(-2)


def _euler_solve(dit, x0, past_frames, l, state, l_mask, num_steps):
    dt = 1.0 / num_steps
    x = x0
    for i in range(num_steps):
        t = torch.full((x.shape[0],), i * dt, device=x.device)
        x = x + dt * dit(x, t, past_frames, l, state, l_mask)
    return x


def teacher_forcing_loss(dit, idm, x1, past_frames, current_frames, l, state, actions, l_mask=None):
    """
    DiT flow matching loss and IDM loss computed independently with no gradient sharing.
    IDM sees GT future frames.

    Returns (flow_loss, idm_loss).

    x1:             (B, K*H*W, in_dim)  clean future VJEPA2 patch embeddings
    past_frames:    (B, T*H*W, in_dim)
    current_frames: (B, T*H*W, in_dim)
    l:              (B, S, lang_dim)
    state:          (B, state_dim)
    actions:        (B, chunk_len, action_dim)
    l_mask:         (B, S) True = ignore
    """
    B = x1.shape[0]

    x0 = torch.randn_like(x1)
    t = torch.rand(B, device=x1.device)
    x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
    flow_loss = F.mse_loss(dit(x_t, t, past_frames, l, state, l_mask), x1 - x0)

    idm_loss = F.mse_loss(idm(current_frames, x1, state), actions)

    return flow_loss, idm_loss


def detached_ode_loss(dit, idm, x1, past_frames, current_frames, l, state, actions, l_mask=None, num_steps=10):
    """
    Full Euler ODE solve with DiT output detached. Only IDM trains.
    Use when DiT is already trained and you want to adapt IDM to ODE-predicted futures.

    Returns idm_loss.

    x1:             (B, K*H*W, in_dim)  used only for shape/device to sample x0
    past_frames:    (B, T*H*W, in_dim)
    current_frames: (B, T*H*W, in_dim)
    l:              (B, S, lang_dim)
    state:          (B, state_dim)
    actions:        (B, chunk_len, action_dim)
    l_mask:         (B, S) True = ignore
    num_steps:      Euler integration steps
    """
    x0 = torch.randn_like(x1)
    with torch.no_grad():
        x1_pred = _euler_solve(dit, x0, past_frames, l, state, l_mask, num_steps)
    return F.mse_loss(idm(current_frames, x1_pred, state), actions)


def end_to_end_loss(dit, idm, x1, past_frames, current_frames, l, state, actions, l_mask=None, ode_solve_steps=1):
    """
    Differentiable Euler ODE solve. Flow loss is averaged over all ODE steps.
    Gradients flow through IDM and through the full ODE chain into DiT.

    # TODO: compare fixed-step backprop (current) against adjoint method for large
    #       ode_solve_steps, where gradient chains through DiT may become expensive.
    #       I'm not sure how it works, but here is a blog post. Maybe there are some other papers that explore this?
    #       https://ilya.schurov.com/post/adjoint-method/

    The flow loss is evaluated at evenly-spaced timesteps (0, dt, 2dt, ...) rather
    than random t ~ Uniform[0,1]. This is a slight bias from standard flow matching
    but is consistent with the ODE trajectory.

    Returns (flow_loss, idm_loss).

    x1:             (B, K*H*W, in_dim)  clean future VJEPA2 patch embeddings
    past_frames:    (B, T*H*W, in_dim)
    current_frames: (B, T*H*W, in_dim)
    l:              (B, S, lang_dim)
    state:          (B, state_dim)
    actions:        (B, chunk_len, action_dim)
    l_mask:         (B, S) True = ignore
    ode_solve_steps: number of Euler steps; 1 is cheapest, more gives better x1_pred
    """
    B = x1.shape[0]
    dt = 1.0 / ode_solve_steps

    x = torch.randn_like(x1)
    flow_losses = []
    for i in range(ode_solve_steps):
        t = torch.full((B,), i * dt, device=x1.device)
        v = dit(x, t, past_frames, l, state, l_mask)
        flow_losses.append(F.mse_loss(v, x1 - x))
        x = x + dt * v

    flow_loss = torch.stack(flow_losses).mean()
    idm_loss = F.mse_loss(idm(current_frames, x, state), actions)

    return flow_loss, idm_loss
