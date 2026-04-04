import pytest
import torch
from wam.models.lewam import LeWAM, TimestepEmbedder, FinalLayer


DIM = 128
DEPTH = 2
HEADS = 4
B = 2

TINY_CFG = dict(
    model_dim=DIM, depth=DEPTH, num_heads=HEADS,
    num_context_frames=4, num_future_frames=4,
    frame_latent_h=2, frame_latent_w=2, fps=10.0,
    action_dim=6, state_dim=6,
    vlm_model_id=None,
)


@pytest.fixture
def model():
    return LeWAM(**TINY_CFG)


@pytest.fixture
def inputs(model):
    return dict(
        x_t_video=torch.randn(B, model.N_fut, model.VJEPA_DIM),
        x_t_action=torch.randn(B, model.N_act, 6),
        context_tokens=torch.randn(B, model.N_ctx, model.VJEPA_DIM),
        t=torch.rand(B),
        state=torch.randn(B, 6),
    )


# ── TimestepEmbedder ────────────────────────────────────────────────────────

class TestTimestepEmbedder:
    def test_output_shape(self):
        te = TimestepEmbedder(DIM)
        t = torch.rand(B)
        assert te(t).shape == (B, DIM)

    def test_different_timesteps_give_different_embeddings(self):
        te = TimestepEmbedder(DIM)
        e0 = te(torch.tensor([0.0]))
        e1 = te(torch.tensor([1.0]))
        assert not torch.allclose(e0, e1)


# ── FinalLayer ───────────────────────────────────────────────────────────────

class TestFinalLayer:
    def test_output_shape(self):
        fl = FinalLayer(DIM, 768)
        x = torch.randn(B, 10, DIM)
        cond = torch.randn(B, DIM)
        assert fl(x, cond).shape == (B, 10, 768)


# ── Attention mask ───────────────────────────────────────────────────────────

class TestAttnMask:
    def test_mask_shape(self, model):
        N = model.N_ctx + model.N_fut + model.N_act
        assert model.attn_mask.shape == (1, 1, N, N)

    def test_context_block_causal(self, model):
        mask = model.attn_mask.squeeze()
        spatial = model.frame_latent_h * model.frame_latent_w
        bs = 2 * spatial
        num_blocks = model.N_ctx // bs
        for i in range(num_blocks):
            r0, r1 = i * bs, (i + 1) * bs
            assert mask[r0:r1, :r1].all(), f"context block {i} should see itself and past blocks"
            if r1 < model.N_ctx:
                assert not mask[r0, r1], f"context block {i} should NOT see next block"

    def test_context_blocks_bidirectional_within(self, model):
        mask = model.attn_mask.squeeze()
        spatial = model.frame_latent_h * model.frame_latent_w
        bs = 2 * spatial
        assert mask[0, bs - 1] and mask[bs - 1, 0], "spatial tokens within a block must be bidirectional"

    def test_future_sees_all_context(self, model):
        mask = model.attn_mask.squeeze()
        C, F = model.N_ctx, model.N_fut
        assert mask[C:C+F, :C].all()

    def test_future_sees_corresponding_actions(self, model):
        mask = model.attn_mask.squeeze()
        C, F, A = model.N_ctx, model.N_fut, model.N_act
        assert mask[C + F - 1, C + F + A - 1], "last future token should see last action"
        assert mask[C, C + F], "first future token should see first action"

    def test_actions_see_all_context(self, model):
        mask = model.attn_mask.squeeze()
        C, F = model.N_ctx, model.N_fut
        assert mask[C+F:, :C].all()

    def test_actions_block_causal(self, model):
        mask = model.attn_mask.squeeze()
        C, F, A = model.N_ctx, model.N_fut, model.N_act
        action_block = mask[C+F:, C+F:]
        assert action_block[0, 0], "first action should see itself"
        assert action_block[A-1, A-1], "last action should see itself"
        assert action_block[A-1, 0], "last action should see first action"

    def test_future_block_causal(self):
        m = LeWAM._build_attn_mask(N_ctx=4, N_fut=6, N_act=2, spatial=1, block_size=2).squeeze()
        C, F, bs = 4, 6, 2
        num_fut_blocks = F // bs
        for i in range(num_fut_blocks):
            r0 = C + i * bs
            assert m[r0, C + (i + 1) * bs - 1], "future block should see own last token"
            if i + 1 < num_fut_blocks:
                assert not m[r0, C + (i + 1) * bs], "future block should NOT see next block"

    @pytest.mark.parametrize("C,F,A,bs", [(4, 6, 2, 2), (8, 8, 4, 2), (0, 4, 4, 1)])
    def test_mask_self_attendance(self, C, F, A, bs):
        mask = LeWAM._build_attn_mask(C, F, A, spatial=1, block_size=bs).squeeze()
        N = C + F + A
        assert mask.shape == (N, N)
        for r in range(N):
            assert mask[r, r], f"Token {r} should attend to itself"


# ── Constructor ──────────────────────────────────────────────────────────────

class TestConstructor:
    def test_from_size(self):
        m = LeWAM.from_size(
            "baby",
            num_context_frames=4, num_future_frames=4,
            frame_latent_h=2, frame_latent_w=2, fps=10.0,
            action_dim=6, state_dim=6,
            vlm_model_id=None,
        )
        assert m.model_dim == 256

    def test_config_stored(self, model):
        assert model.config["model_dim"] == DIM
        assert model.config["depth"] == DEPTH

    def test_count_params(self, model):
        n = model.count_params(millions=False)
        assert n > 0

    def test_action_horizon_derived(self, model):
        expected = int((model.num_future_frames / model.fps) * model.action_fps)
        assert model.action_horizon == expected


# ── Forward pass ─────────────────────────────────────────────────────────────

class TestForward:
    def test_output_shapes(self, model, inputs):
        v_vid, v_act = model(**inputs)
        assert v_vid.shape == (B, model.N_fut, model.VJEPA_DIM)
        assert v_act.shape == (B, model.N_act, 6)

    def test_no_nan(self, model, inputs):
        v_vid, v_act = model(**inputs)
        assert not torch.isnan(v_vid).any()
        assert not torch.isnan(v_act).any()
        assert not torch.isinf(v_vid).any()
        assert not torch.isinf(v_act).any()

    def test_gradient_flows_to_video_input(self, model, inputs):
        inputs["x_t_video"].requires_grad_(True)
        v_vid, _ = model(**inputs)
        v_vid.sum().backward()
        assert inputs["x_t_video"].grad is not None

    def test_gradient_flows_to_action_input(self, model, inputs):
        inputs["x_t_action"].requires_grad_(True)
        _, v_act = model(**inputs)
        v_act.sum().backward()
        assert inputs["x_t_action"].grad is not None

    def test_with_lang_tokens(self):
        m = LeWAM(**{**TINY_CFG, "vlm_model_id": None})
        lang = torch.randn(B, 5, DIM)
        # vlm_model_id=None means num_ca_sources=1, so lang_tokens are ignored
        # just verify it doesn't crash
        v, a = m(
            x_t_video=torch.randn(B, m.N_fut, m.VJEPA_DIM),
            x_t_action=torch.randn(B, m.N_act, 6),
            context_tokens=torch.randn(B, m.N_ctx, m.VJEPA_DIM),
            t=torch.rand(B), state=torch.randn(B, 6),
        )
        assert v.shape == (B, m.N_fut, m.VJEPA_DIM)


# ── ODE solve ────────────────────────────────────────────────────────────────

class TestODESolve:
    def test_output_shapes(self, model):
        model.eval()
        ctx = torch.randn(1, model.N_ctx, model.VJEPA_DIM)
        state = torch.randn(1, 6)
        v, a = model.ode_solve(ctx, state, num_steps=3)
        assert v.shape == (1, model.N_fut, model.VJEPA_DIM)
        assert a.shape == (1, model.N_act, 6)

    def test_no_nan(self, model):
        model.eval()
        ctx = torch.randn(1, model.N_ctx, model.VJEPA_DIM)
        state = torch.randn(1, 6)
        v, a = model.ode_solve(ctx, state, num_steps=5)
        assert not torch.isnan(v).any()
        assert not torch.isnan(a).any()

    def test_no_grad_during_solve(self, model):
        model.eval()
        ctx = torch.randn(1, model.N_ctx, model.VJEPA_DIM, requires_grad=True)
        state = torch.randn(1, 6)
        v, _ = model.ode_solve(ctx, state, num_steps=3)
        assert not v.requires_grad


# ── Runtime config ───────────────────────────────────────────────────────────

class TestRuntimeConfig:
    def test_set_fps(self, model):
        old_t = model.context_pos.t_ids.clone()
        model.set_fps(20.0)
        assert not torch.allclose(old_t, model.context_pos.t_ids)
        assert model.fps == 20.0

    def test_set_patch_grid(self, model):
        old_N = model.N_ctx
        model.set_patch_grid(3, 3)
        assert model.N_ctx != old_N
        assert model.attn_mask.shape[-1] == model.N_ctx + model.N_fut + model.N_act

    def test_set_patch_grid_forward_still_works(self, model):
        model.set_patch_grid(3, 3)
        v, a = model(
            x_t_video=torch.randn(B, model.N_fut, model.VJEPA_DIM),
            x_t_action=torch.randn(B, model.N_act, 6),
            context_tokens=torch.randn(B, model.N_ctx, model.VJEPA_DIM),
            t=torch.rand(B), state=torch.randn(B, 6),
        )
        assert v.shape == (B, model.N_fut, model.VJEPA_DIM)


# ── Overfit ──────────────────────────────────────────────────────────────────

class TestOverfit:
    def test_single_sample_overfit(self):
        torch.manual_seed(42)
        m = LeWAM(**TINY_CFG)
        m.train()

        x_vid = torch.randn(1, m.N_fut, m.VJEPA_DIM)
        x_act = torch.randn(1, m.N_act, 6)
        ctx = torch.randn(1, m.N_ctx, m.VJEPA_DIM)
        t = torch.tensor([0.5])
        state = torch.randn(1, 6)
        target_vid = torch.randn(1, m.N_fut, m.VJEPA_DIM)
        target_act = torch.randn(1, m.N_act, 6)

        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        for _ in range(300):
            v_vid, v_act = m(x_vid, x_act, ctx, t, state)
            loss = (
                torch.nn.functional.mse_loss(v_vid, target_vid)
                + torch.nn.functional.mse_loss(v_act, target_act)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()

        assert loss.item() < 0.1, f"Failed to overfit (loss={loss.item():.4f})"
