import pytest
import torch
from lewam.models.common import (
    Block, RoPE3D, SelfAttention, CrossAttention,
    PatchPositionIds,
    make_video_pos_ids, make_action_pos_ids, concat_pos_ids,
    make_mlp, SwiGLULinear, modulate,
)

HIDDEN = 64
HEADS = 4
HEAD_DIM = HIDDEN // HEADS
B = 2


# ── Position ID creation ─────────────────────────────────────────────────────

class TestPositionIds:
    def test_video_pos_ids_shape(self):
        t, h, w = make_video_pos_ids(num_frames=4, patch_h=3, patch_w=3, fps=10.0)
        assert t.shape == h.shape == w.shape == (4 * 3 * 3,)

    def test_video_pos_ids_temporal_values(self):
        t, _, _ = make_video_pos_ids(num_frames=3, patch_h=1, patch_w=1, fps=10.0)
        assert torch.allclose(t, torch.tensor([0.0, 0.1, 0.2]))

    def test_video_pos_ids_t_offset(self):
        t, _, _ = make_video_pos_ids(num_frames=2, patch_h=1, patch_w=1, fps=10.0, t_offset=5)
        assert torch.allclose(t, torch.tensor([0.5, 0.6]))

    def test_video_pos_ids_spatial_grid(self):
        _, h, w = make_video_pos_ids(num_frames=1, patch_h=2, patch_w=3, fps=1.0)
        assert h.tolist() == [0, 0, 0, 1, 1, 1]
        assert w.tolist() == [0, 1, 2, 0, 1, 2]

    def test_action_pos_ids_shape(self):
        t, h, w = make_action_pos_ids(action_horizon=5, fps=10.0)
        assert t.shape == h.shape == w.shape == (5,)

    def test_action_pos_ids_spatial_is_zero(self):
        _, h, w = make_action_pos_ids(action_horizon=5, fps=10.0)
        assert (h == 0).all()
        assert (w == 0).all()

    def test_action_pos_ids_temporal_values(self):
        t, _, _ = make_action_pos_ids(action_horizon=3, fps=10.0, t_offset=2)
        assert torch.allclose(t, torch.tensor([0.2, 0.3, 0.4]))

    def test_concat_pos_ids(self):
        vid = make_video_pos_ids(2, 1, 1, fps=1.0)
        act = make_action_pos_ids(3, fps=1.0, t_offset=2)
        t, h, w = concat_pos_ids(vid, act)
        assert t.shape == (5,)
        assert h.shape == (5,)

    def test_concat_preserves_order(self):
        a = (torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0]))
        b = (torch.tensor([4.0]), torch.tensor([5.0]), torch.tensor([6.0]))
        t, h, w = concat_pos_ids(a, b)
        assert t.tolist() == [1.0, 4.0]
        assert h.tolist() == [2.0, 5.0]
        assert w.tolist() == [3.0, 6.0]


class TestPatchPositionIds:
    def test_pos_shape(self):
        pp = PatchPositionIds(num_frames=4, patch_h=3, patch_w=3, fps=10.0)
        t, h, w = pp.pos
        assert t.shape == (4 * 3 * 3,)

    def test_set_fps_changes_temporal(self):
        pp = PatchPositionIds(num_frames=2, patch_h=1, patch_w=1, fps=10.0)
        old_t = pp.t_ids.clone()
        pp.set_fps(20.0)
        assert not torch.allclose(old_t, pp.t_ids)

    def test_set_patch_grid_changes_spatial(self):
        pp = PatchPositionIds(num_frames=1, patch_h=2, patch_w=2, fps=1.0)
        assert pp.t_ids.shape == (4,)
        pp.set_patch_grid(3, 3)
        assert pp.t_ids.shape == (9,)


# ── RoPE3D ────────────────────────────────────────────────────────────────────

class TestRoPE3D:
    @pytest.fixture
    def rope(self):
        return RoPE3D(HEAD_DIM)

    def test_output_shape(self, rope):
        x = torch.randn(B, HEADS, 10, HEAD_DIM)
        t = torch.arange(10).float()
        h = torch.zeros(10)
        w = torch.zeros(10)
        out = rope(x, t, h, w)
        assert out.shape == x.shape

    def test_zero_positions_is_identity(self, rope):
        x = torch.randn(B, HEADS, 5, HEAD_DIM)
        zeros = torch.zeros(5)
        out = rope(x, zeros, zeros, zeros)
        assert torch.allclose(out, x, atol=1e-6)

    def test_different_positions_give_different_output(self, rope):
        x = torch.randn(B, HEADS, 5, HEAD_DIM)
        zeros = torch.zeros(5)
        ones = torch.ones(5)
        out_a = rope(x, zeros, zeros, zeros)
        out_b = rope(x, ones, zeros, zeros)
        assert not torch.allclose(out_a, out_b)

    def test_relative_position_invariance(self, rope):
        """RoPE encodes relative positions: shifting all positions by the same amount
        should not change pairwise dot products."""
        x = torch.randn(1, HEADS, 4, HEAD_DIM)
        t = torch.tensor([0.0, 1.0, 2.0, 3.0])
        t_shifted = t + 100.0
        h = w = torch.zeros(4)

        out_a = rope(x, t, h, w)
        out_b = rope(x, t_shifted, h, w)
        dots_a = torch.einsum('bhid,bhjd->bhij', out_a, out_a)
        dots_b = torch.einsum('bhid,bhjd->bhij', out_b, out_b)
        assert torch.allclose(dots_a, dots_b, atol=1e-5)


# ── SelfAttention ─────────────────────────────────────────────────────────────

class TestSelfAttention:
    @pytest.fixture
    def sa(self):
        return SelfAttention(HIDDEN, HEADS)

    def test_output_shape(self, sa):
        x = torch.randn(B, 10, HIDDEN)
        out = sa(x)
        assert out.shape == x.shape

    def test_with_rope(self, sa):
        rope = RoPE3D(HEAD_DIM)
        x = torch.randn(B, 6, HIDDEN)
        pos = make_video_pos_ids(3, 1, 2, fps=1.0)
        out = sa(x, rope=rope, pos=pos)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_with_attn_mask(self, sa):
        N = 8
        x = torch.randn(B, N, HIDDEN)
        mask = torch.tril(torch.ones(N, N, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        out = sa(x, attn_mask=mask)
        assert out.shape == x.shape

    def test_rope_and_mask_together(self, sa):
        rope = RoPE3D(HEAD_DIM)
        N = 6
        x = torch.randn(B, N, HIDDEN)
        pos = make_video_pos_ids(3, 1, 2, fps=1.0)
        mask = torch.ones(1, 1, N, N, dtype=torch.bool)
        out = sa(x, rope=rope, pos=pos, attn_mask=mask)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()


# ── CrossAttention ────────────────────────────────────────────────────────────

class TestCrossAttention:
    @pytest.fixture
    def ca(self):
        return CrossAttention(HIDDEN, HEADS, num_sources=2)

    def test_output_shape(self, ca):
        q = torch.randn(B, 10, HIDDEN)
        sources = [torch.randn(B, 5, HIDDEN), torch.randn(B, 3, HIDDEN)]
        out = ca(q, sources)
        assert out.shape == q.shape

    def test_with_rope_on_q(self, ca):
        rope = RoPE3D(HEAD_DIM)
        q = torch.randn(B, 6, HIDDEN)
        sources = [torch.randn(B, 4, HIDDEN), torch.randn(B, 2, HIDDEN)]
        pos = make_video_pos_ids(3, 1, 2, fps=1.0)
        out = ca(q, sources, rope=rope, q_pos=pos)
        assert out.shape == q.shape
        assert not torch.isnan(out).any()

    def test_none_source_skipped(self, ca):
        q = torch.randn(B, 10, HIDDEN)
        sources = [torch.randn(B, 5, HIDDEN), None]
        out = ca(q, sources)
        assert out.shape == q.shape

    def test_source_mask(self, ca):
        q = torch.randn(B, 10, HIDDEN)
        sources = [torch.randn(B, 5, HIDDEN), torch.randn(B, 3, HIDDEN)]
        masks = [torch.zeros(B, 5, dtype=torch.bool), torch.ones(B, 3, dtype=torch.bool)]
        out = ca(q, sources, source_masks=masks)
        assert out.shape == q.shape


# ── Block ─────────────────────────────────────────────────────────────────────

class TestBlock:
    def _make_pos(self, n_vid_frames, patch_h, patch_w, n_actions):
        vid = make_video_pos_ids(n_vid_frames, patch_h, patch_w, fps=10.0)
        act = make_action_pos_ids(n_actions, fps=10.0, t_offset=n_vid_frames)
        return concat_pos_ids(vid, act)

    def test_sa_only_block(self):
        block = Block(HIDDEN, HEADS, num_sources=0, use_adaln=False)
        x = torch.randn(B, 8, HIDDEN)
        pos = make_video_pos_ids(4, 1, 2, fps=1.0)
        out = block(x, pos=pos)
        assert out.shape == x.shape

    def test_full_block_with_adaln(self):
        block = Block(HIDDEN, HEADS, num_sources=2, use_adaln=True)
        N = 16
        pos = self._make_pos(3, 2, 2, 4)
        x = torch.randn(B, N, HIDDEN)
        cond = torch.randn(B, HIDDEN)
        sources = [torch.randn(B, 5, HIDDEN), torch.randn(B, 3, HIDDEN)]
        mask = torch.ones(1, 1, N, N, dtype=torch.bool)
        out = block(x, sources=sources, cond=cond, attn_mask=mask, pos=pos)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_no_nan_no_inf(self):
        block = Block(HIDDEN, HEADS, num_sources=1, use_adaln=True)
        N = 10
        x = torch.randn(B, N, HIDDEN)
        cond = torch.randn(B, HIDDEN)
        sources = [torch.randn(B, 4, HIDDEN)]
        pos = make_video_pos_ids(5, 1, 2, fps=1.0)
        out = block(x, sources=sources, cond=cond, pos=pos)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_adaln_false_no_cond(self):
        block = Block(HIDDEN, HEADS, num_sources=0, use_adaln=False)
        x = torch.randn(B, 6, HIDDEN)
        pos = make_video_pos_ids(3, 1, 2, fps=1.0)
        out = block(x, pos=pos)
        assert out.shape == x.shape

    def test_block_causal_mask(self):
        """Block should run without error when given a lower-triangular causal mask."""
        block = Block(HIDDEN, HEADS, num_sources=1, use_adaln=True)
        N = 8
        x = torch.randn(B, N, HIDDEN)
        cond = torch.randn(B, HIDDEN)
        sources = [torch.randn(B, 4, HIDDEN)]
        pos = make_video_pos_ids(4, 1, 2, fps=1.0)
        causal = torch.tril(torch.ones(N, N, dtype=torch.bool)).unsqueeze(0).unsqueeze(0)
        out = block(x, sources=sources, cond=cond, attn_mask=causal, pos=pos)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_gradient_flows(self):
        block = Block(HIDDEN, HEADS, num_sources=1, use_adaln=True)
        N = 8
        x = torch.randn(B, N, HIDDEN, requires_grad=True)
        cond = torch.randn(B, HIDDEN)
        sources = [torch.randn(B, 4, HIDDEN)]
        pos = make_video_pos_ids(4, 1, 2, fps=1.0)
        out = block(x, sources=sources, cond=cond, pos=pos)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_overfit_single_sample(self):
        """Block should be able to memorize a fixed input-output pair."""
        torch.manual_seed(42)
        block = Block(HIDDEN, HEADS, num_sources=1, use_adaln=True)
        block.train()

        N = 8
        x = torch.randn(1, N, HIDDEN)
        cond = torch.randn(1, HIDDEN)
        sources = [torch.randn(1, 4, HIDDEN)]
        pos = make_video_pos_ids(4, 1, 2, fps=1.0)
        target = torch.randn(1, N, HIDDEN)

        opt = torch.optim.Adam(block.parameters(), lr=1e-3)
        for _ in range(200):
            out = block(x, sources=sources, cond=cond, pos=pos)
            loss = torch.nn.functional.mse_loss(out, target)
            opt.zero_grad()
            loss.backward()
            opt.step()

        assert loss.item() < 0.05, f"Failed to overfit (loss={loss.item():.4f})"


# ── Utilities ─────────────────────────────────────────────────────────────────

class TestUtilities:
    def test_modulate(self):
        x = torch.ones(2, 4, 8)
        shift = torch.zeros(2, 8)
        scale = torch.ones(2, 8)
        out = modulate(x, shift, scale)
        assert torch.allclose(out, 2 * x)

    def test_make_mlp_shape(self):
        mlp = make_mlp(32, 64, 16)
        x = torch.randn(2, 10, 32)
        assert mlp(x).shape == (2, 10, 16)

    def test_swiglu_not_linear(self):
        sg = SwiGLULinear(8, 8)
        x = torch.randn(2, 8)
        out = sg(x)
        assert out.shape == x.shape
        assert not torch.allclose(out, x)
