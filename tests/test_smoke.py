"\"\"\"
Smoke tests — no GUI-Actor checkpoint required.

These should pass with `pytest tests/` on a CPU-only machine.
They use the StubBackbone so you can verify the head + loss + metric
plumbing before launching real training on the GPU box.
\"\"\"

import torch
import numpy as np
import pytest

from src.models.action_head_with_null import (
    VisionHeadWithNull, NullPatchAdapter,
)
from src.training.loss import ScreenAbstainLoss, ScreenAbstainLossConfig
from src.eval.metrics import compute_refusal_report


# ---------------------------------------------------------------------- #
def test_head_output_shapes():
    head = VisionHeadWithNull(hidden_dim=64, patch_dim=64)
    B, N = 4, 36 * 36
    q = torch.randn(B, 64)
    p = torch.randn(B, N, 64)
    out = head(q, p)
    assert out[\"attn_logits\"].shape == (B, N + 1)
    assert out[\"attn_probs\"].shape == (B, N + 1)
    assert out[\"p_refuse\"].shape == (B,)
    assert out[\"click_logits\"].shape == (B, N)


def test_head_num_new_params():
    head = VisionHeadWithNull(hidden_dim=64, patch_dim=64, use_proj=False)
    # null_patch (64) + null_bias (1)
    assert head.num_new_params == 65


def test_loss_runs_on_mixed_batch():
    head = VisionHeadWithNull(hidden_dim=32, patch_dim=32)
    B, N = 6, 49
    q = torch.randn(B, 32, requires_grad=True)
    p = torch.randn(B, N, 32, requires_grad=True)
    out = head(q, p)

    target = torch.tensor([3, 17, -1, 5, -1, 22], dtype=torch.long)
    is_ref = torch.tensor([0, 0, 1, 0, 1, 0], dtype=torch.bool)

    loss_fn = ScreenAbstainLoss(ScreenAbstainLossConfig(lambda_margin=0.5))
    out_loss = loss_fn(out[\"attn_logits\"], target, is_ref)
    out_loss[\"loss\"].backward()
    assert torch.isfinite(out_loss[\"loss\"])


def test_refusal_loss_targets_null():
    \"\"\"Pure-refusal batch: ground loss should be NaN-free, refuse loss
    should drive p_refuse upward.\"\"\"
    torch.manual_seed(0)
    head = VisionHeadWithNull(hidden_dim=32, patch_dim=32)
    opt = torch.optim.SGD(head.parameters(), lr=0.5)
    loss_fn = ScreenAbstainLoss(ScreenAbstainLossConfig())

    q = torch.randn(8, 32)
    p = torch.randn(8, 25, 32)
    target = torch.full((8,), -1, dtype=torch.long)
    is_ref = torch.ones(8, dtype=torch.bool)

    p_ref_before = head(q, p)[\"p_refuse\"].mean().item()
    for _ in range(20):
        out = head(q, p)
        out_loss = loss_fn(out[\"attn_logits\"], target, is_ref)
        opt.zero_grad(); out_loss[\"loss\"].backward(); opt.step()
    p_ref_after = head(q, p)[\"p_refuse\"].mean().item()
    assert p_ref_after > p_ref_before + 0.1, (p_ref_before, p_ref_after)


def test_metric_aggregation_shapes():
    M = 50
    p_ref = np.random.rand(M)
    argmax = np.random.randint(0, 36 * 36, size=M)
    is_ref = np.random.rand(M) > 0.5
    bboxes = [(10, 10, 50, 50) if not r else None for r in is_ref]
    sizes = [(640, 480)] * M

    rep = compute_refusal_report(
        p_ref, argmax, is_ref, bboxes, sizes,
        patch_grid=(36, 36), threshold=0.5,
    )
    d = rep.to_dict()
    for k in (\"ground_acc\", \"refuse_acc\", \"auroc_refuse\"):
        assert k in d


def test_adapter_shape_compatibility():
    \"\"\"NullPatchAdapter wraps an existing head and adds null logit.\"\"\"
    class FakeHead(torch.nn.Module):
        def forward(self, q, p, mask=None, **kwargs):
            return torch.randn(q.shape[0], p.shape[1])

    adapter = NullPatchAdapter(FakeHead(), patch_dim=32, hidden_dim=32)
    q = torch.randn(3, 32); p = torch.randn(3, 49, 32)
    out = adapter(q, p)
    assert out[\"attn_logits\"].shape == (3, 50)
    assert out[\"p_refuse\"].shape == (3,)


if __name__ == \"__main__\":
    pytest.main([__file__, \"-v\"])
"
