# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pytest
import torch

from ultralytics.nn.modules import C2PSA, C2PSAIRPEK
from ultralytics.nn.modules.block import _AttentionIRPEK, _ContextualProductIRPE


def test_irpe_k_constructor_contract_and_zero_initialization() -> None:
    """Each PSA block should own one shared-head 9x9 contextual table initialized to zero."""
    module = C2PSAIRPEK(256, 256, n=2)

    assert module.c == 128
    assert module.cv1.conv.in_channels == 256
    assert module.cv1.conv.out_channels == 256
    assert module.cv2.conv.in_channels == 256
    assert module.cv2.conv.out_channels == 256
    assert len(module.m) == 2
    for block in module.m:
        assert block.attn.num_heads == 2
        assert block.attn.head_dim == 64
        assert block.attn.key_dim == 32
        assert block.attn.rpe_k.lookup_table_weight.shape == (1, 32, 81)
        torch.testing.assert_close(
            block.attn.rpe_k.lookup_table_weight,
            torch.zeros_like(block.attn.rpe_k.lookup_table_weight),
            rtol=0,
            atol=0,
        )


def test_product_bucket_directions_and_center() -> None:
    """Product IDs should retain signed row/column direction and use bucket 40 for zero offset."""
    rpe = _ContextualProductIRPE(head_dim=32)
    index = rpe._build_relative_position_index(3, 4, torch.device("cpu"))

    assert index.shape == (12, 12)
    assert torch.equal(index.diagonal(), torch.full((12,), 40, dtype=torch.long))
    assert index[0, 1].item() == 39  # key one column to the right
    assert index[1, 0].item() == 41  # key one column to the left
    assert index[0, 4].item() == 31  # key one row below
    assert index[4, 0].item() == 49  # key one row above
    assert 0 <= index.min().item() <= index.max().item() < 81


def test_piecewise_mapping_is_symmetric_and_saturates() -> None:
    """Long signed offsets should compress symmetrically and remain inside [-4, 4]."""
    rpe = _ContextualProductIRPE(head_dim=32)
    offsets = torch.tensor([-100, -16, -3, -2, 0, 2, 3, 16, 100])
    expected = torch.tensor([-4, -4, -2, -2, 0, 2, 2, 4, 4])

    torch.testing.assert_close(rpe._piecewise_index(offsets), expected, rtol=0, atol=0)


def test_rectangular_cache_reuse_invalidation_and_state_dict_exclusion() -> None:
    """The non-persistent index cache should follow H/W while remaining outside checkpoints."""
    rpe = _ContextualProductIRPE(head_dim=32)
    query = torch.randn(1, 2, 12, 32)

    first = rpe(query, 3, 4)
    first_pointer = rpe._relative_position_index.data_ptr()
    second = rpe(query, 3, 4)
    assert first.shape == second.shape == (1, 2, 12, 12)
    assert rpe._relative_position_index.data_ptr() == first_pointer

    rectangular = rpe(torch.randn(1, 2, 15, 32), 3, 5)
    assert rectangular.shape == (1, 2, 15, 15)
    assert rpe._cached_hw == (3, 5)
    assert rpe._relative_position_index.shape == (15, 15)
    assert set(rpe.state_dict()) == {"lookup_table_weight"}


def test_zero_initialized_irpe_matches_transferred_baseline_exactly() -> None:
    """With all baseline weights transferred, zero iRPE should preserve the baseline output exactly."""
    torch.manual_seed(7)
    baseline = C2PSA(256, 256, n=1).eval()
    variant = C2PSAIRPEK(256, 256, n=1).eval()
    transfer = variant.load_state_dict(baseline.state_dict(), strict=False)

    assert transfer.missing_keys == ["m.0.attn.rpe_k.lookup_table_weight"]
    assert transfer.unexpected_keys == []
    x = torch.randn(2, 256, 5, 7)
    with torch.no_grad():
        expected = baseline(x)
        actual = variant(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_nonzero_irpe_changes_the_attention_result() -> None:
    """A learned non-zero contextual table should activate the new causal path."""
    torch.manual_seed(11)
    baseline = C2PSA(256, 256, n=1).eval()
    variant = C2PSAIRPEK(256, 256, n=1).eval()
    variant.load_state_dict(baseline.state_dict(), strict=False)
    x = torch.randn(1, 256, 4, 5)

    with torch.no_grad():
        zero_output = variant(x)
        variant.m[0].attn.rpe_k.lookup_table_weight.normal_(mean=0.0, std=0.1)
        active_output = variant(x)

    assert not torch.allclose(active_output, zero_output)


def test_attention_forward_matches_pre_softmax_irpe_equation() -> None:
    """The implementation should add contextual Product iRPE-K to scaled logits before softmax."""
    torch.manual_seed(13)
    module = _AttentionIRPEK(128, num_heads=2, attn_ratio=0.5).eval()
    with torch.no_grad():
        module.rpe_k.lookup_table_weight.normal_(mean=0.0, std=0.05)
    x = torch.randn(1, 128, 3, 4)

    with torch.no_grad():
        B, C, H, W = x.shape
        N = H * W
        qkv = module.qkv(x)
        q, k, v = qkv.view(B, module.num_heads, module.key_dim * 2 + module.head_dim, N).split(
            [module.key_dim, module.key_dim, module.head_dim], dim=2
        )
        query = q.transpose(-2, -1)
        logits = (query @ k) * module.scale
        logits = logits + module.rpe_k(query, H, W) * module.scale
        weights = logits.softmax(dim=-1)
        expected = (v @ weights.transpose(-2, -1)).view(B, C, H, W) + module.pe(v.reshape(B, C, H, W))
        expected = module.proj(expected)
        actual = module(x)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_irpe_k_forward_backward_is_finite_and_table_learns() -> None:
    """The zero-initialized table and the input should receive finite, non-zero gradients."""
    torch.manual_seed(17)
    module = C2PSAIRPEK(256, 256, n=1)
    x = torch.randn(2, 256, 5, 7, requires_grad=True)
    output = module(x)
    output.square().mean().backward()

    table_gradient = module.m[0].attn.rpe_k.lookup_table_weight.grad
    assert output.shape == x.shape and torch.isfinite(output).all()
    assert x.grad is not None and torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad) > 0
    assert table_gradient is not None and torch.isfinite(table_gradient).all()
    assert torch.count_nonzero(table_gradient) > 0


def test_twenty_by_twenty_p5_attention_contract() -> None:
    """The normal 640-input P5 geometry should preserve shape and finite values."""
    module = _AttentionIRPEK(128, num_heads=2, attn_ratio=0.5).eval()
    x = torch.randn(1, 128, 20, 20)
    with torch.no_grad():
        output = module(x)
    assert output.shape == x.shape
    assert torch.isfinite(output).all()


def test_baseline_c2psa_remains_without_irpe() -> None:
    """The unchanged baseline class must not acquire an iRPE branch."""
    baseline = C2PSA(256, 256, n=1)
    assert not hasattr(baseline.m[0].attn, "rpe_k")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_irpe_k_cuda_fp32_and_fp16_autocast() -> None:
    """The ablation should support finite CUDA FP32 and FP16-autocast forward/backward execution."""
    device = torch.device("cuda:0")
    module = C2PSAIRPEK(256, 256, n=1).to(device)

    fp32_input = torch.randn(1, 256, 4, 5, device=device, requires_grad=True)
    fp32_output = module(fp32_input)
    fp32_output.square().mean().backward()
    assert torch.isfinite(fp32_output).all()
    assert fp32_input.grad is not None and torch.isfinite(fp32_input.grad).all()

    module.zero_grad(set_to_none=True)
    fp16_input = torch.randn(1, 256, 4, 5, device=device, dtype=torch.float16, requires_grad=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        fp16_output = module(fp16_input)
    fp16_output.float().square().mean().backward()
    assert torch.isfinite(fp16_output).all()
    assert fp16_input.grad is not None and torch.isfinite(fp16_input.grad).all()
    assert module.m[0].attn.rpe_k.lookup_table_weight.dtype == torch.float32
