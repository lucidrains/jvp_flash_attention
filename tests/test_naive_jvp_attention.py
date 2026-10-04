"""Tests for the pure-PyTorch tiled attention fallback (`naive_pytorch`)."""

from __future__ import annotations

import math
from functools import partial

import pytest
import torch
import torch.autograd.forward_ad as fwAD
from torch import Tensor, enable_grad
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention

from jvp_flash_attention.jvp_attention import (
    HAS_TRITON,
    MASK_CONST,
    JVPAttn,
    use_naive_attention,
)
from jvp_flash_attention.naive_pytorch import (
    naive_attention,
    prepare_attn_mask,
    tiled_attention,
    tiled_attention_backward,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPES = [torch.float32, torch.float16, torch.bfloat16]
ATOL = {torch.float32: 1e-4, torch.float16: 4e-3, torch.bfloat16: 4e-3}
RTOL = 1e-3
Z, H, N, D = 2, 3, 64, 32
MASK_CASES = [
    pytest.param(None, False, id="none"),
    pytest.param("boolean", False, id="boolean"),
    pytest.param("additive", False, id="additive"),
    pytest.param(None, True, id="causal"),
]


def rand(*shape: int, dtype: torch.dtype = torch.float32, seed: int = 0) -> Tensor:
    """Create a reproducible random tensor on the test device."""

    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(*shape, dtype=dtype, generator=generator).to(DEVICE)


def make_qkv_tangents_target(dtype: torch.dtype, n_ctx: int = N) -> tuple[Tensor, ...]:
    """Create reproducible (q, k, v, q_t, k_t, v_t, target) tensors."""

    return tuple(rand(Z, H, n_ctx, D, dtype=dtype, seed=i) for i in range(7))


def make_duals(*tensors: Tensor) -> tuple[Tensor, ...]:
    """Make dual tensors that retain input gradients, as `torch.autograd` expects."""

    duals = []
    for primal, tangent in zip(tensors[::2], tensors[1::2]):
        dual = fwAD.make_dual(primal, tangent)
        dual.requires_grad = True
        dual.retain_grad()
        duals.append(dual)
    return tuple(duals)


def make_attn_mask(mask_kind: str | None, dtype: torch.dtype, n_ctx: int = N) -> Tensor | None:
    """Create a mask that passes JVP attention's mask validation."""

    if mask_kind is None:
        return None

    generator = torch.Generator(device="cpu").manual_seed(42)
    if mask_kind == "boolean":
        mask = torch.rand(Z, H, n_ctx, n_ctx, generator=generator) > 0.3
        mask[..., 0, 0] = True  # Ensure no head is fully masked
        return mask.to(DEVICE)

    mask = torch.where(
        torch.rand(Z, H, n_ctx, n_ctx, generator=generator) > 0.3,
        torch.zeros(()),
        torch.full((), MASK_CONST),
    )
    mask[..., 0, 0] = 0.0  # Ensure no head is fully masked
    return mask.to(dtype).to(DEVICE)


def sdpa_dual_reference(
    q_p: Tensor,
    k_p: Tensor,
    v_p: Tensor,
    q_t: Tensor,
    k_t: Tensor,
    v_t: Tensor,
    target: Tensor,
    mask: Tensor | None,
    causal: bool,
) -> tuple[Tensor, ...]:
    """Compute SDPA's primal, tangent, and input gradients in float32 MATH mode."""

    with sdpa_kernel(SDPBackend.MATH), fwAD.dual_level(), enable_grad():
        q, k, v = make_duals(
            q_p.clone(), q_t.clone(), k_p.clone(), k_t.clone(), v_p.clone(), v_t.clone()
        )
        out = scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal)
        o_p, o_t = fwAD.unpack_dual(out)
        ((o_p - target) ** 2).mean().backward()

    return o_p.detach(), o_t.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()


def assert_close_metrics(
    got: tuple[Tensor, ...], expected: tuple[Tensor, ...], dtype: torch.dtype
) -> None:
    """Compare primal/tangent/gradient tensors with dtype-appropriate tolerances."""

    for actual, desired in zip(got, expected):
        torch.testing.assert_close(actual, desired, atol=ATOL[dtype], rtol=RTOL)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mask_kind,causal", MASK_CASES)
def test_fwd_matches_sdpa(dtype: torch.dtype, mask_kind: str | None, causal: bool) -> None:
    """The naive forward primal and gradients match SDPA's."""

    q_p, k_p, v_p, *_, target = make_qkv_tangents_target(dtype)
    mask = make_attn_mask(mask_kind, dtype)

    with sdpa_kernel(SDPBackend.MATH), enable_grad():
        q_ref, k_ref, v_ref = (t.clone().requires_grad_() for t in (q_p, k_p, v_p))
        ref = scaled_dot_product_attention(q_ref, k_ref, v_ref, attn_mask=mask, is_causal=causal)
        ((ref - target) ** 2).mean().backward()

    q, k, v = (t.clone().requires_grad_() for t in (q_p, k_p, v_p))
    out = JVPAttn.fwd(q, k, v, attn_mask=mask, causal=causal)
    ((out - target) ** 2).mean().backward()

    assert_close_metrics(
        (out, q.grad, k.grad, v.grad),
        (ref, q_ref.grad, k_ref.grad, v_ref.grad),
        dtype,
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mask_kind,causal", MASK_CASES)
def test_fwd_dual_matches_sdpa(dtype: torch.dtype, mask_kind: str | None, causal: bool) -> None:
    """The naive `fwd_dual` primal, tangent, and gradients match SDPA's."""

    q_p, k_p, v_p, q_t, k_t, v_t, target = make_qkv_tangents_target(dtype)
    mask = make_attn_mask(mask_kind, dtype)
    expected = sdpa_dual_reference(q_p, k_p, v_p, q_t, k_t, v_t, target, mask, causal)

    with fwAD.dual_level(), enable_grad():
        q, k, v = make_duals(
            q_p.clone(), q_t.clone(), k_p.clone(), k_t.clone(), v_p.clone(), v_t.clone()
        )
        out = JVPAttn.fwd_dual(q, k, v, attn_mask=mask, causal=causal)
        o_p, o_t = fwAD.unpack_dual(out)
        ((o_p - target) ** 2).mean().backward()

    assert_close_metrics((o_p, o_t, q.grad, k.grad, v.grad), expected, dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mask_kind,causal", MASK_CASES)
def test_torch_func_jvp_matches_sdpa(dtype: torch.dtype, mask_kind: str | None, causal: bool) -> None:
    """The naive forward under `torch.func.jvp` matches SDPA's primal, tangent, and gradients."""

    q_p, k_p, v_p, q_t, k_t, v_t, target = make_qkv_tangents_target(dtype)
    mask = make_attn_mask(mask_kind, dtype)
    expected = sdpa_dual_reference(q_p, k_p, v_p, q_t, k_t, v_t, target, mask, causal)

    primals = tuple(t.clone().requires_grad_() for t in (q_p, k_p, v_p))
    with enable_grad():
        o_p, o_t = torch.func.jvp(
            partial(JVPAttn.fwd_dual, attn_mask=mask, causal=causal),
            primals,
            (q_t, k_t, v_t),
        )
        ((o_p - target) ** 2).mean().backward()

    assert_close_metrics((o_p, o_t, *[t.grad for t in primals]), expected, dtype)


def test_fwd_dual_with_plain_inputs() -> None:
    """`fwd_dual` accepts plain tensors and returns a plain tensor."""

    q_p, k_p, v_p, *_ = make_qkv_tangents_target(torch.float32)
    out = JVPAttn.fwd_dual(q_p, k_p, v_p)
    with fwAD.dual_level():
        assert fwAD.unpack_dual(out)[1] is None
    torch.testing.assert_close(out, JVPAttn.fwd(q_p, k_p, v_p), atol=1e-6, rtol=1e-6)


def test_apply_interface() -> None:
    """`JVPAttn.apply` returns the kernel's (output, context) structure."""

    q_p, k_p, v_p, q_t, k_t, v_t, _ = make_qkv_tangents_target(torch.float32)
    expected = JVPAttn.fwd(q_p, k_p, v_p)
    args = (None, 0.0, False, None, True, True, True, True)

    out, ctx = JVPAttn.apply(q_p, k_p, v_p, None, None, None, *args)
    assert ctx[0] is None
    assert ctx[1].shape == (Z, H, N)  # Base-2 log-sum-exp
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)

    out, ctx = JVPAttn.apply(q_p, k_p, v_p, q_t, k_t, v_t, *args)
    _, expected_tangent = torch.func.jvp(
        JVPAttn.fwd_dual, (q_p, k_p, v_p), (q_t, k_t, v_t)
    )
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(ctx[0], expected_tangent, atol=1e-6, rtol=1e-6)


def test_lse_matches_logsumexp() -> None:
    """The returned base-2 log-sum-exp matches `logsumexp` of the natural scores."""

    q_p, k_p, v_p, *_ = make_qkv_tangents_target(torch.float32)
    sm_scale = D**-0.5
    _, _, lse = tiled_attention(q_p, k_p, v_p, torch.empty(0), 0, False, sm_scale)
    natural_scores = (q_p @ k_p.transpose(-1, -2)) * sm_scale
    expected = torch.logsumexp(natural_scores, dim=-1) / math.log(2)
    torch.testing.assert_close(lse, expected, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("n_ctx", [32, 96, 130, 258])
def test_bucket_size_invariance(n_ctx: int) -> None:
    """Bucket sizes only affect memory usage, not the result."""

    q_p, k_p, v_p, *_ = make_qkv_tangents_target(torch.float32, n_ctx)
    empty = torch.empty(0, device=DEVICE)
    small = tiled_attention(
        q_p, k_p, v_p, empty, 0, False, D**-0.5, q_bucket_size=32, k_bucket_size=32
    )
    large = tiled_attention(
        q_p, k_p, v_p, empty, 0, False, D**-0.5, q_bucket_size=512, k_bucket_size=512
    )
    torch.testing.assert_close(small[0], large[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(small[2], large[2], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize(
    "n_ctx,q_bucket_size,k_bucket_size",
    [
        pytest.param(32, 128, 128, id="shorter-than-one-tile"),
        pytest.param(96, 64, 64, id="partial-trailing-tiles"),
        pytest.param(130, 64, 32, id="uneven-partial-tiles"),
        pytest.param(258, 128, 128, id="partial-tile-after-full-tiles"),
    ],
)
@pytest.mark.parametrize("mask_kind,causal", MASK_CASES)
def test_arbitrary_seq_lens(
    n_ctx: int, q_bucket_size: int, k_bucket_size: int, mask_kind: str | None, causal: bool
) -> None:
    """Any even length >= 32 works, whether or not the tile size divides it."""

    dtype = torch.float32
    q_p, k_p, v_p, q_t, k_t, v_t, target = make_qkv_tangents_target(dtype, n_ctx)
    mask = make_attn_mask(mask_kind, dtype, n_ctx)
    expected = sdpa_dual_reference(q_p, k_p, v_p, q_t, k_t, v_t, target, mask, causal)

    # default buckets, through the autograd.Function
    with fwAD.dual_level(), enable_grad():
        q, k, v = make_duals(
            q_p.clone(), q_t.clone(), k_p.clone(), k_t.clone(), v_p.clone(), v_t.clone()
        )
        out = JVPAttn.fwd_dual(q, k, v, attn_mask=mask, causal=causal)
        o_p, o_t = fwAD.unpack_dual(out)
        ((o_p - target) ** 2).mean().backward()
    assert_close_metrics((o_p, o_t, q.grad, k.grad, v.grad), expected, dtype)

    # explicit (possibly non-dividing) buckets, through the pure-PyTorch pass
    mask_tensor, MASK_TYPE = prepare_attn_mask(mask, q_p)
    o, o_t, lse = tiled_attention(
        q_p, k_p, v_p, mask_tensor, MASK_TYPE, causal, D**-0.5, q_t, k_t, v_t,
        q_bucket_size=q_bucket_size, k_bucket_size=k_bucket_size,
    )
    torch.testing.assert_close(o, expected[0], atol=ATOL[dtype], rtol=RTOL)
    torch.testing.assert_close(o_t, expected[1], atol=ATOL[dtype], rtol=RTOL)

    do = 2 * (o - target) / o.numel()
    dq, dk, dv = tiled_attention_backward(
        q_p, k_p, v_p, o, do, lse, mask_tensor, MASK_TYPE, causal, D**-0.5,
        q_bucket_size=q_bucket_size, k_bucket_size=k_bucket_size,
    )
    assert_close_metrics((dq, dk, dv), expected[2:5], dtype)


def test_cpu_dispatch() -> None:
    """CPU (and Triton-less) execution dispatches to the naive implementation, with overrides."""

    assert use_naive_attention(torch.device("cpu"))
    assert use_naive_attention(torch.device("cpu"), USE_NAIVE=True)
    if not (HAS_TRITON and torch.cuda.is_available()):
        assert use_naive_attention(torch.device("cuda"))
        assert use_naive_attention(torch.device("cuda"), USE_NAIVE=True)

    with pytest.raises(AssertionError, match="Cannot force the Triton backend"):
        use_naive_attention(torch.device("cpu"), USE_NAIVE=False)
    with pytest.raises(AssertionError, match="USE_NAIVE must be"):
        use_naive_attention(torch.device("cpu"), USE_NAIVE="yes")

    if DEVICE.type == "cpu":
        q_p, k_p, v_p, *_ = make_qkv_tangents_target(torch.float32)
        torch.testing.assert_close(
            JVPAttn.fwd(q_p, k_p, v_p, USE_NAIVE=True),
            naive_attention(q_p, k_p, v_p),
            atol=1e-6,
            rtol=1e-6,
        )
        with pytest.raises(AssertionError, match="Cannot force the Triton backend"):
            JVPAttn.fwd(q_p, k_p, v_p, USE_NAIVE=False)

        q_t, k_t, v_t = (rand(Z, H, N, D, seed=i + 3) for i in range(3))
        with fwAD.dual_level():
            out = JVPAttn.fwd_dual(q_p, k_p, v_p, USE_NAIVE=True)
            out_t = JVPAttn.fwd_dual(
                fwAD.make_dual(q_p, q_t),
                fwAD.make_dual(k_p, k_t),
                fwAD.make_dual(v_p, v_t),
                USE_NAIVE=True,
            )
            o_p, _ = fwAD.unpack_dual(out_t)
        torch.testing.assert_close(out, o_p, atol=1e-6, rtol=1e-6)


def test_causal_mask_rejected() -> None:
    """Causal attention rejects an attention mask."""

    q_p, k_p, v_p, *_ = make_qkv_tangents_target(torch.float32)
    with pytest.raises(ValueError, match="Causal attention does not support an attention mask"):
        JVPAttn.fwd(q_p, k_p, v_p, attn_mask=torch.ones(Z, H, N, N, dtype=torch.bool), causal=True)


def test_dropout_rejected() -> None:
    """Dropout remains unsupported."""

    q_p, k_p, v_p, *_ = make_qkv_tangents_target(torch.float32)
    with pytest.raises(NotImplementedError, match="Dropout is not currently supported"):
        JVPAttn.fwd(q_p, k_p, v_p, dropout_p=0.1)


def test_all_false_head_rejected() -> None:
    """A boolean mask must leave at least one position per head."""

    q_p, k_p, v_p, *_ = make_qkv_tangents_target(torch.float32)
    mask = make_attn_mask("boolean", torch.float32)
    mask[0, 0] = False
    with pytest.raises(AssertionError, match="cannot be all False"):
        JVPAttn.fwd(q_p, k_p, v_p, attn_mask=mask)


def test_invalid_additive_mask_rejected() -> None:
    """Additive masks cannot contain NaN/inf or be all-masked for any head."""

    q_p, k_p, v_p, *_ = make_qkv_tangents_target(torch.float32)

    nan_mask = make_attn_mask("additive", torch.float32)
    nan_mask[0, 0, 0, 0] = float("nan")
    with pytest.raises(AssertionError, match="cannot contain NaNs"):
        JVPAttn.fwd(q_p, k_p, v_p, attn_mask=nan_mask)

    inf_mask = make_attn_mask("additive", torch.float32)
    inf_mask[0, 0, 0, 0] = float("inf")
    with pytest.raises(AssertionError, match="cannot contain -inf or inf"):
        JVPAttn.fwd(q_p, k_p, v_p, attn_mask=inf_mask)

    all_masked = make_attn_mask("additive", torch.float32)
    all_masked[0, 0] = MASK_CONST
    with pytest.raises(AssertionError, match="cannot be all"):
        JVPAttn.fwd(q_p, k_p, v_p, attn_mask=all_masked)


def test_additive_mask_warning() -> None:
    """An additive mask without the masking constant warns."""

    q_p, k_p, v_p, *_ = make_qkv_tangents_target(torch.float32)
    mask = torch.zeros(Z, H, N, N, dtype=torch.float32, device=DEVICE)
    with pytest.raises(UserWarning, match="does not mask out any elements"):
        JVPAttn.fwd(q_p, k_p, v_p, attn_mask=mask)

    # NOTE: The warning can be skipped entirely by disabling mask verification
    out = JVPAttn.fwd(q_p, k_p, v_p, attn_mask=mask, verify_attn_mask=False)
    assert out.shape == (Z, H, N, D)


@pytest.mark.skipif(
    not (HAS_TRITON and torch.cuda.is_available()), reason="requires Triton on a CUDA device"
)
@pytest.mark.parametrize("mask_kind,causal", MASK_CASES)
def test_naive_matches_triton_kernel(mask_kind: str | None, causal: bool) -> None:
    """The naive implementation matches the Triton kernel's outputs and gradients."""

    dtype = torch.float32
    q_p, k_p, v_p, q_t, k_t, v_t, target = make_qkv_tangents_target(dtype)
    mask = make_attn_mask(mask_kind, dtype)

    with fwAD.dual_level(), enable_grad():
        q, k, v = make_duals(
            q_p.clone(), q_t.clone(), k_p.clone(), k_t.clone(), v_p.clone(), v_t.clone()
        )
        out = JVPAttn.fwd_dual(q, k, v, attn_mask=mask, causal=causal)
        o_p, o_t = fwAD.unpack_dual(out)
        ((o_p - target) ** 2).mean().backward()
        triton = (o_p.detach(), o_t.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach())

    with fwAD.dual_level(), enable_grad():
        q, k, v = make_duals(
            q_p.clone(), q_t.clone(), k_p.clone(), k_t.clone(), v_p.clone(), v_t.clone()
        )
        out = naive_attention(q, k, v, attn_mask=mask, causal=causal)
        o_p, o_t = fwAD.unpack_dual(out)
        ((o_p - target) ** 2).mean().backward()
        naive = (o_p.detach(), o_t.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach())

    assert_close_metrics(naive, triton, dtype)
