"""Tiled attention for CPU.

Online softmax over q/k buckets. The explicit JVP and backward mirror the Triton kernels, so no
full (N, N) tensor is ever materialized.
"""

from __future__ import annotations

import torch
from torch import Tensor, arange, cat, einsum

RCP_LN2 = 1.4426950408889634  # 1 / ln(2)
MASK_CONST = -1.0e2  # large negative mask value


def exists(val):
    return val is not None


def default_dtype(t):
    return t.dtype if t.dtype in (torch.float32, torch.float64) else torch.float32


def prepare_attn_mask(
    attn_mask: Tensor | None, q: Tensor, verify_attn_mask: bool = True
) -> tuple[Tensor, int]:
    """-> (contiguous mask, MASK_TYPE), where MASK_TYPE 0: none, 1: bool, 2: additive."""
    if not exists(attn_mask):
        return torch.empty(0, device=q.device, dtype=q.dtype), 0
    if verify_attn_mask:
        validate_attn_mask(attn_mask, q)
    is_bool = attn_mask.dtype == torch.bool
    # bool -> type 1 (attend / not attend), float -> type 2 (additive bias)
    return (attn_mask if is_bool else attn_mask.to(q.dtype)).contiguous(), 1 if is_bool else 2


def validate_attn_mask(mask: Tensor, q: Tensor) -> None:
    """Assert that the mask matches the query and no head is fully masked."""
    z, h, n, _ = q.shape
    assert mask.shape == (z, h, n, n), "The attention mask must have shape (Z, H, N_CTX, N_CTX)."
    assert mask.dtype in (
        torch.bool,
        q.dtype,
    ), "The attention mask must be bool or of the query dtype."

    if mask.dtype == torch.bool:
        assert mask.any(dim=(-1, -2)).all(), "The attention mask cannot be all False for any head."
        return

    assert not torch.isinf(mask).any(), "The attention mask cannot contain -inf or inf."
    assert not torch.isnan(mask).any(), "The attention mask cannot contain NaNs."
    assert (
        (mask != MASK_CONST).any(dim=(-1, -2)).all()
    ), f"The attention mask cannot be all {MASK_CONST} for any head."
    if not (mask == MASK_CONST).any():
        raise UserWarning(
            f"The attention mask does not mask out any elements with {MASK_CONST}; "
            "use that constant for correct masking behavior."
        )


def mask_bias(
    mask: Tensor, MASK_TYPE: int, q_slice: slice, k_slice: slice
) -> tuple[Tensor, Tensor | None]:
    """-> (additive bias, keep mask or None) for one score block."""
    if MASK_TYPE == 1:
        keep = mask[..., q_slice, k_slice]
        # masked (False) positions become -inf; the rest are untouched
        return torch.where(keep, 0.0, MASK_CONST), keep
    if MASK_TYPE == 2:
        return mask[..., q_slice, k_slice], None
    return 0.0, None


def scale_scores(
    scores: Tensor,
    qk_scale: float,
    causal: bool,
    q_pos: Tensor,
    k_pos: Tensor,
    q_slice: slice,
    k_slice: slice,
) -> Tensor:
    """Scale scores; causal masking is post-scale, as in the kernels."""
    if not causal:
        return scores * qk_scale
    # causal mask applied post-scale, matching the triton kernels
    keep = q_pos[q_slice][:, None] >= k_pos[k_slice][None, :]
    return scores * qk_scale + torch.where(keep, 0.0, MASK_CONST)


def tiled_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask_tensor: Tensor,
    MASK_TYPE: int,
    causal: bool,
    sm_scale: float,
    q_t: Tensor | None = None,
    k_t: Tensor | None = None,
    v_t: Tensor | None = None,
    q_bucket_size: int = 128,
    k_bucket_size: int = 128,
) -> tuple[Tensor, Tensor | None, Tensor]:
    """Tiled attention -> (o, o_t, lse); o_t is None without tangents."""
    z, h, n_q, d = q.shape
    n_kv = k.shape[-2]
    device, out_dtype = q.device, q.dtype
    compute_dtype = default_dtype(q)
    q, k, v = (t.to(compute_dtype) for t in (q, k, v))

    enable_jvp = exists(q_t) and exists(k_t) and exists(v_t)
    if enable_jvp:
        q_t, k_t, v_t = (t.to(compute_dtype) for t in (q_t, k_t, v_t))

    qk_scale = sm_scale * RCP_LN2  # base-2 softmax, kernel parity
    q_pos = arange(n_kv - n_q, n_kv, device=device)  # queries aligned to the tail of kv
    k_pos = arange(n_kv, device=device)
    q_chunks = q.split(q_bucket_size, dim=-2)
    k_chunks, v_chunks = k.split(k_bucket_size, dim=-2), v.split(k_bucket_size, dim=-2)
    if enable_jvp:
        kt_chunks, vt_chunks = k_t.split(k_bucket_size, dim=-2), v_t.split(k_bucket_size, dim=-2)

    o_chunks, ot_chunks, lse_chunks = [], [], []
    for i, q_chunk in enumerate(q_chunks):
        q_len = q_chunk.shape[-2]
        q_slice = slice(i * q_bucket_size, i * q_bucket_size + q_len)
        qt_chunk = q_t[..., q_slice, :] if enable_jvp else None

        # running softmax stats (max detached: softmax is invariant to it)
        m = q_chunk.new_full((z, h, q_len, 1), float("-inf"))
        l_i = torch.ones_like(m)
        acc = torch.zeros_like(q_chunk)
        if enable_jvp:
            g_acc, mu, p_tv_acc = torch.zeros_like(acc), torch.zeros_like(m), torch.zeros_like(acc)

        for j, (k_chunk, v_chunk) in enumerate(zip(k_chunks, v_chunks)):
            k_len = k_chunk.shape[-2]
            k_start = j * k_bucket_size
            k_slice = slice(k_start, k_start + k_len)
            if causal and k_pos[k_slice][0] > q_pos[q_slice][-1]:
                break  # every key past this block is masked out

            bias, keep = mask_bias(mask_tensor, MASK_TYPE, q_slice, k_slice)
            scores = einsum("z h i d, z h j d -> z h i j", q_chunk, k_chunk)
            scores = scale_scores(scores + bias, qk_scale, causal, q_pos, k_pos, q_slice, k_slice)

            # online softmax; max is detached as softmax is shift-invariant
            m_ij = torch.maximum(m, scores.amax(dim=-1, keepdim=True)).detach()
            p = torch.exp2(scores - m_ij)
            if exists(keep):
                p = p * keep  # zero out masked keys after exp

            alpha = torch.exp2(m - m_ij)  # correction for the running max
            l_i = l_i * alpha + p.sum(dim=-1, keepdim=True)
            acc = acc * alpha + einsum("z h i j, z h j d -> z h i d", p, v_chunk)

            if enable_jvp:
                # t_scores = d(scores); the tangents flow through the same softmax
                t_scores = einsum("z h i d, z h j d -> z h i j", qt_chunk, k_chunk)
                t_scores = t_scores + einsum("z h i d, z h j d -> z h i j", q_chunk, kt_chunks[j])
                if exists(keep):
                    t_scores = t_scores * keep
                t_p = p * (t_scores * sm_scale)  # chain rule: base-2 -> natural scores
                g_acc = g_acc * alpha + einsum("z h i j, z h j d -> z h i d", t_p, v_chunk)
                mu = mu * alpha + t_p.sum(dim=-1, keepdim=True)
                p_tv_acc = p_tv_acc * alpha + einsum(
                    "z h i j, z h j d -> z h i d", p, vt_chunks[j]
                )

            m = m_ij

        # fully masked rows get lse = m, matching the kernel epilogue
        empty = l_i == 0.0
        l_i = torch.where(empty, 1.0, l_i)
        lse = m + torch.where(empty, 0.0, torch.log2(l_i))
        o_chunk = acc / l_i

        o_chunks.append(o_chunk.to(out_dtype))
        lse_chunks.append(lse.squeeze(-1))
        if enable_jvp:
            # t_o = (g_acc - mu * o + p_tv_acc) / l_i
            ot_chunks.append((g_acc / l_i - (mu / l_i) * o_chunk + p_tv_acc / l_i).to(out_dtype))

    o = cat(o_chunks, dim=-2)
    o_t = cat(ot_chunks, dim=-2) if enable_jvp else None
    return o, o_t, cat(lse_chunks, dim=-1)


def tiled_attention_backward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    o: Tensor,
    do: Tensor,
    lse: Tensor,
    mask_tensor: Tensor,
    MASK_TYPE: int,
    causal: bool,
    sm_scale: float,
    q_bucket_size: int = 128,
    k_bucket_size: int = 128,
) -> tuple[Tensor, Tensor, Tensor]:
    """Tiled backward -> (dq, dk, dv); recomputes p from lse, no autograd graph."""
    z, h, n_q, d = q.shape
    n_kv = k.shape[-2]
    device, out_dtype = q.device, q.dtype
    compute_dtype = default_dtype(q)
    q, k, v, o, do = (t.to(compute_dtype) for t in (q, k, v, o, do))
    lse = lse.to(compute_dtype).unsqueeze(-1)

    qk_scale = sm_scale * RCP_LN2
    q_pos = arange(n_kv - n_q, n_kv, device=device)
    k_pos = arange(n_kv, device=device)
    delta = (o * do).sum(dim=-1, keepdim=True)  # rowsum(o * do), kernel preprocess parity

    q_chunks = q.split(q_bucket_size, dim=-2)
    k_chunks, v_chunks = k.split(k_bucket_size, dim=-2), v.split(k_bucket_size, dim=-2)
    dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)

    for i, q_chunk in enumerate(q_chunks):
        q_slice = slice(i * q_bucket_size, i * q_bucket_size + q_chunk.shape[-2])
        do_chunk = do[..., q_slice, :]
        delta_chunk = delta[..., q_slice, :]
        dq_chunk = torch.zeros_like(q_chunk)

        for j, (k_chunk, v_chunk) in enumerate(zip(k_chunks, v_chunks)):
            k_len = k_chunk.shape[-2]
            k_start = j * k_bucket_size
            k_slice = slice(k_start, k_start + k_len)
            if causal and k_pos[k_slice][0] > q_pos[q_slice][-1]:
                break

            bias, keep = mask_bias(mask_tensor, MASK_TYPE, q_slice, k_slice)
            scores = einsum("z h i d, z h j d -> z h i j", q_chunk, k_chunk)
            scores = scale_scores(scores + bias, qk_scale, causal, q_pos, k_pos, q_slice, k_slice)

            p = torch.exp2(scores - lse[..., q_slice, :])  # recompute p from lse, no autograd graph
            if exists(keep):
                p = p * keep

            # softmax vjp -> raw scores
            dp = einsum("z h i d, z h j d -> z h i j", do_chunk, v_chunk)
            ds = p * (dp - delta_chunk) * sm_scale

            dq_chunk = dq_chunk + einsum("z h i j, z h j d -> z h i d", ds, k_chunk)
            dk[..., k_slice, :] += einsum("z h i j, z h i d -> z h j d", ds, q_chunk)
            dv[..., k_slice, :] += einsum("z h i j, z h i d -> z h j d", p, do_chunk)

        dq[..., q_slice, :] = dq_chunk

    return dq.to(out_dtype), dk.to(out_dtype), dv.to(out_dtype)


def naive_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    attn_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    causal: bool = False,
    sm_scale: float | None = None,
    verify_attn_mask: bool = True,
) -> Tensor:
    """Drop-in `JVPAttn.fwd` for CPU."""
    if dropout_p != 0.0:
        raise NotImplementedError("Dropout is not currently supported in JVP attention.")

    _, _, n, dim = q.shape
    dim_k, dim_v = k.shape[-1], v.shape[-1]
    assert dim == dim_k == dim_v, (
        "JVP attention requires equal query, key, and value head dimensions"
        f" but got {dim}, {dim_k}, and {dim_v}"
    )
    assert dim in {
        16,
        32,
        64,
        128,
        256,
    }, f"JVP attention only supports head dims in {{16, 32, 64, 128, 256}}, but got {dim}"
    assert (
        n >= 1 and k.shape[-2] >= n
    ), f"JVP attention requires sequence length >= 1 and key length >= query length, but got {n} and {k.shape[-2]}"
    if causal and exists(attn_mask):
        raise ValueError("Causal attention does not support an attention mask.")

    mask_tensor, MASK_TYPE = prepare_attn_mask(attn_mask, q, verify_attn_mask)
    sm_scale = dim**-0.5 if sm_scale is None else sm_scale

    out, _, _ = tiled_attention(q, k, v, mask_tensor, MASK_TYPE, causal, sm_scale)
    return out
