# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the FlashInfer project
"""Shape-parameterized reference/quantization helpers for the MoE monokernel.

Single source of the block-FP8 quantization, the fp8 and fp32 reference MoE,
and the routing/weight generators used by BOTH the accuracy test
(tests/moe/test_monomoe.py) and the autotuner (tools/tune_monomoe.py).  Every
helper takes (N, K) explicitly so it works for any registered shape, not just
the E256/N512/K2048 one.

These are pure PyTorch (no FlashInfer import), so both the pytest module and
the tuner can import them without pulling in the kernel until they choose to.
"""

import torch
import torch.nn.functional as F

_FP8_MAX = 448.0  # e4m3 dynamic range


def quant_fp8_block_wise(w, block_row=128, block_col=128):
    """Block-wise FP8 quantization. w: [E, rows, cols] -> (fp8, scales)."""
    Ee, rows, cols = w.shape
    rb = (rows + block_row - 1) // block_row
    cb = (cols + block_col - 1) // block_col
    wf = w.float()
    scales = torch.zeros(Ee, rb, cb, device=w.device, dtype=torch.float32)
    w_fp8 = torch.zeros_like(wf)
    for ri in range(rb):
        r0, r1 = ri * block_row, min((ri + 1) * block_row, rows)
        for ci in range(cb):
            c0, c1 = ci * block_col, min((ci + 1) * block_col, cols)
            block = wf[:, r0:r1, c0:c1]
            amax = block.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-12)
            s = amax / _FP8_MAX
            scales[:, ri, ci] = s.reshape(Ee)
            w_fp8[:, r0:r1, c0:c1] = (block / s).clamp(-_FP8_MAX, _FP8_MAX)
    return w_fp8.to(torch.float8_e4m3fn), scales


def quant_act_block_wise(x_float, group_size=128):
    """Per-token-dynamic 1xgroup_size activation quant. x_float: [K]."""
    n_groups = x_float.shape[0] // group_size
    x_fp8 = torch.zeros_like(x_float, dtype=torch.float8_e4m3fn)
    scales = torch.zeros(n_groups, device=x_float.device, dtype=torch.float32)
    for g in range(n_groups):
        g0, g1 = g * group_size, (g + 1) * group_size
        block = x_float[g0:g1]
        amax = block.abs().max().clamp(min=1e-12)
        scales[g] = amax / _FP8_MAX
        x_fp8[g0:g1] = (
            (block * (_FP8_MAX / amax))
            .clamp(-_FP8_MAX, _FP8_MAX)
            .to(torch.float8_e4m3fn)
        )
    return x_fp8, scales


def block_wise_gemm(w_fp8, scales_bw, x_fp8, x_scales, block_row=128, block_col=128):
    """result[row] = sum_blocks W*x * w_scale * x_scale.  w:[rows,cols], x:[cols]."""
    rows, cols = w_fp8.shape
    wf, xf = w_fp8.float(), x_fp8.float()
    rb, cb = scales_bw.shape
    result = torch.zeros(rows, device=w_fp8.device, dtype=torch.float32)
    for ri in range(rb):
        r0, r1 = ri * block_row, min((ri + 1) * block_row, rows)
        for ci in range(cb):
            c0, c1 = ci * block_col, min((ci + 1) * block_col, cols)
            result[r0:r1] += (
                (wf[r0:r1, c0:c1] @ xf[c0:c1]) * scales_bw[ri, ci] * x_scales[ci]
            )
    return result


def routing_softmax_topk(logits, top_k):
    """Greedy softmax->topk->renormalize, lowest-index tie-break (matches kernel)."""
    scores = torch.softmax(logits.float(), dim=-1)
    M = scores.shape[0]
    ids = torch.zeros(M, top_k, dtype=torch.int64, device=logits.device)
    wts = torch.zeros(M, top_k, dtype=torch.float32, device=logits.device)
    s = scores.clone()
    for k in range(top_k):
        v, idx = s.max(dim=-1)
        wts[:, k] = v
        ids[:, k] = idx
        s.scatter_(1, idx.unsqueeze(1), float("-inf"))
    wts = wts / wts.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    return wts, ids


def python_reference_fp8(x, w13_fp8, s13, w2_fp8, s2, topk_w, topk_ids, N, K):
    """Block-wise fp8 reference matching the kernel's math.  Returns out [M, K].

    This is the apples-to-apples fp8 reference (same block-dequant GEMM, SiLU,
    block re-quant of the intermediate, block down GEMM as the kernel), so the
    accuracy gate measures kernel bugs, not fp8 vs fp32 drift."""
    M, top_k = x.shape[0], topk_ids.shape[1]
    out = torch.zeros(M, K, device=x.device, dtype=torch.bfloat16)
    for tok in range(M):
        xq, x_scales = quant_act_block_wise(x[tok].float(), group_size=128)
        for ki in range(top_k):
            eid = int(topk_ids[tok, ki])
            rw = float(topk_w[tok, ki])
            raw = block_wise_gemm(w13_fp8[eid], s13[eid], xq, x_scales)
            gate, up = raw[:N], raw[N:]
            silu = rw * (up * gate) / (1.0 + torch.exp(-gate))
            silu_bf16 = silu.bfloat16()
            sq, s2_act = quant_act_block_wise(silu_bf16.float(), group_size=128)
            out[tok] += block_wise_gemm(w2_fp8[eid], s2[eid], sq, s2_act).bfloat16()
    return out


def moe_reference_fp32(x, w13, w2, topk_w, topk_ids, N, K):
    """FP32 ground truth from UNQUANTIZED weights (no fp8 anywhere).

    Uses the kernel's [gate || up] fc1 half-ordering and the given (already
    renormalized) top-K selection.  ``w13`` is [E, 2N, K], ``w2`` is [E, K, N].
    Vectorized over the intermediate dim (fast enough for tuner sweeps)."""
    m = x.shape[0]
    out = torch.zeros(m, K, device=x.device, dtype=torch.float32)
    xf, w13f, w2f = x.float(), w13.float(), w2.float()
    for t in range(m):
        for j in range(topk_ids.shape[1]):
            e = int(topk_ids[t, j])
            wgt = float(topk_w[t, j])
            gate = xf[t] @ w13f[e, :N, :].T
            up = xf[t] @ w13f[e, N:, :].T
            h = F.silu(gate) * up
            out[t] += wgt * (h @ w2f[e].T)
    return out


def make_weights(dev, E, N, K, scale=0.1, seed=42):
    """Random block-FP8 up/down weights for a shape.

    Returns (w13_fp8, s13, w2_fp8, s2, w13_f, w2_f): the quantized tensors +
    their scales, plus the unquantized fp32 weights (for the fp32 reference)."""
    g = torch.Generator(device=dev).manual_seed(seed)
    w13_f = torch.randn(E, 2 * N, K, device=dev, generator=g) * scale
    w2_f = torch.randn(E, K, N, device=dev, generator=g) * scale
    w13_fp8, s13 = quant_fp8_block_wise(w13_f)
    w2_fp8, s2 = quant_fp8_block_wise(w2_f)
    return w13_fp8, s13, w2_fp8, s2, w13_f, w2_f


def cosine(a, b):
    """Flattened cosine similarity of two tensors."""
    return F.cosine_similarity(
        a.float().reshape(-1), b.float().reshape(-1), dim=0
    ).item()
