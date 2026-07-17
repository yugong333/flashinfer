"""Accuracy test for the single-kernel block-FP8 MoE (monomoe).

The kernel serves the curated shapes in
``csrc/fused_moe/monomoe/shapes.json`` on Hopper (SM90a): block-wise (128x128)
FP8 weights, per-token-dynamic 1x128 activation quantization, up to 16 tokens
(BS8 kernel for M<=8, BS16 companion for 8<M<=16 where the shape opts in).

The block-FP8 Python reference (shared with the tuner via
``tools/_tune_helpers``) replicates the kernel's exact math (block-wise dequant
GEMM, SiLU gating, block-wise re-quant of the intermediate, block-wise down
GEMM), so the comparison is apples-to-apples in fp8.

Two shapes are exercised by default: the E256/N512/K2048 shape (with a BS16
companion) and the cheaper E64/N512/K2048 shape (BS8-only), so multi-shape
JIT + the per-shape BS16 gating are both covered on one GPU.
"""

import os
import sys

import pytest
import torch

from flashinfer.fused_moe import alloc_scratchpad, has_monomoe, mono_moe
from flashinfer.utils import is_sm90a_supported

# Shared shape-parameterized helpers (single implementation, also used by the
# autotuner) live under the monomoe tools dir.
_TOOLS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "csrc",
    "fused_moe",
    "monomoe",
    "tools",
)
sys.path.insert(0, os.path.normpath(_TOOLS))
import _tune_helpers as H  # noqa: E402

BLOCK = 128

# (E, N, K) test shapes.  e256_n512_k2048: has a BS16 companion;
# e64_n512_k2048: BS8-only, cheap.
SHAPES = [(256, 512, 2048), (64, 512, 2048)]


def _make_weights(dev, E, N, K, scale=0.1, seed=42):
    """Quantized block-FP8 up/down weights (fp8 tensors + scales) for a shape."""
    w13_fp8, s13, w2_fp8, s2, _w13f, _w2f = H.make_weights(
        dev, E, N, K, scale=scale, seed=seed
    )
    return w13_fp8, s13, w2_fp8, s2


def _run_and_compare(x, logits, weights, N, K, top_k, scratchpad=None, config_id=None):
    """Run mono_moe against the block-FP8 reference; return (out, cos)."""
    w13_fp8, s13, w2_fp8, s2 = weights
    topk_w, topk_ids = H.routing_softmax_topk(logits, top_k)
    ref = H.python_reference_fp8(x, w13_fp8, s13, w2_fp8, s2, topk_w, topk_ids, N, K)

    # mono_moe applies the up-weight TMA interleave internally by default.
    out = mono_moe(
        x,
        logits,
        w13_fp8,
        s13,
        w2_fp8,
        s2,
        top_k=top_k,
        scoring_func="softmax",
        renormalize=True,
        scratchpad=scratchpad,
        config_id=config_id,
    )
    assert out.shape == x.shape
    assert out.dtype == torch.bfloat16
    cos = H.cosine(out, ref)
    return out, cos


def _require_monomoe(dev, E=256, N=512, K=2048):
    if not is_sm90a_supported(dev):
        pytest.skip("monomoe requires SM90a (Hopper)")
    if not has_monomoe(E, N, K):
        pytest.skip(f"monomoe unavailable for (E={E}, N={N}, K={K})")


def _m_supported(E, N, K, m):
    """A BS8-only shape (no bs16) cannot serve M>8."""
    import json

    if m <= 8:
        return True
    with open(os.path.join(_TOOLS, "..", "shapes.json")) as f:
        shapes = json.load(f)["shapes"]
    for s in shapes:
        if (s.get("E"), s.get("N"), s.get("K")) == (E, N, K):
            return bool(s.get("bs16"))
    return False


# M spans both the BS8 kernel (M<=8) and the BS16 companion (8<M<=16).
@pytest.mark.parametrize("E,N,K", SHAPES)
@pytest.mark.parametrize("m", [1, 2, 8, 9, 16])
@pytest.mark.parametrize("top_k", [1, 8])
def test_monomoe_accuracy(E, N, K, m, top_k):
    dev = torch.device("cuda")
    _require_monomoe(dev, E, N, K)
    if not _m_supported(E, N, K, m):
        pytest.skip(f"(E={E},N={N},K={K}) has no BS16 companion; M={m} > 8")

    torch.manual_seed(42)
    weights = _make_weights(dev, E, N, K)
    x = torch.randn(m, K, device=dev, dtype=torch.bfloat16)
    logits = torch.randn(m, E, device=dev, dtype=torch.bfloat16)

    _, cos = _run_and_compare(x, logits, weights, N, K, top_k)
    print(f"\n[monomoe] E={E} N={N} K={K} m={m} top_k={top_k}: cos_sim={cos:.5f}")
    assert cos > 0.98, f"cosine similarity too low: {cos:.5f}"


@pytest.mark.parametrize("m", [1, 8, 16])
def test_monomoe_scratchpad_reuse_no_contamination(m):
    """A scratchpad reused across calls with DIFFERENT inputs must not leak
    state between launches.

    The kernel's cross-block handoffs (the act-scale sentinel and the
    Phase-4->5 arrival counters) live in the scratchpad and are meant to be
    self-maintaining via a per-launch parity double-buffer.  If that reset
    discipline were wrong, a second launch on the same buffer could observe
    a stale scale/flag from the first and corrupt its output.  We run several
    independent problems through ONE scratchpad and require each to match its
    own reference — a fresh-scratchpad control run is compared against too.
    """
    E, N, K = 256, 512, 2048
    dev = torch.device("cuda")
    _require_monomoe(dev, E, N, K)

    weights = _make_weights(dev, E, N, K, seed=7)
    scratch = alloc_scratchpad(dev, E, N, K)

    # Distinct (x, logits) per iteration so a leaked buffer would show up as a
    # mismatch against THIS iteration's reference.
    prev_out = None
    for i in range(4):
        torch.manual_seed(100 + i)
        x = torch.randn(m, K, device=dev, dtype=torch.bfloat16)
        logits = torch.randn(m, E, device=dev, dtype=torch.bfloat16)

        out_reuse, cos_reuse = _run_and_compare(
            x, logits, weights, N, K, top_k=8, scratchpad=scratch
        )
        # Control: a fresh scratchpad for the same inputs must match exactly.
        out_fresh, _ = _run_and_compare(
            x, logits, weights, N, K, top_k=8, scratchpad=alloc_scratchpad(dev, E, N, K)
        )

        print(f"\n[monomoe reuse] m={m} iter={i}: cos_sim={cos_reuse:.5f}")
        assert cos_reuse > 0.98, f"reuse iter {i}: cos too low {cos_reuse:.5f}"
        # Reused-buffer output must match the fresh-buffer output for the same
        # inputs.  Not bit-exact: the Phase-4 cross-block atomicAdd reduces in
        # nondeterministic order, so runs differ by up to a rounding ULP —
        # contamination (a leaked scale/flag) would instead flip whole
        # rows/experts, far above this tolerance.
        cos_rf = H.cosine(out_reuse, out_fresh)
        assert cos_rf > 0.9999, (
            f"reuse iter {i}: reuse vs fresh diverged ({cos_rf:.6f})"
        )
        if prev_out is not None:
            # Different inputs must produce a different output (guards against
            # the kernel silently returning a stale buffer).
            assert not torch.equal(out_reuse, prev_out)
        prev_out = out_reuse.clone()


@pytest.mark.parametrize("scale", [1e-2, 3e-3])
def test_monomoe_small_scales(scale):
    """Correctness must hold when weight/activation magnitudes are tiny.

    Block-FP8 rescales each 128x128 tile to the e4m3 range, so uniformly
    scaling the inputs is numerically inert *until* the OUTPUT magnitude
    (~scale^2 here) sinks below the fp8 pipeline's dynamic-range floor — at
    which point the fp8 reference itself loses correlation with fp32, so the
    kernel can no longer be distinguished from it.  These scales keep the
    output above that floor (~1e-6) while still exercising the tiny-scale
    path: the act-scale sentinel publish clamps to >= FLT_MIN so a
    legitimately small scale can never flush to the 0.0f "not published"
    sentinel and mis-handoff a consumer.
    """
    E, N, K = 256, 512, 2048
    dev = torch.device("cuda")
    _require_monomoe(dev, E, N, K)

    torch.manual_seed(2024)
    weights = _make_weights(dev, E, N, K, scale=scale, seed=11)
    x = torch.randn(8, K, device=dev, dtype=torch.bfloat16) * scale
    logits = torch.randn(8, E, device=dev, dtype=torch.bfloat16)

    out, cos = _run_and_compare(x, logits, weights, N, K, top_k=8)
    # No NaN/Inf even at the fp8 subnormal boundary.
    assert torch.isfinite(out.float()).all(), "small-scale output has NaN/Inf"
    print(f"\n[monomoe small-scale] scale={scale:g}: cos_sim={cos:.5f}")
    assert cos > 0.98, f"cosine similarity too low: {cos:.5f}"


@pytest.mark.parametrize("config_id", [0, 1, 2, 3, 4, 5, 6, 7])
def test_monomoe_config_id(config_id):
    """Every tunable config of the E256/N512/K2048 shape must (a) run and match
    the fp8 reference, and (b) config 0 must be byte-identical to the default
    (no-arg) call — the config-0 identity guarantee."""
    E, N, K, m = 256, 512, 2048, 8
    dev = torch.device("cuda")
    _require_monomoe(dev, E, N, K)

    torch.manual_seed(123)
    weights = _make_weights(dev, E, N, K)
    x = torch.randn(m, K, device=dev, dtype=torch.bfloat16) * 0.1
    logits = torch.randn(m, E, device=dev, dtype=torch.bfloat16)

    _, cos = _run_and_compare(x, logits, weights, N, K, top_k=8, config_id=config_id)
    print(f"\n[monomoe cfg{config_id}] cos_sim={cos:.5f}")
    assert cos > 0.98, f"config {config_id}: cosine too low {cos:.5f}"

    if config_id == 0:
        # Explicit config 0 and the default resolution (also config 0 when
        # untuned) run the SAME instantiated kernel (bare Base — the config-0
        # identity is structural in the binding).  They are not bit-for-bit
        # across launches because Phase-4's cross-block atomicAdd reduces in
        # nondeterministic order (same ULP jitter as running any config twice),
        # so the check is the same >0.9999 cosine used by the reuse test — a
        # DIFFERENT kernel would diverge far more.
        out0, _ = _run_and_compare(x, logits, weights, N, K, top_k=8, config_id=0)
        out_default, _ = _run_and_compare(x, logits, weights, N, K, top_k=8)
        assert H.cosine(out0, out_default) > 0.9999, (
            "config 0 must match the default resolution (config-0 identity)"
        )


def test_monomoe_unregistered_shape_refused():
    """A shape not in shapes.json is refused before any kernel launch."""
    dev = torch.device("cuda")
    _require_monomoe(dev)
    x = torch.randn(8, 2048, device=dev, dtype=torch.bfloat16)
    logits = torch.randn(8, 256, device=dev, dtype=torch.bfloat16)
    # N=999 is not a registered shape.
    w13 = torch.zeros(256, 2 * 999, 2048, device=dev, dtype=torch.float8_e4m3fn)
    s13 = torch.zeros(256, 2 * 999 // 128, 2048 // 128, device=dev)
    w2 = torch.zeros(256, 2048, 999, device=dev, dtype=torch.float8_e4m3fn)
    s2 = torch.zeros(256, 2048 // 128, 999 // 128 + 1, device=dev)
    with pytest.raises(ValueError, match="unregistered shape"):
        mono_moe(x, logits, w13, s13, w2, s2, top_k=8)


# ── config resolution (CPU-only; no kernel build) ───────────────────────────
# These exercise the batch-size-aware config selection directly, so they run
# without a GPU / JIT build.

import flashinfer.fused_moe.monomoe as _mm  # noqa: E402


@pytest.fixture
def _clean_config_env(monkeypatch):
    """Clear all MONOMOE_* config env vars so each test starts from a known
    state (config resolution reads these live)."""
    for var in (
        "MONOMOE_CONFIG",
        "MONOKERNEL_CONFIG",
        "MONOMOE_TUNED_JSON",
        "MONOMOE_REQUIRE_TUNED",
    ):
        monkeypatch.delenv(var, raising=False)
    yield monkeypatch


def test_env_config_per_shape_and_bare(_clean_config_env):
    """Bare id, per-shape key, and E/N/K key forms of MONOMOE_CONFIG."""
    entry = _mm._registry_entry(256, 512, 2048)
    key = entry["key"]  # e256_n512_k2048

    _clean_config_env.setenv("MONOMOE_CONFIG", "2")
    assert _mm._parse_env_config(256, 512, 2048, entry, m=8) == 2

    _clean_config_env.setenv("MONOMOE_CONFIG", f"{key}:3")
    assert _mm._parse_env_config(256, 512, 2048, entry, m=8) == 3

    _clean_config_env.setenv("MONOMOE_CONFIG", "E256N512K2048:5")
    assert _mm._parse_env_config(256, 512, 2048, entry, m=8) == 5

    # A pin for a DIFFERENT shape must not match this one.
    _clean_config_env.setenv("MONOMOE_CONFIG", "e64_n512_k2048:2")
    assert _mm._parse_env_config(256, 512, 2048, entry, m=8) is None


def test_env_config_per_m(_clean_config_env):
    """`shape@M:id` selects by batch size (largest pinned M' <= m), falling
    through to an all-M pin, then a bare id, below the smallest per-M pin."""
    entry = _mm._registry_entry(256, 512, 2048)
    key = entry["key"]

    _clean_config_env.setenv("MONOMOE_CONFIG", f"{key}@1:3,{key}@16:7")
    assert _mm._parse_env_config(256, 512, 2048, entry, m=1) == 3
    assert _mm._parse_env_config(256, 512, 2048, entry, m=8) == 3  # largest <= 8
    assert _mm._parse_env_config(256, 512, 2048, entry, m=12) == 3
    assert _mm._parse_env_config(256, 512, 2048, entry, m=16) == 7

    # Precedence: per-M (>= its M) beats all-M beats bare; below the smallest
    # per-M pin, fall through to the all-M pin.
    _clean_config_env.setenv("MONOMOE_CONFIG", f"9,{key}:5,{key}@16:7")
    assert _mm._parse_env_config(256, 512, 2048, entry, m=1) == 5  # all-M
    assert _mm._parse_env_config(256, 512, 2048, entry, m=8) == 5
    assert _mm._parse_env_config(256, 512, 2048, entry, m=16) == 7  # per-M

    # E/N/K per-M form.
    _clean_config_env.setenv("MONOMOE_CONFIG", "E256N512K2048@8:6")
    assert _mm._parse_env_config(256, 512, 2048, entry, m=8) == 6
    assert _mm._parse_env_config(256, 512, 2048, entry, m=1) is None  # below pin


def test_tuned_json_explicit_and_default_dir(_clean_config_env, tmp_path):
    """The tuned-JSON lookup picks per-M configs from MONOMOE_TUNED_JSON when
    set, and from the default dir otherwise."""
    import json

    entry = _mm._registry_entry(256, 512, 2048)
    key = entry["key"]
    doc = {
        "shape_key": key,
        "E": 256,
        "N": 512,
        "K": 2048,
        "top_k": 8,
        "best_per_M": {"1": {"config_id": 3}, "16": {"config_id": 7}},
    }

    # 1. Explicit MONOMOE_TUNED_JSON path.
    p = tmp_path / "best.json"
    p.write_text(json.dumps(doc))
    _clean_config_env.setenv("MONOMOE_TUNED_JSON", str(p))
    assert _mm._lookup_tuned_config(256, 512, 2048, entry, m=1) == 3
    assert _mm._lookup_tuned_config(256, 512, 2048, entry, m=8) == 3  # largest <= 8
    assert _mm._lookup_tuned_config(256, 512, 2048, entry, m=16) == 7
    _clean_config_env.delenv("MONOMOE_TUNED_JSON", raising=False)

    # 2. Default dir (no env var): drop <key>.json into _default_tuned_dir().
    default_dir = _mm._default_tuned_dir()
    os.makedirs(default_dir, exist_ok=True)
    dst = os.path.join(default_dir, f"{key}.json")
    existed = os.path.exists(dst)
    backup = None
    if existed:
        with open(dst) as f:
            backup = f.read()
    try:
        with open(dst, "w") as f:
            json.dump(doc, f)
        assert _mm._lookup_tuned_config(256, 512, 2048, entry, m=16) == 7
        assert _mm._lookup_tuned_config(256, 512, 2048, entry, m=8) == 3
    finally:
        if backup is not None:
            with open(dst, "w") as f:
                f.write(backup)
        elif os.path.exists(dst):
            os.remove(dst)
