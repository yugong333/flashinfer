"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Single-kernel ("mono") top-K Mixture-of-Experts, block-FP8, Hopper (SM90a).
The full pipeline — routing, up-projection, SiLU, down-projection and
reduction — runs inside one `__global__` launch.  See
docs/design_docs/monomoe_kernel.md for the design.

Supported shapes are a **curated registry** declared in
``csrc/fused_moe/monomoe/shapes.json`` (keyed by ``(E, N, K)``); one JIT
module is compiled per shape, baking that shape's tunable configs.  A tensor
whose ``(E, N, K)`` is not in the registry is rejected up front (add it to
shapes.json, regenerate, rebuild).  Token count M <= 16 (BS8 kernel for
M <= 8, BS16 companion for 8 < M <= 16).

Config selection (``config_id``) precedence, highest first:
  1. the explicit ``config_id=`` kwarg,
  2. the ``MONOMOE_CONFIG`` env var — bare id, per-shape ``shape:id``, or
     per-(shape, M) ``shape@M:id`` (see :func:`_parse_env_config`),
  3. the tuner-emitted best-config JSON — ``MONOMOE_TUNED_JSON`` if set, else
     the default dir ``$FLASHINFER_WORKSPACE_DIR/monomoe_tuned/`` where
     ``tune_monomoe.py`` writes by default; the config is chosen per token
     count M (largest tuned M' <= M),
  4. config 0 — the shipped default, byte-identical to the pre-tuning kernel.
A registered-but-untuned shape works immediately at config 0 (with a one-time
warning); tuning only improves performance.  Both the env per-M syntax and the
tuned JSON select the config based on the runtime batch size.
"""

import contextlib
import functools
import json
import os
import warnings
from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..trace.templates.moe import mono_moe_trace
from ..utils import backend_requirement, supported_compute_capability

_BLOCK = 128  # block-wise FP8 quantization tile (128 x 128)
_MONOMOE_BS_MAX = 16  # BS8 kernel serves M <= 8; BS16 serves 8 < M <= 16

_SCORING_SIGMOID = 0
_SCORING_SOFTMAX = 1


def _monomoe_csrc_dir():
    """Path to csrc/fused_moe/monomoe (installed package or dev checkout)."""
    from ..jit import env as jit_env

    standard = jit_env.FLASHINFER_CSRC_DIR / "fused_moe" / "monomoe"
    if standard.exists():
        return standard
    from pathlib import Path

    dev = Path(__file__).parent.parent.parent / "csrc" / "fused_moe" / "monomoe"
    if dev.exists():
        return dev
    raise FileNotFoundError(
        f"monomoe csrc dir not found (checked {standard} and {dev})."
    )


@functools.cache
def _load_registry() -> dict:
    """Parse shapes.json into an (E, N, K) -> shape-entry registry.

    shapes.json is the single source of truth (also drives the C++ code
    generator).  Only the fields the Python API needs are consumed here:
    (E, N, K), aliases, default_top_k, optional routing defaults, and the
    per-config knob table (for the UCH interleave decision)."""
    path = _monomoe_csrc_dir() / "shapes.json"
    with open(path) as f:
        data = json.load(f)
    by_enk: dict = {}
    by_name: dict = {}
    for s in data["shapes"]:
        if "key" not in s:  # skip "_comment"-only entries
            continue
        enk = (int(s["E"]), int(s["N"]), int(s["K"]))
        s["_enk"] = enk
        by_enk[enk] = s
        for nm in [s["key"], *s.get("aliases", [])]:
            by_name[nm.lower()] = s
    return {"by_enk": by_enk, "by_name": by_name, "path": str(path)}


def _registry_entry(E: int, N: int, K: int) -> Optional[dict]:
    """Registry entry for a shape, or None if the shape is not registered."""
    return _load_registry()["by_enk"].get((int(E), int(N), int(K)))


def registered_shapes() -> list:
    """List of (E, N, K) tuples for every registered shape (used by AOT)."""
    return list(_load_registry()["by_enk"].keys())


def _config_uch(entry: dict, config_id: int) -> int:
    """UP_COL_HALVES for a shape/config, mirroring the C++ up_col_halves
    detector and the generator's ``uch_of``.  Determines whether the up
    weights must be interleaved (UCH == 1) or read raw (UCH >= 2)."""
    N, K = entry["N"], entry["K"]
    configs = {int(c["id"]): c for c in entry["configs"]}
    c = configs.get(int(config_id), configs.get(0))
    if c is not None and c.get("uch") is not None:
        return int(c["uch"])
    if entry.get("up_col_halves") is not None:
        return int(entry["up_col_halves"])  # decoupled shape: pinned on Base
    dct = int(c["dct"]) if c is not None else 0
    v = (2 * N * dct) // (_BLOCK * K)
    return v if v > 0 else 1


@functools.cache
def _get_monomoe_module(E: int, N: int, K: int):
    """Lazily build and load the monomoe CUDA extension for one shape."""
    try:
        from ..jit.monomoe import load_monomoe_module

        return load_monomoe_module(int(E), int(N), int(K))
    except (ImportError, FileNotFoundError, RuntimeError) as e:
        raise ImportError(
            f"Failed to load the MonoMoe kernel CUDA extension via JIT for "
            f"(E={E}, N={N}, K={K}). Ensure a Hopper (SM90a) GPU and the CUDA "
            f"toolkit are available and that csrc/fused_moe/monomoe/ sources "
            f"exist.\nError: {e}"
        ) from e


def _resolve_config_id(E: int, N: int, K: int, entry: dict, m: int) -> int:
    """Resolve the config_id when the caller did not pass one explicitly.

    Precedence: MONOMOE_CONFIG env > tuner JSON (MONOMOE_TUNED_JSON or the
    default dir) > 0.  Both the env per-M syntax and the tuned JSON select on
    the runtime batch size ``m``.  Returns config 0 (the shipped default) when
    nothing pins a config; a one-time warning fires for a registered-but-
    untuned shape when MONOMOE_REQUIRE_TUNED is unset.  MONOMOE_REQUIRE_TUNED=1
    turns the untuned case into an error instead.
    """
    # 1. MONOMOE_CONFIG env (bare id | shape:id | shape@M:id, per (shape, M)).
    cid = _parse_env_config(E, N, K, entry, m)
    if cid is not None and cid >= 0:
        return cid
    # 2. tuner-emitted best-config JSON (explicit path or the default location).
    cid = _lookup_tuned_config(E, N, K, entry, m)
    if cid is not None and cid >= 0:
        return cid
    # 3. config 0 fallback (with the require-tuned policy).
    if os.environ.get("MONOMOE_REQUIRE_TUNED", "") not in ("", "0"):
        raise RuntimeError(
            f"monomoe shape (E={E}, N={N}, K={K}) is untuned for M={m} and "
            f"MONOMOE_REQUIRE_TUNED is set. Run tools/tune_monomoe.py and point "
            f"MONOMOE_TUNED_JSON at its output (or drop it in the default dir "
            f"{_default_tuned_dir()}), or pass config_id= explicitly."
        )
    _warn_untuned(entry["_enk"])
    return 0


@functools.cache
def _warn_untuned(enk: tuple) -> None:
    """One-time (per shape) warning that an untuned shape uses config 0."""
    warnings.warn(
        f"monomoe shape E={enk[0]} N={enk[1]} K={enk[2]} has no tuned config; "
        f"using config 0 (shipped default). Run tools/tune_monomoe.py (its "
        f"output lands in {_default_tuned_dir()} and is picked up "
        f"automatically), or set MONOMOE_TUNED_JSON / MONOMOE_CONFIG.",
        stacklevel=3,
    )


def _parse_env_config(E: int, N: int, K: int, entry: dict, m: int) -> Optional[int]:
    """Parse MONOMOE_CONFIG (or the deprecated MONOKERNEL_CONFIG) for this
    (shape, M).

    Syntax (comma-separated pins; first matching one wins per category):

      * ``2``                      — bare id, any shape and any M.
      * ``e256_n512_k2048:1``      — per-shape (by key/alias), all M.
      * ``E256N512K2048:1``        — per-shape (by E/N/K key), all M.
      * ``e256_n512_k2048@8:3``    — per (shape, M): applies to token counts
                                     ``>= 8`` up to the next larger pin, i.e.
                                     the largest pinned M' <= m is chosen
                                     (same bucketing as the tuned JSON).
      * ``E256N512K2048@16:7``     — per (shape, M) by E/N/K key.

    Per-(shape, M) pins take precedence over an all-M pin for the same shape,
    which takes precedence over a bare id.  Returns None if unset/unmatched.
    """
    raw = os.environ.get("MONOMOE_CONFIG")
    if raw is None:
        raw = os.environ.get("MONOKERNEL_CONFIG")
        if raw is not None:
            warnings.warn(
                "MONOKERNEL_CONFIG is deprecated; use MONOMOE_CONFIG.",
                stacklevel=4,
            )
    raw = (raw or "").strip()
    if not raw:
        return None

    names = {entry["key"].lower(), *[a.lower() for a in entry.get("aliases", [])]}
    enk_key = f"e{E}n{N}k{K}"

    bare: Optional[int] = None
    all_m: Optional[int] = None
    per_m: dict[int, int] = {}  # pinned M' -> config_id, for this shape
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:  # bare id (any shape / any M)
            if part.lstrip("-").isdigit():
                bare = int(part)
            continue
        key, _, val = part.partition(":")
        key = key.strip().lower()
        val = val.strip()
        if not val.lstrip("-").isdigit():
            continue
        cid = int(val)
        # `key` is either `shape` (all-M) or `shape@M` (per-M).
        if "@" in key:
            shape_key, _, m_str = key.partition("@")
            shape_key = shape_key.strip()
            if (shape_key in names or shape_key == enk_key) and m_str.isdigit():
                per_m[int(m_str)] = cid
        elif key in names or key == enk_key:
            all_m = cid

    # Per-(shape, M) pin wins when a pinned M' <= m exists (largest such M').
    # `shape@M:id` reads as "for token counts >= M"; below the smallest per-M
    # pin we fall through to the all-M pin (`shape:id`), then the bare id.
    eligible = sorted(mm for mm in per_m if mm <= m)
    if eligible:
        return per_m[eligible[-1]]
    if all_m is not None:
        return all_m
    return bare


def _default_tuned_dir() -> str:
    """Default directory searched for tuner output when MONOMOE_TUNED_JSON is
    unset: ``$FLASHINFER_WORKSPACE_DIR/monomoe_tuned`` (mirrors the MLA
    autotuner's ``.../autotune`` convention).  ``tune_monomoe.py`` writes here
    by default, so a tuned deployment is picked up with no env var."""
    from ..jit.env import FLASHINFER_WORKSPACE_DIR

    return str(FLASHINFER_WORKSPACE_DIR / "monomoe_tuned")


def _tuned_json_paths(entry: dict) -> list:
    """Candidate tuner-JSON paths, most specific first.

    If ``MONOMOE_TUNED_JSON`` is set it is the only candidate (explicit wins).
    Otherwise the default dir is searched for a per-shape file
    ``<key>.json`` and a combined ``monomoe_tuned.json``."""
    explicit = os.environ.get("MONOMOE_TUNED_JSON", "").strip()
    if explicit:
        return [explicit]
    d = _default_tuned_dir()
    return [
        os.path.join(d, f"{entry['key']}.json"),
        os.path.join(d, "monomoe_tuned.json"),
    ]


def _rec_for_shape(doc, E: int, N: int, K: int, entry: dict):
    """Extract the (E,N,K) record from a tuner-JSON document that is either a
    single shape record (has ``best_per_M``) or ``{shape_key: record}``."""
    if not isinstance(doc, dict):
        return None
    if "best_per_M" in doc:
        if (doc.get("E"), doc.get("N"), doc.get("K")) == (E, N, K):
            return doc
        return None
    return doc.get(entry["key"]) or next(
        (
            v
            for v in doc.values()
            if isinstance(v, dict) and (v.get("E"), v.get("N"), v.get("K")) == (E, N, K)
        ),
        None,
    )


def _lookup_tuned_config(E: int, N: int, K: int, entry: dict, m: int) -> Optional[int]:
    """Look up the best config for (shape, M) in the tuner JSON.

    Uses ``MONOMOE_TUNED_JSON`` when set, else the default dir
    (:func:`_default_tuned_dir`).  Picks the largest tuned M' <= m (matching
    the kernel's BS8/BS16 bucketing).  Returns None if no file / shape / M
    matches."""
    for path in _tuned_json_paths(entry):
        if not path or not os.path.exists(path):
            continue
        try:
            with open(path) as f:
                doc = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        rec = _rec_for_shape(doc, E, N, K, entry)
        if not rec:
            continue
        best = rec.get("best_per_M", {})
        tuned_ms = sorted(int(k) for k in best)
        if not tuned_ms:
            continue
        pick = None
        for tm in tuned_ms:
            if tm <= m:
                pick = tm
        if pick is None:
            pick = tuned_ms[0]
        return int(best[str(pick)]["config_id"])
    return None


@functools.cache
@flashinfer_api
def has_monomoe(E: int = 256, N: int = 512, K: int = 2048) -> bool:
    """Return True if the monomoe CUDA extension can be built and loaded for
    the given shape (default: the E256/N512/K2048 shape)."""
    if _registry_entry(E, N, K) is None:
        return False
    try:
        _get_monomoe_module(E, N, K)
        return True
    except ImportError:
        return False


@functools.cache
@flashinfer_api
def get_scratchpad_size_bytes(
    E: int, N: int, K: int, config_id: Optional[int] = None
) -> int:
    """Return the global scratchpad size (bytes) required by the kernel for a
    shape.

    Sourced from the C++ `sizeof(MoEGemmSpec<Dims>)` so the buffer can never
    desync from the kernel's struct layout.  With ``config_id`` set, sizes to
    that config (BS8 + BS16 variant); with ``config_id=None`` sizes to the max
    over ALL of the shape's configs, so one buffer can be reused across config
    switches without reallocation.
    """
    mod = _get_monomoe_module(E, N, K)
    if config_id is None:
        return int(mod.monomoe_scratchpad_size_max())
    return int(mod.monomoe_scratchpad_size(int(config_id)))


@flashinfer_api
def alloc_scratchpad(
    device: torch.device,
    E: int = 256,
    N: int = 512,
    K: int = 2048,
    config_id: Optional[int] = None,
) -> torch.Tensor:
    """Allocate a zero-initialized scratchpad on ``device`` for the kernel.

    Returns a 1-D ``uint8`` tensor sized to
    ``get_scratchpad_size_bytes(E, N, K, config_id)``.  With the default
    ``config_id=None`` the buffer is sized to the max over all of the shape's
    configs, so it can be reused across config switches.  The zero fill
    establishes the kernel's handoff invariants (the 0.0f activation-scale
    sentinel, launch parity counters, and readiness flags — see
    docs/design_docs/monomoe_kernel.md §2/§4); afterwards the kernel
    self-maintains them, so allocate once and reuse the same tensor for every
    :func:`mono_moe` invocation of that shape.
    """
    nbytes = get_scratchpad_size_bytes(E, N, K, config_id)
    return torch.zeros(nbytes, dtype=torch.uint8, device=device)


@flashinfer_api
def interleave_for_tma_wgmma_up(w_fp8: torch.Tensor) -> torch.Tensor:
    """Repack fp8 up-projection weights for the Pair_Layout WGMMA A-tile.

    Under the pair layout, each warp's 16-row SHM stripe holds 8 gate rows
    and 8 up rows, so ``silu(gate) * up`` becomes a per-lane register
    operation after the WGMMA (no cross-warp exchange).

    Input layout: ``[E, 2*N, K]`` row-major fp8 — the first ``N`` rows per
    expert are gate weights, the last ``N`` are up weights.  ``N`` must be a
    multiple of 64.

    Output layout (still ``[E, 2*N, K]``, identical byte footprint): for
    every expert ``e`` and every 64-gate-row block ``b``, the 128-row slab
    at ``128*b`` packs, per warpgroup ``wg`` and warp ``w``::

        rows [wg*64 + w*16     .. +8) = gate[e, 64b + wg*32 + w*8 .. +8, :]
        rows [wg*64 + w*16 + 8 .. +8) =   up[e, 64b + wg*32 + w*8 .. +8, :]

    Under SWZ128 the TMA applies the 8-row x 128-byte core-matrix XOR swizzle
    automatically, so this only rearranges GM rows (no byte-level
    permutation).  The result is cached on the input tensor as
    ``_tma_interleaved_up``.

    The down-projection weights need no preparation — the raw ``[E, K, N]``
    row-major fp8 tensor is passed straight through.

    Parameters
    ----------
    w_fp8 : torch.Tensor
        FP8 up/gate weight tensor with shape [E, 2*N, K] (row-major).

    Returns
    -------
    torch.Tensor
        Repacked weight tensor with the same shape and dtype as *w_fp8*.
    """
    cached = getattr(w_fp8, "_tma_interleaved_up", None)
    if cached is not None:
        return cached

    E, rows, K = w_fp8.shape
    if rows % 2 != 0:
        raise ValueError(f"expected rows = 2*N, got rows={rows}")
    n_half = rows // 2
    if n_half % 64 != 0:
        raise ValueError(f"N (half of rows) must be a multiple of 64; got N={n_half}")

    gate = w_fp8[:, :n_half, :]
    up = w_fp8[:, n_half:, :]

    blocks = n_half // 64
    gate_r = gate.reshape(E, blocks, 64, K)
    up_r = up.reshape(E, blocks, 64, K)

    # Per warpgroup (32 rows) and warp (8 rows): gate(8) then up(8) = one
    # 16-row warp stripe; 4 warps per WG, 2 WGs per 128-row slab.
    gate_wg0 = gate_r[:, :, :32, :].reshape(E, blocks, 4, 8, K)
    gate_wg1 = gate_r[:, :, 32:, :].reshape(E, blocks, 4, 8, K)
    up_wg0 = up_r[:, :, :32, :].reshape(E, blocks, 4, 8, K)
    up_wg1 = up_r[:, :, 32:, :].reshape(E, blocks, 4, 8, K)

    wg0 = torch.stack([gate_wg0, up_wg0], dim=3).reshape(E, blocks, 64, K)
    wg1 = torch.stack([gate_wg1, up_wg1], dim=3).reshape(E, blocks, 64, K)
    result = torch.cat([wg0, wg1], dim=2).reshape(E, blocks * 128, K).contiguous()

    with contextlib.suppress(AttributeError, RuntimeError):
        w_fp8._tma_interleaved_up = result
    return result


def _check_shapes(
    activations_in: torch.Tensor,
    router_logits: torch.Tensor,
    expert_weights_up: torch.Tensor,
    expert_scales_up: torch.Tensor,
    expert_weights_down: torch.Tensor,
    expert_scales_down: torch.Tensor,
) -> tuple:
    """Validate the input tensors against the curated shape registry.

    Derives ``(E, N, K)`` from the down weights (``[E, K, N]``), confirms the
    shape is registered, cross-checks every operand's extents, and returns
    ``(m, E, N, K, entry)``.  A shape not in shapes.json is refused.
    """
    # Explicit raises (not assert): these validate user-provided tensor
    # shapes that, if wrong, let the fixed-shape CUDA kernel read/write out
    # of bounds.  `assert` would be stripped under `python -O`.
    if activations_in.dim() != 2:
        raise ValueError(f"activations_in must be [M, K], got {activations_in.dim()}D")
    if expert_weights_down.dim() != 3:
        raise ValueError(
            f"expert_weights_down must be [E, K, N], got {expert_weights_down.dim()}D"
        )
    if expert_weights_up.dim() != 3:
        raise ValueError(
            f"expert_weights_up must be [E, 2*N, K], got {expert_weights_up.dim()}D"
        )

    # (E, N, K) come from the down weights [E, K, N]; K is cross-checked
    # against the activations' feature dim.
    E, K, N = (int(x) for x in expert_weights_down.shape)
    entry = _registry_entry(E, N, K)
    if entry is None:
        raise ValueError(
            f"monomoe: unregistered shape (E={E}, N={N}, K={K}). Supported "
            f"shapes: {registered_shapes()}. To add one, declare it in "
            f"csrc/fused_moe/monomoe/shapes.json, regenerate "
            f"(tools/gen_shapes.py), and rebuild."
        )

    BS = _MONOMOE_BS_MAX
    m = activations_in.size(0)
    if m > BS:
        raise ValueError(f"this kernel caps tokens at {BS}; got M={m}")
    if activations_in.size(1) != K:
        raise ValueError(f"activations K must be {K}, got {activations_in.size(1)}")
    if tuple(router_logits.shape) != (m, E):
        raise ValueError(
            f"router_logits must be [{m}, {E}], got {tuple(router_logits.shape)}"
        )

    # Up weights: [E, 2*N, K]; up scales block-wise [E, 2N/128, K/128].
    if tuple(expert_weights_up.shape) != (E, 2 * N, K):
        raise ValueError(
            f"expert_weights_up must be [{E}, {2 * N}, {K}], got {tuple(expert_weights_up.shape)}"
        )
    if tuple(expert_scales_up.shape) != (E, (2 * N) // _BLOCK, K // _BLOCK):
        raise ValueError(
            f"expert_scales_up must be [{E}, {(2 * N) // _BLOCK}, {K // _BLOCK}], "
            f"got {tuple(expert_scales_up.shape)}"
        )
    # Down scales block-wise [E, K/128, N/128].
    if tuple(expert_scales_down.shape) != (E, K // _BLOCK, N // _BLOCK):
        raise ValueError(
            f"expert_scales_down must be [{E}, {K // _BLOCK}, {N // _BLOCK}], "
            f"got {tuple(expert_scales_down.shape)}"
        )
    return m, E, N, K, entry


@supported_compute_capability([90])
def _check_mono_moe_supported(
    activations_in: torch.Tensor,
    router_logits: torch.Tensor,
    expert_weights_up: torch.Tensor,
    expert_scales_up: torch.Tensor,
    expert_weights_down: torch.Tensor,
    expert_scales_down: torch.Tensor,
    top_k: int,
    scoring_func: str = "softmax",
    renormalize: bool = True,
    expert_bias: Optional[torch.Tensor] = None,
    routed_scaling_factor: float = 1.0,
    out: Optional[torch.Tensor] = None,
    scratchpad: Optional[torch.Tensor] = None,
    interleave_up: bool = True,
    config_id: Optional[int] = None,
) -> bool:
    """Backend-requirement check for :func:`mono_moe`.

    Carries the ``@supported_compute_capability([90])`` annotation so the
    ``@backend_requirement`` wrapper rejects any non-Hopper device with a
    clear ``BackendSupportedError`` *before* the SM90a-only kernel is JIT
    compiled.  Also runs the registry shape-contract validation so shape and
    architecture support are decided in one place.  The signature mirrors
    :func:`mono_moe` because ``@backend_requirement`` forwards all kwargs.
    """
    _check_shapes(
        activations_in,
        router_logits,
        expert_weights_up,
        expert_scales_up,
        expert_weights_down,
        expert_scales_down,
    )
    return True


@backend_requirement({}, common_check=_check_mono_moe_supported)
@flashinfer_api(trace=mono_moe_trace)
def mono_moe(
    activations_in: torch.Tensor,
    router_logits: torch.Tensor,
    expert_weights_up: torch.Tensor,
    expert_scales_up: torch.Tensor,
    expert_weights_down: torch.Tensor,
    expert_scales_down: torch.Tensor,
    top_k: int,
    scoring_func: str = "softmax",
    renormalize: bool = True,
    expert_bias: Optional[torch.Tensor] = None,
    routed_scaling_factor: float = 1.0,
    out: Optional[torch.Tensor] = None,
    scratchpad: Optional[torch.Tensor] = None,
    interleave_up: bool = True,
    config_id: Optional[int] = None,
) -> torch.Tensor:
    """Single-kernel block-FP8 top-K MoE (curated shapes, SM90a only).

    The ``(E, N, K)`` shape is derived from the weight tensors and must be one
    of the shapes registered in ``csrc/fused_moe/monomoe/shapes.json``; up to
    16 tokens (the BS8 kernel serves M <= 8; a BS16 companion serves
    8 < M <= 16).

    Args:
        activations_in: bf16 input activations ``[M, K]`` (``M <= 16``).
        router_logits: bf16 router logits ``[M, E]``.
        expert_weights_up: fp8_e4m3 up/gate weights ``[E, 2*N, K]``. For an
            interleaved (UCH==1) config this function applies
            :func:`interleave_for_tma_wgmma_up` by default; for a raw (UCH>=2)
            config the tensor is passed through untouched (interleaving is
            skipped regardless of ``interleave_up``). Pass ``interleave_up=False``
            if a UCH==1 tensor is already interleaved.
        expert_scales_up: fp32 block-wise scales ``[E, 2N/128, K/128]``.
        expert_weights_down: fp8_e4m3 down weights ``[E, K, N]`` (raw row-major).
        expert_scales_down: fp32 block-wise scales ``[E, K/128, N/128]``.
        top_k: experts selected per token (1..8).
        scoring_func: ``"sigmoid"`` or ``"softmax"``.
        renormalize: renormalize the top-K weights to sum to 1.
        expert_bias: optional fp32 per-expert selection bias ``[E]``
            (GLM-style noaux_tc routing; sigmoid scoring only).  Winners are
            ranked by ``sigmoid(logit) + bias`` while the routing weight
            stays the unbiased sigmoid.
        routed_scaling_factor: scalar folded into every routing weight.
        out: optional bf16 output buffer ``[M, K]``; allocated if omitted.
        scratchpad: optional reusable uint8 scratchpad from
            :func:`alloc_scratchpad`; allocated per-call if omitted.
        interleave_up: apply the gate/up TMA repack to ``expert_weights_up``
            for UCH==1 configs (default True).
        config_id: tunable KernelConfig id (see shapes.json).  ``None`` (the
            default) resolves via ``MONOMOE_CONFIG`` env, the tuner JSON, then
            config 0.  Pass an explicit id to override that resolution (the
            tuner and tests do this).

    Returns:
        bf16 MoE output ``[M, K]``.
    """
    if not (1 <= top_k <= 8):
        raise ValueError(f"top_k must be in [1, 8], got {top_k}")
    sf_map = {"sigmoid": _SCORING_SIGMOID, "softmax": _SCORING_SOFTMAX}
    if scoring_func not in sf_map:
        raise ValueError(
            f"scoring_func must be 'sigmoid' or 'softmax', got {scoring_func!r}"
        )

    m, E, N, K, entry = _check_shapes(
        activations_in,
        router_logits,
        expert_weights_up,
        expert_scales_up,
        expert_weights_down,
        expert_scales_down,
    )

    if expert_bias is not None:
        if scoring_func != "sigmoid":
            raise ValueError("expert_bias requires scoring_func='sigmoid'")
        if tuple(expert_bias.shape) != (E,):
            raise ValueError(
                f"expert_bias must be [{E}], got {tuple(expert_bias.shape)}"
            )

    # Resolve config_id: explicit kwarg wins; else env / tuner-JSON / config 0.
    if config_id is None:
        config_id = _resolve_config_id(E, N, K, entry, m)
    config_id = int(config_id)

    # Interleave only for UCH==1 (interleaved) configs.  A raw (UCH>=2) config
    # reads the unmodified [E, 2*N, K] tensor; interleaving it would corrupt
    # the up-projection, so the request is overridden to False.
    if interleave_up and _config_uch(entry, config_id) == 1:
        expert_weights_up = interleave_for_tma_wgmma_up(expert_weights_up)

    if out is None:
        out = torch.empty(m, K, dtype=torch.bfloat16, device=activations_in.device)
    if scratchpad is None:
        scratchpad = alloc_scratchpad(activations_in.device, E, N, K, config_id)

    mod = _get_monomoe_module(E, N, K)
    mod.monomoe_topk(
        activations_in,
        router_logits,
        expert_weights_up,
        expert_scales_up,
        expert_weights_down,
        expert_scales_down,
        out,
        scratchpad,
        int(top_k),
        int(sf_map[scoring_func]),
        bool(renormalize),
        expert_bias,
        float(routed_scaling_factor),
        config_id,
    )
    return out
