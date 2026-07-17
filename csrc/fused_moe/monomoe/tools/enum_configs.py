#!/usr/bin/env python3
"""Monokernel config feasibility enumerator.

Given an MoE shape (N, K, E, BS) and GPU limits (SM count, SHM cap), enumerate
the VALID monokernel KernelConfig tuples and emit them as JSON for the
instantiation generator (task 6) and the tuning harness (task 7).

Two enumeration modes:

  coupled   (Phase A) — one blocks-per-expert (bpe) shared by up and down.
            bpe | gcd(2N/128, K/128).  Tunables: groups (grid = groups*bpe
            <= SMs), K_STEP_UP, K_STEP_DOWN, UP_W_SLOTS.

  decoupled (Phase B) — up and down bpe chosen separately.
            up_bpe | (2N/128), down_bpe | (K/128); grid equality
            up_groups*up_bpe == down_groups*down_bpe <= SMs; barrier ratio
            R = up_groups/down_groups must be a positive integer.

The geometry constraints below mirror the C++ `MoECoreDims` static_asserts.
SHM is estimated analytically here for fast pruning; the AUTHORITATIVE size is
`sizeof(MoE_SHM<Dims>)`, which `--verify` cross-checks by generating a probe
.cu, compiling it with nvcc, and reading back the real bytes (the kernel's own
budget static_assert is the final gate).

Usage:
  enum_configs.py --shape e256_n1024_k3072 [--mode coupled] [--json out.json]
  enum_configs.py --shape e256_n512_k2048 --mode coupled --verify  # nvcc check
  enum_configs.py --N 1024 --K 3072 --E 256 --bs 8 --mode both
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

# ── GPU / kernel limits ─────────────────────────────────────────────────
H200_SM_COUNT = 132  # fallback when no CUDA device can be queried
# Conservative fallback budget (H100/H200 opt-in max is ~227 KB; the kernel's
# historical per-Dims static_assert ceiling was 224 KB).  detect_shm_budget()
# replaces this with the queried hardware value when a GPU is present.
SHM_BUDGET_BYTES = 224 * 1024


def detect_shm_budget(default=SHM_BUDGET_BYTES):
    """Return the per-block dynamic shared-memory ceiling (bytes) of the local
    GPU — the ``opt-in`` maximum, which is what a kernel gets after
    ``cudaFuncSetAttribute(MaxDynamicSharedMemorySize)``.

    This is the hard cap on ``sizeof(MoE_SHM<Dims>)`` for any feasible config,
    so it must track the actual hardware (H100/H200 ≈ 227 KB opt-in, A100
    ≈ 164 KB, L40S ≈ 99 KB, B200 ≈ 227 KB).  The static 224 KB assumed the
    H-series; on a smaller-SHM card it would wrongly admit infeasible configs.

      1. torch.cuda.get_device_properties(d).shared_memory_per_block_optin —
         the opt-in max (exactly the attribute the kernel raises to).
      2. cudart cudaDeviceGetAttribute(MaxSharedMemoryPerBlockOptin=97).
      3. `default` as a last resort, with a stderr warning.

    Returns the integer byte budget.  Never raises — falls back to `default`.
    """
    # 1) torch — shared_memory_per_block_optin is the opt-in dynamic-SHM max.
    try:
        import torch

        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            v = getattr(
                torch.cuda.get_device_properties(dev),
                "shared_memory_per_block_optin",
                0,
            )
            if v:
                return int(v)
    except Exception:
        pass

    # 2) cudart attribute 97 = cudaDevAttrMaxSharedMemoryPerBlockOptin.
    try:
        import ctypes

        rt = ctypes.CDLL("libcudart.so")
        val = ctypes.c_int()
        if rt.cudaDeviceGetAttribute(ctypes.byref(val), 97, 0) == 0 and val.value:
            return int(val.value)
    except Exception:
        pass

    # 3) last resort.
    print(
        f"[enum_configs] WARNING: could not detect SHM opt-in budget; "
        f"using default {default // 1024} KB (pass --shm-budget to override).",
        file=sys.stderr,
    )
    return default


def detect_sm_count(default=H200_SM_COUNT):
    """Return the SM (streaming-multiprocessor) count of the local CUDA GPU.

    The grid feasibility math is ``grid = groups * bpe <= SM_COUNT`` (coupled)
    / ``grid <= SM_COUNT`` (decoupled), so the SM count is the hard ceiling on
    how many CTAs the persistent monokernel can occupy.  Hardcoding 132 (H200)
    silently over- or under-counts on any other card, so detect it:

      1. torch.cuda.get_device_properties(d).multi_processor_count — exact,
         and torch is already in the build/run venv.
      2. nvidia-smi → device name → known-SM lookup, when torch is absent.
      3. `default` (H200=132) as a last resort, with a stderr warning.

    Returns the integer SM count.  Never raises — falls back to `default`.
    """
    # 1) torch (authoritative; multi_processor_count is the real SM count).
    try:
        import torch

        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            return torch.cuda.get_device_properties(dev).multi_processor_count
    except Exception:
        pass

    # 2) nvidia-smi name → SM-count table (no torch needed).
    KNOWN_SM = {
        "H200": 132,
        "H100": 132,
        "H800": 132,
        "A100": 108,
        "A800": 108,
        "L40S": 142,
        "L40": 142,
        "L4": 58,
        "B200": 148,
        "GH200": 132,
    }
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            timeout=10,
        ).decode()
        name = out.splitlines()[0].strip() if out.strip() else ""
        for key, sm in KNOWN_SM.items():
            if key in name:
                return sm
        if name:
            print(
                f"[enum_configs] WARNING: GPU '{name}' not in the SM table; "
                f"falling back to {default} SMs (pass --sms to override).",
                file=sys.stderr,
            )
    except Exception:
        pass

    # 3) last resort.
    print(
        f"[enum_configs] WARNING: could not detect GPU SM count; "
        f"using default {default} (pass --sms to override).",
        file=sys.stderr,
    )
    return default


ATOM = 128  # SWZ128 / WGMMA-M software atom (col & K unit)
WGMMA_K = 128  # one K-substep = 128 (4x m64n8k32)
BS_DEFAULT = 8
TOPK_MAX = 8

# Named shapes (N = moe_intermediate_size = N_HALF; fused gate+up = 2N).
# Shape-keyed E{E}_N{N}_K{K}; a shape not listed here can still be enumerated
# via explicit --N/--K/--E/--bs.
SHAPES = {
    "e256_n512_k2048": dict(N=512, K=2048, E=256, bs=8),
    "e256_n1024_k3072": dict(N=1024, K=3072, E=256, bs=8),
}

# Tunable ranges (multiples of 128 upward for K-steps; powers of two for slots).
K_STEP_CHOICES = [128, 256, 384, 512, 768]
UP_W_SLOTS_CHOICES = [2, 4, 8]
# Down-proj weight/activation TMA ring depth: 2 = classic double buffer,
# 4 = rolling ring (2 K-steps of lookahead; needs K_TILES_DOWN % 4 == 0).
DOWN_PIPE_DEPTH_CHOICES = [2, 4]


def divisors_in_atoms(total_rows):
    """bpe values (in 128-row atoms) that evenly divide `total_rows/128`."""
    n_atoms = total_rows // ATOM
    return [d for d in range(1, n_atoms + 1) if n_atoms % d == 0]


def shm_estimate(N, K, bs, dct, uch, kup, kdn, slots, dpd=2):
    """Analytic SHM estimate (bytes) for fast pruning.

    Mirrors the MoE_SHM<Dims> layout: the dominant region is the max of the
    three weight/activation buffers that alias each other in time, plus the
    BS/DPD-dependent buffers modeled explicitly (they scale with the new
    tunables — the old flat remainder was calibrated for BS8/DPD2 only) and
    a small fixed remainder for the rest (scales, mbarriers, routing
    tables).

    The estimate is used ONLY for fast PRE-pruning, so it must be a LOWER
    bound: never drop a config that is actually feasible (a false negative
    is unrecoverable; a false positive is caught by the exact `--verify`
    gate).  The AUTHORITATIVE size is `sizeof(MoE_SHM<D>)` from --verify.
    """
    t_tile = 8 if bs <= 8 else bs
    # bf16_in_full: [K/128][bs][128] bf16 = K*bs*2 bytes.
    bf16_in = K * bs * 2
    # up weight slots: SLOTS * (UCH atoms) * (K_STEP_UP/128 substeps) * 128*128 fp8.
    up_w = slots * uch * (kup // WGMMA_K) * ATOM * ATOM
    # down weight ring: DPD slots * (DCT/128 atoms) * (KDN/128 substeps) * 128*128 fp8.
    down_w = dpd * (dct // ATOM) * (kdn // WGMMA_K) * ATOM * ATOM
    union = max(bf16_in, up_w, down_w)
    # BS/DPD-dependent buffers outside the union:
    #   fp8_act_full [K/128][8][T_TILE+1][16] fp8   = K * (T_TILE+1)
    #   a_down_wgmma ring [DPD][KDN/128][T_TILE][8][16] fp8
    #   out_accum    [BS][DCT+1] fp32
    #   partial_result union: max(down_out [DCT][T_TILE], post_silu
    #                              [UCH*128][T_TILE+1]) fp32
    fp8_act = K * (t_tile + 1)
    a_down = dpd * (kdn // WGMMA_K) * t_tile * 128
    out_accum = bs * (dct + 1) * 4
    partial = max(dct * t_tile, uch * 128 * (t_tile + 1)) * 4
    # Fixed remainder lower bound (scales, mbarriers, routing tables,
    # act_scale, topk arrays): intentionally small so no feasible config is
    # pruned; boundary cases are settled by --verify.
    fixed = 2 * 1024
    return union + fp8_act + a_down + out_accum + partial + fixed


# ── Geometry validity (mirrors MoECoreDims static_asserts) ───────────────
def coupled_uch(N, dct, K):
    """UP_COL_HALVES derived from the coupling UP_GROUPS == DOWN_GROUPS."""
    num = 2 * N * dct
    den = ATOM * K
    if num % den != 0:
        return None  # non-integer => infeasible coupling
    return num // den


def enum_coupled(N, K, E, bs, sm_count, shm_budget):
    """Phase A: shared bpe. Yields config dicts."""
    up_rows = 2 * N
    down_rows = K
    # bpe must divide BOTH up atoms and down atoms.
    up_div = set(divisors_in_atoms(up_rows))
    down_div = set(divisors_in_atoms(down_rows))
    bpe_atoms = sorted(up_div & down_div)
    out = []
    for bpe in bpe_atoms:
        dct = down_rows // bpe  # down col tile (rows in K)
        up_tile = up_rows // bpe  # up col tile (rows in 2N)
        if dct % ATOM != 0 or up_tile % ATOM != 0:
            continue
        uch = coupled_uch(N, dct, K)
        if uch is None or uch < 1:
            continue
        # Hard kernel limit (until task 4 lifts it): UP_COL_HALVES <= 2.
        if uch > 2:
            continue
        # DOWN_COL_HALVES kernel limit: <= 4 (DCT <= 512), matching the
        # static_assert in moe_down_projection.cu.
        if dct // ATOM > 4:
            continue
        down_grid = K // dct
        for slots in UP_W_SLOTS_CHOICES:
            for kup in K_STEP_CHOICES:
                if K % kup != 0:
                    continue
                k_tiles_up = K // kup
                if k_tiles_up % slots != 0 or k_tiles_up < slots:
                    continue
                # Deferred up-proj epilogue: ceil(BS/4) waves must fit in
                # K_TILES_UP - 1 iterations (WAVES <= DEFER_ITERS assert).
                if k_tiles_up - 1 < (bs + 3) // 4:
                    continue
                for kdn in K_STEP_CHOICES:
                    # Down-proj reduction dim is N (NOT K).
                    if N % kdn != 0:
                        continue
                    # Inter-expert lookahead slot parity: K_TILES_DOWN even
                    # (kernel static_assert in moe_down_projection.cuh).
                    if (N // kdn) % 2 != 0:
                        continue
                    for dpd in DOWN_PIPE_DEPTH_CHOICES:
                        # DPD=4 needs K_TILES_DOWN % 4 == 0 so the
                        # cross-expert stitch lands ring slots 0,1 (kernel
                        # static_assert in moe_internal.h).
                        if dpd == 4 and (N // kdn) % 4 != 0:
                            continue
                        # groups range: grid = groups*bpe <= SMs; grid must
                        # be a multiple of both UP_GRID(=bpe) and DOWN_GRID.
                        max_groups = sm_count // bpe
                        for groups in range(1, max_groups + 1):
                            grid = groups * bpe
                            if grid % down_grid != 0:
                                continue
                            # Each expert group must own >= 1 expert.
                            if groups > E or grid // down_grid > E:
                                continue
                            shm = shm_estimate(N, K, bs, dct, uch, kup, kdn, slots, dpd)
                            if shm > shm_budget:
                                continue
                            out.append(
                                dict(
                                    mode="coupled",
                                    N=N,
                                    K=K,
                                    E=E,
                                    bs=bs,
                                    grid=grid,
                                    groups=groups,
                                    bpe=bpe,
                                    down_col_tile=dct,
                                    up_col_halves=uch,
                                    k_step_up=kup,
                                    k_step_down=kdn,
                                    up_w_slots=slots,
                                    down_pipe_depth=dpd,
                                    up_grid=bpe,
                                    down_grid=down_grid,
                                    shm_est=shm,
                                    sms_used=grid,
                                )
                            )
    return out


def enum_decoupled(N, K, E, bs, sm_count, shm_budget):
    """Phase B: up_bpe and down_bpe chosen separately."""
    up_rows = 2 * N
    out = []
    up_bpes = divisors_in_atoms(up_rows)
    down_bpes = divisors_in_atoms(K)
    for up_bpe in up_bpes:
        up_tile = up_rows // up_bpe
        uch = up_tile // ATOM
        if uch < 1 or uch > 2:  # kernel limit until task 4
            continue
        for down_bpe in down_bpes:
            dct = K // down_bpe
            if dct % ATOM != 0 or dct // ATOM > 4:  # kernel: DCT <= 512
                continue
            # grid equality + integer barrier ratio.
            #   grid = up_groups*up_bpe = down_groups*down_bpe <= SMs
            #   R = up_groups / down_groups must be a positive integer.
            # grid must be a common multiple of up_bpe and down_bpe.
            base = up_bpe * down_bpe // math.gcd(up_bpe, down_bpe)  # lcm
            for grid in range(base, sm_count + 1, base):
                up_groups = grid // up_bpe
                down_groups = grid // down_bpe
                if up_groups % down_groups != 0:
                    continue
                R = up_groups // down_groups
                if R < 1:
                    continue
                # Each expert group must own >= 1 expert.
                if up_groups > E or down_groups > E:
                    continue
                for slots in UP_W_SLOTS_CHOICES:
                    for kup in K_STEP_CHOICES:
                        if K % kup != 0:
                            continue
                        k_tiles_up = K // kup
                        if k_tiles_up % slots != 0 or k_tiles_up < slots:
                            continue
                        # Deferred epilogue: ceil(BS/4) waves must fit in
                        # K_TILES_UP - 1 iterations.
                        if k_tiles_up - 1 < (bs + 3) // 4:
                            continue
                        for kdn in K_STEP_CHOICES:
                            if N % kdn != 0:
                                continue
                            # Inter-expert lookahead: K_TILES_DOWN even.
                            if (N // kdn) % 2 != 0:
                                continue
                            for dpd in DOWN_PIPE_DEPTH_CHOICES:
                                if dpd == 4 and (N // kdn) % 4 != 0:
                                    continue
                                shm = shm_estimate(
                                    N, K, bs, dct, uch, kup, kdn, slots, dpd
                                )
                                if shm > shm_budget:
                                    continue
                                out.append(
                                    dict(
                                        mode="decoupled",
                                        N=N,
                                        K=K,
                                        E=E,
                                        bs=bs,
                                        grid=grid,
                                        up_groups=up_groups,
                                        down_groups=down_groups,
                                        barrier_ratio=R,
                                        up_bpe=up_bpe,
                                        down_bpe=down_bpe,
                                        down_col_tile=dct,
                                        up_col_halves=uch,
                                        k_step_up=kup,
                                        k_step_down=kdn,
                                        up_w_slots=slots,
                                        down_pipe_depth=dpd,
                                        shm_est=shm,
                                        sms_used=grid,
                                    )
                                )
    return out


# ── nvcc cross-check (authoritative SHM) ─────────────────────────────────
PROBE_TEMPLATE = r"""
#define INSIDE_MONOMOE_IMPLEMENTATION
#include "moe_interface.h"
#include "moe_internal.h"
#include <cstdio>
using namespace monomoe;
template <uint32_t NN, uint32_t KK, uint32_t BSZ, uint32_t GRID, uint32_t DCT,
          uint32_t KUP, uint32_t KDN, uint32_t SLOTS, uint32_t DPD>
struct GDims {
  static constexpr uint32_t HIDDEN_STATES = KK;
  static constexpr uint32_t K = KK;
  static constexpr uint32_t N = NN;
  static constexpr uint32_t BS = BSZ;
  static constexpr uint32_t M = BSZ;
  static constexpr uint32_t NUM_EXPERTS = 256;
  static constexpr QuantGranularity QUANT_GRAN = QuantGranularity::BLOCK_WISE;
  static constexpr uint32_t BLOCK_SCALE_ROW = 128;
  static constexpr uint32_t BLOCK_SCALE_COL = 128;
  static constexpr uint32_t UP_SCALE_ROWS = (2 * NN + 127) / 128;
  static constexpr uint32_t UP_SCALE_COLS = (KK + 127) / 128;
  static constexpr uint32_t DOWN_SCALE_ROWS = (KK + 127) / 128;
  static constexpr uint32_t DOWN_SCALE_COLS = (NN + 127) / 128;
  struct KernelConfig {
    static constexpr uint32_t GRID_SIZE = GRID;
    static constexpr uint32_t BLOCK_SIZE = 384;
    static constexpr bool USE_WGMMA = true;
    static constexpr bool USE_TMA = true;
    static constexpr uint32_t K_STEP_DOWN = KDN;
    static constexpr uint32_t K_STEP_UP = KUP;
    static constexpr uint32_t DOWN_COL_TILE = DCT;
    static constexpr uint32_t UP_W_SLOTS = SLOTS;
    static constexpr uint32_t DOWN_PIPE_DEPTH = DPD;
    static constexpr bool USE_PAIR_LAYOUT = true;
  };
};
template <typename D> void row(int idx) {
  printf("SHM %d %zu\n", idx, sizeof(MoE_SHM<D>));
}
int main() {
__ROWS__
  return 0;
}
"""


def _compile_run(rows_src, idxs, src_dir):
    """Compile one TU containing `row<...>(idx)` lines; return {idx: bytes}.

    A static_assert / budget failure makes the WHOLE TU fail, so the caller
    bisects to attribute infeasibility to specific configs.
    """
    src = PROBE_TEMPLATE.replace("__ROWS__", rows_src)
    with tempfile.TemporaryDirectory() as td:
        cu = os.path.join(td, "probe.cu")
        exe = os.path.join(td, "probe")
        with open(cu, "w") as f:
            f.write(src)
        cp = subprocess.run(
            [
                "nvcc",
                "-gencode",
                "arch=compute_90a,code=sm_90a",
                "-std=c++17",
                "--expt-relaxed-constexpr",
                "-I",
                src_dir,
                cu,
                "-o",
                exe,
            ],
            capture_output=True,
            text=True,
        )
        if cp.returncode != 0:
            return None  # at least one config in this batch is infeasible
        run = subprocess.run([exe], capture_output=True, text=True)
        out = {}
        for line in run.stdout.splitlines():
            if line.startswith("SHM "):
                _, idx, b = line.split()
                out[int(idx)] = int(b)
        return out


def verify_shm(configs, src_dir):
    """Exact sizeof(MoE_SHM) per config, via batched nvcc compiles.

    Fast path: compile all configs in ONE TU.  If that fails (some config
    trips a static_assert / SHM-budget gate), bisect the batch so the
    feasible majority still gets exact sizes and the infeasible ones are
    marked None (dropped).  Returns {idx: bytes-or-None}.
    """

    def row_line(i, c):
        return "  row<GDims<%d,%d,%d,%d,%d,%d,%d,%d,%d>>(%d);" % (
            c["N"],
            c["K"],
            c["bs"],
            c["grid"],
            c["down_col_tile"],
            c["k_step_up"],
            c["k_step_down"],
            c["up_w_slots"],
            c.get("down_pipe_depth", 2),
            i,
        )

    results = {}

    def recurse(idxs):
        if not idxs:
            return
        rows = "\n".join(row_line(i, configs[i]) for i in idxs)
        got = _compile_run(rows, idxs, src_dir)
        if got is not None:
            results.update(got)
            return
        if len(idxs) == 1:
            results[idxs[0]] = None  # this single config is infeasible
            return
        mid = len(idxs) // 2
        recurse(idxs[:mid])
        recurse(idxs[mid:])

    recurse(list(range(len(configs))))
    return results


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--shape",
        choices=SHAPES.keys(),
        help="named shape (e256_n512_k2048 / e256_n1024_k3072)",
    )
    ap.add_argument("--N", type=int, help="moe_intermediate_size (N_HALF)")
    ap.add_argument("--K", type=int, help="hidden_size")
    ap.add_argument("--E", type=int, default=256)
    ap.add_argument("--bs", type=int, default=BS_DEFAULT)
    ap.add_argument(
        "--mode", choices=["coupled", "decoupled", "both"], default="coupled"
    )
    ap.add_argument(
        "--sms",
        type=int,
        default=None,
        help="SM count ceiling for grid feasibility (grid <= SMs). "
        "Default: auto-detect the local GPU (torch / "
        "nvidia-smi), falling back to H200=132.",
    )
    ap.add_argument(
        "--shm-budget",
        type=int,
        default=None,
        help="Per-block dynamic SHM ceiling in bytes (sizeof "
        "MoE_SHM must fit). Default: auto-detect the local "
        "GPU's opt-in max, falling back to 224 KB.",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="nvcc cross-check exact sizeof(MoE_SHM) + feasibility",
    )
    ap.add_argument("--json", help="write config list to this path")
    ap.add_argument(
        "--src-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"),
        help="monokernel src dir (for --verify nvcc -I)",
    )
    args = ap.parse_args()

    # Resolve hardware ceilings: explicit flags win; otherwise detect the GPU.
    if args.sms is None:
        args.sms = detect_sm_count()
    if args.shm_budget is None:
        args.shm_budget = detect_shm_budget()

    if args.shape:
        s = SHAPES[args.shape]
        N, K, E, bs = s["N"], s["K"], s["E"], s["bs"]
    elif args.N and args.K:
        N, K, E, bs = args.N, args.K, args.E, args.bs
    else:
        ap.error("provide --shape or both --N and --K")

    configs = []
    if args.mode in ("coupled", "both"):
        configs += enum_coupled(N, K, E, bs, args.sms, args.shm_budget)
    if args.mode in ("decoupled", "both"):
        configs += enum_decoupled(N, K, E, bs, args.sms, args.shm_budget)

    print(
        f"# shape N={N} K={K} E={E} bs={bs} | mode={args.mode} | "
        f"SMs={args.sms} SHM_budget={args.shm_budget // 1024}KB",
        file=sys.stderr,
    )
    print(f"# {len(configs)} candidate(s) after analytic pruning", file=sys.stderr)

    if args.verify:
        src_dir = os.path.normpath(args.src_dir)
        print(f"# verifying exact SHM via nvcc (-I {src_dir}) ...", file=sys.stderr)
        shm = verify_shm(configs, src_dir)
        kept = []
        n_assert = n_budget = 0
        for i, c in enumerate(configs):
            real = shm.get(i)
            if real is None:
                # `sizeof(MoE_SHM<D>)` instantiates MoECoreDims, so a
                # compile failure here means a GEOMETRY static_assert tripped
                # (divisibility, K_TILES%SLOTS, UP_COL_HALVES<=2, ...).
                n_assert += 1
                continue
            c["shm_exact"] = real
            c["shm_exact_kb"] = real // 1024
            # The kernel-body budget static_assert is NOT exercised by the
            # sizeof probe, so enforce it here against the EXACT size (this
            # is the authoritative budget gate, tighter than shm_est).
            if real > args.shm_budget:
                n_budget += 1
                continue
            kept.append(c)
        configs = kept
        print(
            f"# {len(configs)} feasible after nvcc verify "
            f"({n_assert} rejected by geometry static_assert, "
            f"{n_budget} over SHM budget)",
            file=sys.stderr,
        )

    # Sort: fewer SMs first is NOT the goal; keep stable by (grid desc, shm asc).
    configs.sort(key=lambda c: (-c["sms_used"], c.get("shm_exact", c["shm_est"])))

    if args.json:
        with open(args.json, "w") as f:
            json.dump(
                dict(
                    N=N,
                    K=K,
                    E=E,
                    bs=bs,
                    mode=args.mode,
                    sms=args.sms,
                    shm_budget=args.shm_budget,
                    configs=configs,
                ),
                f,
                indent=2,
            )
        print(f"# wrote {len(configs)} configs -> {args.json}", file=sys.stderr)

    # Human-readable table to stdout.
    for c in configs:
        shm_kb = c.get("shm_exact_kb", c["shm_est"] // 1024)
        tag = "exact" if "shm_exact_kb" in c else "est"
        dpd = c.get("down_pipe_depth", 2)
        if c["mode"] == "coupled":
            print(
                f"coupled  grid={c['grid']:3d} bpe={c['bpe']} "
                f"groups={c['groups']:2d} DCT={c['down_col_tile']:3d} "
                f"UCH={c['up_col_halves']} KUP={c['k_step_up']:3d} "
                f"KDN={c['k_step_down']:3d} SLOTS={c['up_w_slots']} "
                f"DPD={dpd} SHM={shm_kb}KB({tag})"
            )
        else:
            print(
                f"decoup   grid={c['grid']:3d} up={c['up_groups']}x{c['up_bpe']} "
                f"dn={c['down_groups']}x{c['down_bpe']} R={c['barrier_ratio']} "
                f"DCT={c['down_col_tile']:3d} UCH={c['up_col_halves']} "
                f"KUP={c['k_step_up']:3d} KDN={c['k_step_down']:3d} "
                f"SLOTS={c['up_w_slots']} DPD={dpd} SHM={shm_kb}KB({tag})"
            )


if __name__ == "__main__":
    main()
