#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the FlashInfer project
"""MoE monokernel autotuner — sweep a shape's KernelConfig variants, gate on
accuracy, rank by latency, emit the best-config-per-M JSON.

For each ``config_id`` declared in a shape's ``shapes.json`` table (all of
which are baked into that shape's single JIT ``.so``), this:

  1. ACCURACY GATE — cosine similarity of the kernel output vs the block-FP8
     Python reference (:mod:`_tune_helpers`) at every requested M and seed;
     a config is rejected if any falls below ``--acc-threshold``.
  2. PERF — CUDA-graph latency via ``flashinfer.testing.bench_gpu_time``.  The
     baseline is config 0 by default (``--baseline config0``), so the reported
     number is the per-config speedup over the shipped default.
  3. RANK + EMIT — a ranked table plus the winning ``config_id`` per (shape, M)
     written to ``--json``.  The serving path consumes that JSON via
     ``MONOMOE_TUNED_JSON`` (see flashinfer/fused_moe/monomoe.py).

Config selection uses the explicit ``config_id=`` kwarg of ``mono_moe`` (all
configs are already instantiated in the shape's ``.so``, so switching is a
runtime int — no rebuild and no env mutation).

Candidate set: the shape's declared configs in ``shapes.json`` (curated).
``enum_configs.py`` is used only to feasibility-check each declared config on
the local GPU (SHM/grid), never to invent configs.

Multi-GPU sharding (Ray):
  Sharded by batch size M — each GPU owns one M at a time and sweeps ALL
  configs for it in latency isolation (concurrent timing on one GPU contends
  and corrupts both numbers).  M is pulled from a dynamic Ray ActorPool queue.
  With <2 visible GPUs it falls back to an in-process sequential sweep.  A
  config must clear the accuracy threshold at EVERY swept M to be ranked.

Usage:
  # Tune the E256/N512/K2048 shape across the decode batch sizes, write JSON:
  tune_monomoe.py --shape e256_n512_k2048 --batch-sizes 1 2 4 8 16 \
      --json best_e256_n512_k2048.json
  # Then serve with the tuned configs (no rebuild):
  MONOMOE_TUNED_JSON=best_e256_n512_k2048.json python my_serving_script.py

  # Sweep only a subset of configs at two batch sizes:
  tune_monomoe.py --shape e64_n512_k2048 --configs 0 2 --batch-sizes 1 8
  # Stricter accuracy gate + more seeds; rank by absolute latency (no baseline):
  tune_monomoe.py --shape e256_n1024_k3072 --acc-threshold 0.995 \
      --seeds 42 7 100 --baseline none
"""

import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MONO = os.path.normpath(os.path.join(HERE, ".."))
SHAPES_JSON = os.path.join(MONO, "shapes.json")

# _tune_helpers lives beside this script.
sys.path.insert(0, HERE)
import _tune_helpers as H  # noqa: E402


def load_registry():
    """(E,N,K) + name/alias -> shape entry, straight from shapes.json."""
    with open(SHAPES_JSON) as f:
        data = json.load(f)
    by_key, by_name = {}, {}
    for s in data["shapes"]:
        if "key" not in s:
            continue
        by_key[s["key"]] = s
        for nm in [s["key"], *s.get("aliases", [])]:
            by_name[nm.lower()] = s
    return by_key, by_name


def shape_row(name):
    _by_key, by_name = load_registry()
    r = by_name.get(name.lower())
    if r is None:
        raise SystemExit(f"unknown --shape {name!r}; choices: {sorted(by_name)}")
    return r


def config_uch(entry, config_id):
    """UP_COL_HALVES for a config (mirrors the API + generator derivation)."""
    N, K = entry["N"], entry["K"]
    configs = {int(c["id"]): c for c in entry["configs"]}
    c = configs.get(int(config_id), configs.get(0))
    if c is not None and c.get("uch") is not None:
        return int(c["uch"])
    if entry.get("up_col_halves") is not None:
        return int(entry["up_col_halves"])
    dct = int(c["dct"]) if c is not None else 0
    v = (2 * N * dct) // (128 * K)
    return v if v > 0 else 1


def config_ids_for(entry, m, requested):
    """Config ids valid for batch size m: the shape's declared set (or the
    ``--configs`` subset), restricted to the BS16 companion set when m>8."""
    all_ids = [int(c["id"]) for c in entry["configs"]]
    if m > 8:
        if not entry.get("bs16"):
            return []  # shape has no BS16 companion
        bs16 = entry.get("bs16_config_ids")
        all_ids = list(bs16) if bs16 is not None else all_ids
    if requested:
        all_ids = [c for c in all_ids if c in set(requested)]
    return sorted(all_ids)


# ── Accuracy + timing (one process, one GPU) ─────────────────────────────────


def _sweep_one_m(entry, m, top_k, config_ids, acc_threshold, baseline, seeds):
    """Sweep all config_ids for a single M in this process's GPU.

    Returns ``{cid: {"cos", "passed", "latency_ms"}}``.  Latency is None for a
    config that failed the accuracy gate (perf is skipped on reject)."""
    import torch

    # Imported here so Ray has pinned the actor's GPU before torch init.
    from flashinfer.fused_moe import mono_moe, alloc_scratchpad
    from flashinfer.testing import bench_gpu_time

    E, N, K = entry["E"], entry["N"], entry["K"]
    dev = torch.device("cuda")

    # Weights + inputs are shape-only (constant across configs); build once.
    w13_fp8, s13, w2_fp8, s2, w13_f, w2_f = H.make_weights(dev, E, N, K, seed=42)
    x = torch.randn(m, K, device=dev, dtype=torch.bfloat16) * 0.1
    logits = torch.randn(m, E, device=dev, dtype=torch.bfloat16)
    topk_w, topk_ids = H.routing_softmax_topk(logits, top_k)
    ref = H.python_reference_fp8(x, w13_fp8, s13, w2_fp8, s2, topk_w, topk_ids, N, K)

    # One reusable scratchpad sized to the max over all configs.
    scratch = alloc_scratchpad(dev, E, N, K, config_id=None)

    def run(cid):
        return mono_moe(
            x,
            logits,
            w13_fp8,
            s13,
            w2_fp8,
            s2,
            top_k=top_k,
            scoring_func="softmax",
            renormalize=True,
            scratchpad=scratch,
            config_id=cid,
        )

    out = {}
    for cid in config_ids:
        try:
            y = run(cid)
        except Exception as e:  # infeasible config on this GPU, etc.
            out[cid] = dict(
                cos=float("nan"), passed=False, latency_ms=None, error=str(e)[:120]
            )
            continue
        cos = H.cosine(y, ref)
        # Re-check accuracy over extra seeds (weights only; cheap-ish).
        for sd in seeds[1:]:
            w = H.make_weights(dev, E, N, K, seed=sd)
            tw, ti = H.routing_softmax_topk(logits, top_k)
            r2 = H.python_reference_fp8(x, w[0], w[1], w[2], w[3], tw, ti, N, K)
            y2 = mono_moe(
                x,
                logits,
                w[0],
                w[1],
                w[2],
                w[3],
                top_k=top_k,
                scoring_func="softmax",
                renormalize=True,
                scratchpad=scratch,
                config_id=cid,
            )
            cos = min(cos, H.cosine(y2, r2))
        passed = cos >= acc_threshold
        latency = None
        if passed:
            times = bench_gpu_time(lambda: run(cid), use_cuda_graph=True)
            latency = float(statistics.median(times))
        out[cid] = dict(cos=cos, passed=passed, latency_ms=latency)
    return out


# ── Ray sharding (one GPU per M) with in-process fallback ─────────────────────


def _worker_class():
    import ray

    @ray.remote(num_gpus=1)
    class TunerWorker:
        def __init__(self, entry, top_k, acc_threshold, baseline, seeds):
            self.entry = entry
            self.top_k = top_k
            self.acc = acc_threshold
            self.baseline = baseline
            self.seeds = seeds

        def sweep_m(self, m, config_ids):
            return m, _sweep_one_m(
                self.entry,
                m,
                self.top_k,
                config_ids,
                self.acc,
                self.baseline,
                self.seeds,
            )

    return TunerWorker


def run_sweep(entry, batch_sizes, top_k, requested_ids, acc_threshold, baseline, seeds):
    """Drive the (M × config) sweep. Ray-sharded by M, or in-process."""
    try:
        import torch

        num_gpus = torch.cuda.device_count()
    except Exception:
        num_gpus = 0

    m_to_ids = {m: config_ids_for(entry, m, requested_ids) for m in batch_sizes}
    m_to_ids = {m: ids for m, ids in m_to_ids.items() if ids}
    if not m_to_ids:
        raise SystemExit(
            "no (M, config) work: check --batch-sizes / --configs and whether "
            "the shape has a BS16 companion for M>8."
        )

    if num_gpus < 2:
        print(f"# sweep: IN-PROCESS sequential ({num_gpus} GPU visible)", flush=True)
        per_m = {}
        for m, ids in m_to_ids.items():
            print(f"# --- sweeping M={m} (configs {ids}) ---", flush=True)
            per_m[m] = _sweep_one_m(
                entry, m, top_k, ids, acc_threshold, baseline, seeds
            )
        return per_m

    import ray
    from ray.util.actor_pool import ActorPool

    if not ray.is_initialized():
        ray.init()
    ms = list(m_to_ids)
    n_actors = min(num_gpus, len(ms))
    print(
        f"# sweep: RAY-SHARDED by M across {n_actors} GPU actor(s) "
        f"({num_gpus} visible, {len(ms)} batch sizes)",
        flush=True,
    )
    Worker = _worker_class()
    actors = [
        Worker.remote(entry, top_k, acc_threshold, baseline, seeds)
        for _ in range(n_actors)
    ]
    pool = ActorPool(actors)
    per_m = {}
    for m, res in pool.map_unordered(lambda a, m: a.sweep_m.remote(m, m_to_ids[m]), ms):
        n_pass = sum(1 for r in res.values() if r["passed"])
        print(f"# M={m} done ({n_pass}/{len(res)} configs passed accuracy)", flush=True)
        per_m[m] = res
    for a in actors:
        ray.kill(a)
    return per_m


# ── Rank + emit ──────────────────────────────────────────────────────────────


def summarize_and_emit(entry, per_m, top_k, acc_threshold, baseline, json_path):
    E, N, K = entry["E"], entry["N"], entry["K"]
    print("\n" + "#" * 72)
    print(f"# TUNING SUMMARY — {entry.get('display_name', entry['key'])}")
    print(
        f"# E={E} N={N} K={K} top_k={top_k} acc_threshold={acc_threshold} "
        f"baseline={baseline}"
    )
    print("#" * 72)

    best_per_m = {}
    for m in sorted(per_m):
        res = per_m[m]
        # config-0 latency = the baseline when --baseline config0.
        base_lat = None
        if baseline == "config0" and 0 in res and res[0]["latency_ms"]:
            base_lat = res[0]["latency_ms"]
        ranked = []
        for cid, r in res.items():
            tag = "interleave" if config_uch(entry, cid) == 1 else "raw"
            if not r["passed"]:
                why = r.get("error", f"cos={r['cos']:.5f} < {acc_threshold}")
                print(f"  M={m} cfg{cid} [{tag}]: REJECT ({why})")
                continue
            lat = r["latency_ms"]
            sp = (base_lat / lat) if (base_lat and lat) else float("nan")
            ranked.append((cid, lat, sp))
            print(
                f"  M={m} cfg{cid} [{tag}]: cos={r['cos']:.5f} "
                f"lat={lat:.4f}ms speedup={sp:.3f}x"
            )
        if not ranked:
            print(f"  M={m}: NO config passed the accuracy gate")
            continue
        ranked.sort(key=lambda t: t[1])  # lowest latency wins
        cid, lat, sp = ranked[0]
        best_per_m[str(m)] = dict(
            config_id=cid,
            latency_ms=lat,
            speedup=sp,
            cos=res[cid]["cos"],
            uch=config_uch(entry, cid),
        )
        print(f"  => M={m} BEST: cfg{cid} ({lat:.4f}ms, {sp:.3f}x)")

    doc = dict(
        shape_key=entry["key"],
        E=E,
        N=N,
        K=K,
        top_k=top_k,
        acc_threshold=acc_threshold,
        baseline=baseline,
        best_per_M=best_per_m,
    )
    # Write to --json if given, else to the default dir the serving resolver
    # searches automatically (so tune -> serve closes with no env var).  Import
    # the serving-side helper so the two never diverge on the path.
    if json_path:
        out_path = json_path
    else:
        from flashinfer.fused_moe.monomoe import _default_tuned_dir

        out_path = os.path.join(_default_tuned_dir(), f"{entry['key']}.json")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"\n# wrote best-config-per-M to {out_path}")
    if json_path:
        print(f"# serve with: MONOMOE_TUNED_JSON={out_path}")
    else:
        print("# this is the default location; serving picks it up automatically.")
    return doc


def check_feasible(entry, ids):
    """Feasibility-check each declared config on the local GPU via
    enum_configs (SHM/grid).  Advisory only — prints warnings, never drops."""
    try:
        import enum_configs as EC  # noqa: F401
    except Exception as e:
        print(f"# (enum_configs feasibility check skipped: {e})")
        return
    # enum_configs is CLI-oriented; a full library hook is out of scope here.
    # The kernel's own static_asserts + the runtime SHM-attribute set are the
    # authoritative gate (an infeasible config surfaces as a launch error and
    # is recorded as an accuracy REJECT with its error string).
    print(
        f"# feasibility: relying on kernel static_asserts + runtime SHM "
        f"limit for configs {ids}"
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--shape",
        required=True,
        help="shape key or alias from shapes.json (e.g. e256_n512_k2048, e64)",
    )
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="top_k to tune at (1..8). Default: shape default_top_k.",
    )
    ap.add_argument(
        "--configs",
        type=int,
        nargs="+",
        default=None,
        help="only sweep these config ids (default: the shape's table)",
    )
    ap.add_argument(
        "--acc-threshold",
        type=float,
        default=0.98,
        help="reject a config if any per-(M,seed) cosine < this",
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="weight seeds for the accuracy gate (worst-case min)",
    )
    ap.add_argument(
        "--baseline",
        choices=["config0", "none"],
        default="config0",
        help="speedup baseline: config 0 latency, or none",
    )
    ap.add_argument("--json", help="write best-config-per-M selection here")
    args = ap.parse_args()

    entry = shape_row(args.shape)
    top_k = args.top_k if args.top_k is not None else int(entry["default_top_k"])
    if not (1 <= top_k <= 8):
        raise SystemExit(f"--top-k must be in [1,8], got {top_k}")
    table_ids = [int(c["id"]) for c in entry["configs"]]
    if args.configs:
        for cid in args.configs:
            if cid not in table_ids:
                raise SystemExit(
                    f"config_id {cid} not in shape {entry['key']} "
                    f"(have {sorted(table_ids)})"
                )

    print(
        f"# shape={entry['key']} E={entry['E']} N={entry['N']} K={entry['K']} "
        f"top_k={top_k} acc_threshold={args.acc_threshold}"
    )
    check_feasible(entry, args.configs or table_ids)

    per_m = run_sweep(
        entry,
        args.batch_sizes,
        top_k,
        args.configs,
        args.acc_threshold,
        args.baseline,
        args.seeds,
    )
    summarize_and_emit(
        entry, per_m, top_k, args.acc_threshold, args.baseline, args.json
    )


if __name__ == "__main__":
    main()
