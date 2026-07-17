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
"""

import functools
import os
import shutil
from pathlib import Path

from . import env as jit_env
from .core import gen_jit_spec, logger, sm90a_nvcc_flags


def _get_monomoe_csrc_dir() -> Path:
    """Get the path to the MonoMoe kernel CUDA source directory.

    Handles both the installed package (data/csrc/fused_moe/monomoe) and a
    development checkout (../../csrc/fused_moe/monomoe relative to this file).
    """
    standard_path = jit_env.FLASHINFER_CSRC_DIR / "fused_moe" / "monomoe"
    if standard_path.exists():
        return standard_path

    dev_path = Path(__file__).parent.parent.parent / "csrc" / "fused_moe" / "monomoe"
    if dev_path.exists():
        return dev_path

    raise FileNotFoundError(
        f"MonoMoe kernel CUDA sources not found. Checked:\n"
        f"  - {standard_path}\n"
        f"  - {dev_path}\n"
        f"Please ensure the csrc/fused_moe/monomoe/ directory exists."
    )


def get_monomoe_uri(E: int, N: int, K: int) -> str:
    """Unique identifier for a monomoe module, keyed on the (E, N, K) shape.

    One module is JIT-compiled per registered shape (each bakes ALL of that
    shape's tunable configs, dispatched at runtime by ``config_id``), so the
    URI — and hence the build dir and cached ``.so`` — must be shape-specific.
    """
    return f"monomoe_E{E}_N{N}_K{K}"


# Files actually fed to nvcc as compilation units.  Everything else in the
# monomoe tree is `#include`d into one of these two TUs (whole-program unity
# build), so listing it here would cause duplicate-symbol link errors.
#
#   - monomoe_binding.cu   #includes monomoe_wrapper.cuh -> src/moe.cuh (which in
#                          turn #includes the up/down/routing/scale .cuh files)
#   - src/moe_tma.cu       standalone host TU: definitions of the create_*_tma_desc
#                          factories (declared in src/moe_tma.h, never #included)
_SOURCE_FILES = [
    "monomoe_binding.cu",
    "src/moe_tma.cu",
]

# Header / include-only sources copied alongside the compiled TUs so the
# `#include` graph resolves inside the gen directory.  These are NOT compiled.
_INCLUDE_FILES = [
    "monomoe_wrapper.cuh",
    "src/moe.cuh",
    "src/moe_up_projection.cuh",
    "src/moe_down_projection.cuh",
    "src/moe_routing.cuh",
    "src/moe_scale_inputs.cuh",
    "src/moe_interface.h",
    "src/moe_internal.h",
    "src/moe_tma.h",
    "src/ptx_utils.h",
    "generated/dims_generated.inc",
    "generated/configs_generated.inc",
]


@functools.cache
def gen_monomoe_module(E: int, N: int, K: int):
    """
    Generate the JIT compilation spec for one MonoMoe shape.

    Compiles the single-kernel top-K MoE pipeline (routing, up-projection,
    SiLU, down-projection, reduction) for the block-FP8 WGMMA+TMA path,
    specialized to the ``(E, N, K)`` shape.  The module bakes ALL of that
    shape's tunable configs (from ``configs_generated.inc``) and dispatches on
    a runtime ``config_id``; the ``-DMONOMOE_SHAPE_E{E}_N{N}_K{K}`` define
    selects which shape's blocks compile, so only this shape's kernels are
    instantiated.  Hopper (SM90a) only — the kernel uses `wgmma.mma_async` and
    TMA, which require SM90.

    Args:
        E: number of experts.
        N: intermediate size per half (gate/up each have N rows; 2*N total).
        K: hidden size.

    Returns:
        JitSpec that can be built and loaded.
    """
    csrc_dir = _get_monomoe_csrc_dir()
    uri = get_monomoe_uri(E, N, K)

    gen_directory = jit_env.FLASHINFER_GEN_SRC_DIR / uri
    # `src/` must exist so the `#include "src/..."` paths resolve and so the
    # `src/*.cuh` include-only files land where moe.cuh expects them.
    os.makedirs(gen_directory / "src", exist_ok=True)

    def _copy(rel_name: str) -> Path:
        src_path = csrc_dir / rel_name
        if not src_path.exists():
            raise FileNotFoundError(f"monomoe source file not found: {src_path}")
        dest_path = gen_directory / rel_name
        os.makedirs(dest_path.parent, exist_ok=True)
        shutil.copy(src_path, dest_path)
        return dest_path

    sources = [_copy(fname) for fname in _SOURCE_FILES]
    for fname in _INCLUDE_FILES:
        _copy(fname)

    # Hopper (SM90a) ONLY: the kernel uses wgmma.mma_async + TMA and is
    # hard-specialized to a single SM90a shape — it cannot target any
    # other architecture.  Use the `sm90a_nvcc_flags` constant directly
    # (matching gen_cutlass_fused_moe_sm90_module / gen_gemm_sm90_module)
    # rather than the multi-arch `get_nvcc_flags_list` path: both resolve
    # to `compute_90a,sm_90a` today, but the constant is the honest,
    # robust choice for a kernel that is SM90a-only by construction (it
    # always emits the `a` suffix and never depends on the detected /
    # FLASHINFER_CUDA_ARCH_LIST arch set).
    spec = gen_jit_spec(
        name=uri,
        sources=sources,
        extra_cuda_cflags=sm90a_nvcc_flags
        + [
            "-DFLASHINFER_ENABLE_BF16",
            "-DFLASHINFER_ENABLE_FP8_E4M3",
            # Selects the active shape in configs_generated.inc so only this
            # shape's DimsTunable configs instantiate (bounded compile).
            f"-DMONOMOE_SHAPE_E{E}_N{N}_K{K}",
        ],
        extra_include_paths=[
            str(gen_directory),
            str(gen_directory / "src"),
            str(jit_env.FLASHINFER_INCLUDE_DIR),
            str(jit_env.FLASHINFER_CSRC_DIR),
            # CuTe SM90 arch headers: ptx_utils.h delegates the WGMMA control
            # ops (fence/commit/wait) to cute::warpgroup_* primitives.
            *(str(p) for p in jit_env.CUTLASS_INCLUDE_DIRS),
        ],
    )

    logger.info(f"Generated monomoe JIT spec: {spec.name}")
    return spec


@functools.cache
def load_monomoe_module(E: int, N: int, K: int):
    """
    Build and load the MonoMoe CUDA extension for one ``(E, N, K)`` shape via
    FlashInfer's JIT system.

    Returns the loaded module exposing `monomoe_topk`,
    `monomoe_scratchpad_size`, and `monomoe_scratchpad_size_max`.
    """
    spec = gen_monomoe_module(E, N, K)
    module = spec.build_and_load()
    logger.info(f"monomoe module loaded successfully (E={E}, N={N}, K={K})")
    return module
