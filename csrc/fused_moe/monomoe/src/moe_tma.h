#ifndef MOE_TMA_H
#define MOE_TMA_H

#pragma once

// TMA (`CUtensorMap`) descriptor factories + device-side load helpers for
// the BS8 TMA+WGMMA path.  The host factories (defined in moe_tma.cu) call
// `cuTensorMapEncodeTiled` (CUDA Driver API, 12.0+); the descriptors are
// passed to the kernel as `__grid_constant__ CUtensorMap const` parameters.
// See docs/design_docs/monomoe_kernel.md "TMA descriptors" for the descriptor table and layouts.

#include <cuda.h>

#include <cstdint>

#include "moe_interface.h"
#include "ptx_utils.h"

#if defined(CUDA_VERSION) && (CUDA_VERSION < 12000)
#error "moe_tma.h requires CUDA 12.0 or later for CUtensorMap / cuTensorMapEncodeTiled"
#endif

namespace monomoe {

/**
 * @brief Descriptor for the fp8 up-projection weight tensor `[E, 2*N, K]`.
 *
 * SWIZZLE_128B, boxDim = (128 K, 128 rows), innermost axis = K, outer axis
 * = the flattened `E * 2N` row index.  The TMA hardware applies the 8-row ×
 * 128-B core-matrix XOR swizzle at write time; the matching WGMMA A
 * descriptor uses swizzle=1, LBO=16, SBO=1024 (CUTLASS Major::K B128).
 *
 * The descriptor is layout-agnostic; only the row coordinate differs
 * between the two consumers:
 *   * interleaved (UCH==1, `tma_load_up_wgmma_tile`): the Python
 *     pre-interleave packs each 128-row WGMMA A-tile (gate+up stripes)
 *     contiguously; coord1 = e*2N + 2*base_row_up.
 *   * raw (UCH>=2, `tma_load_up_wgmma_tile_raw`): no pre-interleave; two
 *     TMAs per tile at coord1 = e*2N + row (gate) / e*2N + N + row (up).
 *
 * Raises TVM_FFI_ICHECK on encode failure.
 */
CUtensorMap create_up_weight_tma_desc(const void* weights_ptr, uint32_t num_experts, uint32_t N,
                                      uint32_t K);

/**
 * @brief Descriptor for the bf16 activation tensor `[BS, K_hidden]`.
 *
 * SWIZZLE_NONE, boxDim = (128 K, box_rows tokens), innermost axis = K.
 * Used by the Phase-1 routing-window load (`moe_load_full_bf16_input`).
 * `box_rows` = Dims::BS of the launching kernel (8 for the BS8 path, 16
 * for the BS16 path); rows past `batch_size_cap` are hardware zero-filled
 * so the armed expect_tx byte count is BS-exact regardless of the runtime
 * token count.
 */
CUtensorMap create_activations_tma_desc(const void* activations_ptr, uint32_t batch_size_cap,
                                        uint32_t K_hidden, uint32_t box_rows = 8u);

/**
 * @brief Descriptor for the fp8 down-projection weight tensor `[E, K, N]`.
 *
 * SWIZZLE_128B, boxDim = (128 N, row_box rows).  Axis ordering is flipped
 * vs the up-proj descriptor: N (the down-proj reduction dim) is innermost
 * and the outer axis is the flattened `e * K + output_row` index —
 * matching the `[E, K, N]` GM layout where each output row is a contiguous
 * N-vector.  The RAW row-major tensor is expected (no pre-interleave).
 *
 * `row_box` = rows per TMA issue: 128 (one 16 KB atom) or 256 (two stacked
 * atoms, halving the issue count when DOWN_COL_TILE = 256).  The TMA
 * boxDim hardware cap is 256/axis.
 */
CUtensorMap create_down_weight_tma_desc(const void* weights_ptr, uint32_t num_experts, uint32_t K,
                                        uint32_t N, uint32_t row_box = 128u);

/**
 * @brief Descriptor for the fp8 intermediate activations
 * `spec->temp_fp8[temp_rows, N]` consumed by the Phase-4 down-projection.
 *
 * SWIZZLE_128B, boxDim = (128 N, 8 rows); innermost axis = N, outer axis =
 * the sorted_slot row index.  Each expert's routed tokens occupy a
 * contiguous row slab (written that way by the Phase-3 epilogue), so one
 * bulk TMA per K-step fetches a whole expert's activations.
 *
 * SWZ128 precondition: N must be a multiple of 128 (the row stride feeds
 * the swizzle).  `temp_rows` = BS * MAX_TOPK (guard padding excluded).
 * `t_tile` = rows per TMA issue (CoreDims::T_TILE): 8 on the BS8 path,
 * 16 (two stacked SWZ128 atoms) on the BS16 path.
 */
CUtensorMap create_down_activation_tma_desc(const void* activations_ptr, uint32_t temp_rows,
                                            uint32_t N, uint32_t t_tile = 8u);

// ─── Device-side TMA load helpers ────────────────────────────────────────
//
// Thin wrappers over `tma_load_2d` (ptx_utils.h) that bake in each path's
// coordinate convention.  Shared caller contract for ALL helpers below:
//
//   * Call from exactly ONE thread per block — the TMA launcher (warp 8
//     lane 0, `is_tma_launcher_thread`).  The helpers do NOT gate on
//     threadIdx; duplicate callers issue duplicate TMAs and corrupt the
//     barrier's transaction-byte accounting.
//   * Pre-arm the target mbarrier exactly once with
//     `mbarrier_arrive_expect_tx(bar, total_bytes)` covering every TMA
//     issue pointing at it, BEFORE calling.
//   * SHM destinations for SWZ128 loads must be 1024-B aligned (the
//     swizzle atom alignment; satisfied by the alignas(1024) SHM layout).

/**
 * @brief One 128×128 up-weight TMA from the INTERLEAVED layout (UCH==1).
 *
 * The Python pre-interleave packs, per 64-gate-row block k:
 *   rows [128k+ 0, +32) = gate[64k     , +32)
 *   rows [128k+32, +32) =   up[64k     , +32)
 *   rows [128k+64, +32) = gate[64k + 32, +32)
 *   rows [128k+96, +32) =   up[64k + 32, +32)
 * so gate row `base_row_up` maps to interleaved row `2 * base_row_up`.
 * tx_bytes per issue: 16384.
 */
__device__ __forceinline__ void tma_load_up_wgmma_tile(
    CUtensorMap const& desc, std::uint32_t expert_id, std::uint32_t N, std::uint32_t base_row_up,
    std::uint32_t k_start, void* dest_slot_smem_ptr, std::uint64_t* bar_smem_ptr) {
  const std::uint32_t global_row = expert_id * 2u * N + 2u * base_row_up;
  tma_load_2d(desc, /*coord0=*/k_start, /*coord1=*/global_row, dest_slot_smem_ptr, bar_smem_ptr);
}

/**
 * @brief One 128×128 up-weight TMA from the RAW layout (UCH>=2).
 *
 * Fetches a contiguous 128-row tile straight from the unmodified
 * `[E, 2*N, K]` tensor: gate rows live at [0, N), up rows at [N, 2N) per
 * expert.  The same descriptor as the interleaved loader is reused — only
 * the row coordinate differs.  tx_bytes per issue: 16384.
 */
__device__ __forceinline__ void tma_load_up_wgmma_tile_raw(
    CUtensorMap const& desc, std::uint32_t expert_id, std::uint32_t N, std::uint32_t row_in_half,
    bool is_up, std::uint32_t k_start, void* dest_slot_smem_ptr, std::uint64_t* bar_smem_ptr) {
  const std::uint32_t global_row = expert_id * 2u * N + (is_up ? N : 0u) + row_in_half;
  tma_load_2d(desc, /*coord0=*/k_start, /*coord1=*/global_row, dest_slot_smem_ptr, bar_smem_ptr);
}

/**
 * @brief One 8-token × 128-K bf16 activation box (2048 B, SWIZZLE_NONE).
 *
 * Always fetches all 8 tokens starting at token 0 (activations are
 * expert-invariant), so the coordinates are (k_start, 0).
 */
__device__ __forceinline__ void tma_load_bf16_input_tile(CUtensorMap const& desc,
                                                         std::uint32_t k_start, void* dest_smem_ptr,
                                                         std::uint64_t* bar_smem_ptr) {
  tma_load_2d(desc, /*coord0=*/k_start, /*coord1=*/0u, dest_smem_ptr, bar_smem_ptr);
}

/**
 * @brief Phase-1 routing-window load: the full `[BS, HIDDEN_STATES]` BF16
 * input tile via K_BLOCKS_TOTAL back-to-back 128-K box issues.
 *
 * Multiple issues (not one big box) because `cuTensorMapEncodeTiled` caps
 * per-axis boxDim at 256 elements.  The destination is the tile-major
 * `[K_BLOCKS_TOTAL][BS][128]` SHM buffer — each issue writes a compact
 * 2 KB box into its own slab, matching the TMA's natural write order (see
 * the bf16_in_full comment in moe_internal.h).
 *
 * The caller pre-arms `bar_smem_ptr` once with the cumulative
 * tx_bytes = K_BLOCKS_TOTAL * BS * 128 * sizeof(A_element).
 */
template <typename Dims>
__device__ __forceinline__ void moe_load_full_bf16_input(
    CUtensorMap const& activations_desc,
    A_element (&dest)[Dims::HIDDEN_STATES / 128u][Dims::BS][128u], std::uint64_t* bar_smem_ptr) {
  constexpr std::uint32_t K_STEP_WGMMA = 128u;
  static_assert(Dims::HIDDEN_STATES % K_STEP_WGMMA == 0,
                "moe_load_full_bf16_input requires HIDDEN_STATES to be a "
                "multiple of 128 (the SWZ128 atom K-width).");
  constexpr std::uint32_t K_BLOCKS_TOTAL = Dims::HIDDEN_STATES / K_STEP_WGMMA;

#pragma unroll
  for (std::uint32_t kk = 0u; kk < K_BLOCKS_TOTAL; ++kk) {
    const std::uint32_t k_start = kk * K_STEP_WGMMA;
    tma_load_bf16_input_tile(activations_desc, /*k_start=*/k_start,
                             /*dest_smem_ptr=*/&dest[kk][0][0],
                             /*bar_smem_ptr=*/bar_smem_ptr);
  }
}

/**
 * @brief One 128-row down-weight TMA (16384 B, SWZ128) from the RAW
 * `[E, K, N]` tensor.
 *
 * coord0 = k_start (the starting N column — the down-proj reduction dim),
 * coord1 = expert_id * K + base_col (the starting output row).
 */
__device__ __forceinline__ void tma_load_down_wgmma_tile(CUtensorMap const& desc,
                                                         std::uint32_t expert_id, std::uint32_t K,
                                                         std::uint32_t base_col,
                                                         std::uint32_t k_start, void* dest_smem_ptr,
                                                         std::uint64_t* bar_smem_ptr) {
  const std::uint32_t global_row = expert_id * K + base_col;
  tma_load_2d(desc, /*coord0=*/k_start, /*coord1=*/global_row, dest_smem_ptr, bar_smem_ptr);
}

/**
 * @brief One bulk down-activation TMA: an 8-row × 128-N fp8 atom (1024 B,
 * SWZ128) starting at the expert's contiguous slab in temp_fp8.
 *
 * Only call when the expert's routed_count > 0 (otherwise neither arm the
 * barrier nor issue; the WGMMA tolerates the stale SHM — see the
 * routed_count == 0 comment in moe_down_projection.cu).  The box always
 * fetches the full 8 rows; rows beyond routed_count carry bytes from
 * neighboring slabs that the rank-filtered accumulate never reads.
 */
__device__ __forceinline__ void tma_load_down_wgmma_activation_bulk(CUtensorMap const& desc,
                                                                    std::uint32_t k_start,
                                                                    std::uint32_t expert_slot_start,
                                                                    void* dest_smem_ptr,
                                                                    std::uint64_t* bar_smem_ptr) {
  tma_load_2d(desc, /*coord0=*/k_start, /*coord1=*/expert_slot_start, dest_smem_ptr, bar_smem_ptr);
}

}  // namespace monomoe

#endif  // MOE_TMA_H
