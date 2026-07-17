#ifndef PTX_UTILS_H
#define PTX_UTILS_H

#pragma once

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace monomoe {

// ── cp.async scalar helpers (sm_80+) ──────────────────────────────────────
//
// Non-blocking 4-byte global→shared copies for small latency-sensitive
// prefetches (e.g. the per-expert block-scale tile).  Issue with
// `cp_async_cg_4`, checkpoint with `cp_async_commit_group`, drain with
// `cp_async_wait_group<N>()` before reading the destination.

__device__ static inline void cp_async_cg_4(void* smem_dst, const void* gmem_src) {
  const std::uint32_t smem_addr = static_cast<std::uint32_t>(__cvta_generic_to_shared(smem_dst));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem_addr), "l"(gmem_src));
}

__device__ static inline void cp_async_commit_group() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <std::uint32_t N>
__device__ static inline void cp_async_wait_group() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory");
}

// ── Hopper WGMMA (sm_90a) helpers ─────────────────────────────────────────
//
// Wrappers for `wgmma.mma_async` with fp8 (e4m3) operands and fp32
// accumulators.  Key semantics vs `mma.sync`:
//
//   * Issued by a full warpgroup (4 warps); every thread must execute it.
//   * A and B live in shared memory, addressed by 64-bit matrix
//     descriptors (base address + leading/stride byte offsets + swizzle).
//   * The accumulator D stays in registers, distributed across the WG.
//   * Asynchronous: `commit_group` checkpoints, `wait_group<N>` waits for
//     all but the last N groups, and `wgmma_fence` orders prior register
//     writes before the SHM reads of subsequent WGMMAs.
//
// Reference: PTX ISA §9.7.15 (Asynchronous Warpgroup MMA).

/**
 * @brief Build a WGMMA 64-bit shared-memory matrix descriptor.
 *
 * Bit layout (LSB first): [13:0] addr>>4, [29:16] LBO>>4, [45:32] SBO>>4,
 * [63:62] swizzle mode (0=none, 1=128B, 2=64B, 3=32B).  LBO is the byte
 * offset between consecutive 8-row × 16-byte core matrices along the
 * leading dimension; SBO along the stride dimension.  All three must be
 * 16-byte aligned (hence the >>4).
 */
__device__ static __forceinline__ std::uint64_t make_wgmma_desc(const void* addr,
                                                                std::uint64_t leading_byte_offset,
                                                                std::uint64_t stride_byte_offset,
                                                                std::uint32_t swizzle_mode = 0) {
  std::uint64_t shm_addr;
  asm volatile("cvta.to.shared.u64 %0, %1;\n"
               : "=l"(shm_addr)
               : "l"(reinterpret_cast<std::uint64_t>(addr)));

  std::uint64_t desc = 0;
  desc |= (shm_addr >> 4) & 0x3FFFULL;
  desc |= ((leading_byte_offset >> 4) & 0x3FFFULL) << 16;
  desc |= ((stride_byte_offset >> 4) & 0x3FFFULL) << 32;
  desc |= (static_cast<std::uint64_t>(swizzle_mode) & 0x3ULL) << 62;
  return desc;
}

/** Order prior register writes before SHM reads by subsequent WGMMAs. */
__device__ static __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

/** Close the currently-outstanding WGMMA group. */
__device__ static __forceinline__ void wgmma_commit_group() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

/**
 * @brief Wait until at most N WGMMA groups are still outstanding.
 *
 * N must be a PTX integer literal, hence explicit specializations.
 */
template <std::uint32_t N>
__device__ __forceinline__ void wgmma_wait_group();

template <>
__device__ __forceinline__ void wgmma_wait_group<0>() {
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}
template <>
__device__ __forceinline__ void wgmma_wait_group<1>() {
  asm volatile("wgmma.wait_group.sync.aligned 1;\n" ::: "memory");
}
template <>
__device__ __forceinline__ void wgmma_wait_group<2>() {
  asm volatile("wgmma.wait_group.sync.aligned 2;\n" ::: "memory");
}

/**
 * @brief wgmma.mma_async m64n8k32 fp8×fp8 → fp32, SHM A and SHM B.
 *
 * Accumulator layout per thread (lane l, warp w in the WG), same mapping as
 * mma.sync m16n8k32 replicated 4× across the warps:
 *   d[0]: row = w*16 + l/4 + 0,  col = (l%4)*2 + 0
 *   d[1]: row = w*16 + l/4 + 0,  col = (l%4)*2 + 1
 *   d[2]: row = w*16 + l/4 + 8,  col = (l%4)*2 + 0
 *   d[3]: row = w*16 + l/4 + 8,  col = (l%4)*2 + 1
 *
 * Hard-codes scale_D = 1 (accumulate) and scaleA = scaleB = +1.
 */
__device__ static __forceinline__ void wgmma_m64n8k32_e4m3_e4m3_f32(std::uint64_t desc_a,
                                                                    std::uint64_t desc_b, float& d0,
                                                                    float& d1, float& d2,
                                                                    float& d3) {
  constexpr std::uint32_t scale_D = 1;
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %6, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3}, %4, %5, p, %7, %8;\n"
      "}\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
      : "l"(desc_a), "l"(desc_b), "r"(scale_D), "n"(1), "n"(1));
}

/**
 * @brief wgmma.mma_async m64n16k32 fp8×fp8 → fp32, SHM A and SHM B.
 *
 * BS16 up/down-projection shape: one instruction consumes all 16 token
 * columns per K-substep instead of two stacked m64n8k32 issues.
 *
 * Accumulator layout (per thread in the warpgroup, for N=16):
 *   d[0..7] — 8 fp32 values.  The mapping is the m64n8k32 fragment
 *   replicated for the second 8-column quadrant (N=[8,16)):
 *     d[0]: row = w*16 + l/4 + 0,  col = (l%4)*2 + 0
 *     d[1]: row = w*16 + l/4 + 0,  col = (l%4)*2 + 1
 *     d[2]: row = w*16 + l/4 + 8,  col = (l%4)*2 + 0
 *     d[3]: row = w*16 + l/4 + 8,  col = (l%4)*2 + 1
 *     d[4]: row = w*16 + l/4 + 0,  col = 8 + (l%4)*2 + 0
 *     d[5]: row = w*16 + l/4 + 0,  col = 8 + (l%4)*2 + 1
 *     d[6]: row = w*16 + l/4 + 8,  col = 8 + (l%4)*2 + 0
 *     d[7]: row = w*16 + l/4 + 8,  col = 8 + (l%4)*2 + 1
 *   (64 M rows × 16 N columns / 128 lanes = 8 fp32 elements per lane.)
 *
 * Hard-codes scale_D = 1 (accumulate) and scaleA = scaleB = +1, matching
 * wgmma_m64n8k32_e4m3_e4m3_f32 — the only modes this kernel uses.
 */
__device__ static __forceinline__ void wgmma_m64n16k32_e4m3_e4m3_f32(
    std::uint64_t desc_a, std::uint64_t desc_b, float& d0, float& d1, float& d2, float& d3,
    float& d4, float& d5, float& d6, float& d7) {
  constexpr std::uint32_t scale_D = 1;
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %10, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
      "{%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, p, %11, %12;\n"
      "}\n"
      : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3), "+f"(d4), "+f"(d5), "+f"(d6), "+f"(d7)
      : "l"(desc_a), "l"(desc_b), "r"(scale_D), "n"(1), "n"(1));
}

// ── Hopper mbarrier helpers (sm_90a) ──────────────────────────────────────
//
// Wrappers around the `mbarrier.*` family used to gate TMA transfers.  An
// mbarrier is a 64-bit, 16-byte-aligned SHM object with two counters:
//
//   1. Arrival counter   — set by `mbarrier.init`, decremented by arrives.
//   2. Transaction bytes — set by `arrive.expect_tx`, decremented by the
//      TMA engine as bytes land in SHM.
//
// The barrier completes (flips its parity bit) when both reach zero.
// Consumers spin on `try_wait.parity` with a self-cycling parity register,
// so no explicit reset is needed between uses.
//
// Reference: PTX ISA §9.7.12; CUDA Hopper Tuning Guide §1.4.1.2.

/** Convert a generic SHM pointer to a 32-bit shared state-space address. */
__device__ static __forceinline__ std::uint32_t cvta_to_shared_u32(const void* ptr) {
  std::uint64_t shm_u64;
  asm volatile("cvta.to.shared.u64 %0, %1;\n"
               : "=l"(shm_u64)
               : "l"(reinterpret_cast<std::uint64_t>(ptr)));
  return static_cast<std::uint32_t>(shm_u64);
}

/**
 * @brief Initialize an mbarrier in SHM.
 *
 * Must run exactly once per barrier before any arrive/wait, and must be
 * followed by `fence_mbarrier_init_release_cluster()` before any remote
 * arrival or TMA issue targeting the barrier.
 */
__device__ static __forceinline__ void mbarrier_init(std::uint64_t* bar,
                                                     std::uint32_t arrival_count) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
  std::uint32_t bar_addr = cvta_to_shared_u32(bar);
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_addr), "r"(arrival_count));
#else
  (void)bar;
  (void)arrival_count;
  asm volatile("trap;");
#endif
}

/** Publish prior `mbarrier.init` writes to cluster-visible arrivals. */
__device__ static __forceinline__ void fence_mbarrier_init_release_cluster() {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
#else
  asm volatile("trap;");
#endif
}

/**
 * @brief Arrive on an mbarrier and set its transaction-bytes counter.
 *
 * The caller must issue the TMA(s) targeting this barrier AFTER this call;
 * `tx_bytes` must cover the total bytes of every TMA pointing at it.
 */
__device__ static __forceinline__ void mbarrier_arrive_expect_tx(std::uint64_t* bar,
                                                                 std::uint32_t tx_bytes) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
  std::uint32_t bar_addr = cvta_to_shared_u32(bar);
  [[maybe_unused]] std::uint64_t state;
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 %0, [%1], %2;\n"
               : "=l"(state)
               : "r"(bar_addr), "r"(tx_bytes)
               : "memory");
#else
  (void)bar;
  (void)tx_bytes;
  asm volatile("trap;");
#endif
}

/**
 * @brief Plain arrive on an mbarrier (no transaction bytes).
 *
 * Consumer-side "slot empty" signal for producer/consumer pipelines: each
 * consuming warp arrives once per phase after its last read of the guarded
 * buffer; the producer waits the phase via mbarrier_try_wait_parity before
 * overwriting.  The arrive carries release semantics for the consumer's
 * prior SHM reads.
 */
__device__ static __forceinline__ void mbarrier_arrive(std::uint64_t* bar) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
  std::uint32_t bar_addr = cvta_to_shared_u32(bar);
  [[maybe_unused]] std::uint64_t state;
  asm volatile("mbarrier.arrive.shared::cta.b64 %0, [%1];\n"
               : "=l"(state)
               : "r"(bar_addr)
               : "memory");
#else
  (void)bar;
  asm volatile("trap;");
#endif
}

/**
 * @brief Non-blocking parity-based wait on an mbarrier.
 *
 * Returns true iff the barrier completed for the expected phase.  Typical
 * use:
 *   while (!mbarrier_try_wait_parity(bar, parity)) {}
 *   parity ^= 1;  // flip for the next use of the same slot
 */
__device__ static __forceinline__ bool mbarrier_try_wait_parity(std::uint64_t* bar,
                                                                std::uint32_t parity) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
  std::uint32_t bar_addr = cvta_to_shared_u32(bar);
  std::uint32_t done;
  asm volatile(
      "{\n"
      ".reg .pred P;\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2;\n"
      "selp.u32 %0, 1, 0, P;\n"
      "}\n"
      : "=r"(done)
      : "r"(bar_addr), "r"(parity)
      : "memory");
  return done != 0;
#else
  (void)bar;
  (void)parity;
  asm volatile("trap;");
  return false;
#endif
}

// ── Hopper TMA bulk-tensor copy (sm_90a) ──────────────────────────────────

/**
 * @brief Issue one 2D TMA bulk-tensor load, tracked by an mbarrier.
 *
 * The TMA engine copies one `boxDim` tile of the tensor described by
 * `desc` (a `__grid_constant__ CUtensorMap` kernel parameter — parameter
 * space is coherent with all threads on SM90+) into SHM, decrementing the
 * barrier's transaction-bytes counter as bytes land.  Returns immediately.
 *
 * Caller contract:
 *   - Issued by exactly one thread in the block (the TMA launcher).
 *   - `bar_smem` must be pre-armed with `mbarrier_arrive_expect_tx`
 *     covering the total byte count of every TMA issue targeting it.
 *   - Coordinate order matches `cuTensorMapEncodeTiled`: coord0 = innermost
 *     (fastest) global axis, coord1 = outer axis.
 */
__device__ static __forceinline__ void tma_load_2d(CUtensorMap const& desc, std::uint32_t coord0,
                                                   std::uint32_t coord1, void* dst_smem,
                                                   std::uint64_t* bar_smem) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900)
  std::uint32_t dst_addr = cvta_to_shared_u32(dst_smem);
  std::uint32_t bar_addr = cvta_to_shared_u32(bar_smem);
  std::uint64_t desc_addr = reinterpret_cast<std::uint64_t>(&desc);
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile"
      ".mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];\n"
      :
      : "r"(dst_addr), "l"(desc_addr), "r"(coord0), "r"(coord1), "r"(bar_addr)
      : "memory");
#else
  (void)desc;
  (void)coord0;
  (void)coord1;
  (void)dst_smem;
  (void)bar_smem;
  asm volatile("trap;");
#endif
}

}  // namespace monomoe

#endif
