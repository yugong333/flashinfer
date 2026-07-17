/*
 * TVM-FFI binding for the MonoMoe kernel.
 *
 * Exports:
 *   - monomoe_topk             top-K MoE entry point (runtime config_id)
 *   - monomoe_scratchpad_size  scratchpad size (bytes) for one config_id
 *   - monomoe_scratchpad_size_max  scratchpad size covering ALL configs
 *
 * ONE SHAPE PER MODULE.  The JIT layer compiles this TU once per registered
 * (E, N, K) shape, passing exactly one `-DMONOMOE_SHAPE_E{E}_N{N}_K{K}`.  The
 * generated `configs_generated.inc` binds the `MONO_ACTIVE_*` macros (Base
 * type, config X-macro table, BS16 companion) to that one shape; every other
 * shape's blocks are `#if`'d out, so the binary only instantiates the active
 * shape's configs.  This preserves "no rebuild between configs" (all of a
 * shape's configs are baked in, switched by a runtime `config_id`) without a
 * monolithic all-shapes binary.
 *
 * The implementation (and the whole `src/moe.cuh` unity build) lives in
 * `monomoe_wrapper.cuh`, which is `#include`d here so the launcher template is
 * in this translation unit.
 *
 * Copyright (c) 2025 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0.
 */

#include <type_traits>

#include "generated/configs_generated.inc"
#include "monomoe_wrapper.cuh"

// Declaration of the launcher function template (defined and instantiated
// per Dims variant at the bottom of monomoe_wrapper.cuh via the config table).
template <class Dims>
void monomoe_topk_launcher(TensorView activations_in, TensorView router_logits,
                           TensorView expert_weights_up, TensorView expert_scales_up,
                           TensorView expert_weights_down, TensorView expert_scales_down,
                           TensorView activations_out, TensorView scratchpad, int64_t top_k,
                           int64_t scoring_func, bool renormalize,
                           ffi::Optional<TensorView> expert_bias, double routed_scaling_factor);

namespace {

// Maps one config row to its Dims type: config 0 is the bare Base (byte-
// identical to the pre-tuning shipped kernel), any other id is a DimsTunable
// clone of Base with that row's knobs.  std::conditional_t names both types
// but only the selected one is passed to the launcher, so config 0 never
// instantiates a DimsTunable — the config-0 identity is structural, not an
// invariant that could drift.
template <class Base, int64_t ID, uint32_t GRID, uint32_t DCT, uint32_t KUP, uint32_t KDN,
          uint32_t SLOTS, uint32_t UCH, uint32_t DPD>
using DimsForConfig =
    std::conditional_t<ID == 0, Base,
                       ::monomoe::DimsTunable<Base, GRID, DCT, KUP, KDN, SLOTS, UCH, DPD>>;

// Does the (E, N, K) weight-shape signature match this Dims variant?  The
// kernel is hard-specialized per shape, so the runtime tensors must match
// exactly before launch (otherwise the kernel reads/writes out of bounds).
// The token cap is checked separately so BS8 vs BS16 can share one shape.
template <class Dims>
bool shape_matches(const TensorView& activations_in, const TensorView& expert_weights_up,
                   const TensorView& expert_weights_down) {
  return activations_in.ndim() == 2 && activations_in.size(1) == Dims::K &&
         expert_weights_up.ndim() == 3 && expert_weights_up.size(0) == Dims::NUM_EXPERTS &&
         expert_weights_up.size(1) == 2 * Dims::N && expert_weights_up.size(2) == Dims::K &&
         expert_weights_down.ndim() == 3 && expert_weights_down.size(0) == Dims::NUM_EXPERTS &&
         expert_weights_down.size(1) == Dims::K && expert_weights_down.size(2) == Dims::N;
}

}  // namespace

namespace {

// TEMP_FP8_OFFSET regression anchor (design doc §4): the down-activation TMA
// descriptor addresses `spec->temp_fp8` as `scratchpad_ptr + TEMP_FP8_OFFSET`,
// so that constant must equal `offsetof(MoEGemmSpec<Dims>, temp_fp8)` for
// EVERY instantiated variant (the layout is Dims-dependent, so each config
// must be checked, not just Base).  Wrapped in a helper so the X-macro can
// pass Dims with commas without tripping the function-like `offsetof` macro.
template <class Dims>
constexpr bool temp_fp8_offset_ok() {
  return offsetof(::monomoe::MoEGemmSpec<Dims>, temp_fp8) ==
         ::monomoe::MoEGemmSpec<Dims>::TEMP_FP8_OFFSET;
}

}  // namespace

#define X(ID, GRID, DCT, KUP, KDN, SLOTS, UCH, DPD)                                                \
  static_assert(temp_fp8_offset_ok<                                                                \
                    DimsForConfig<MONO_ACTIVE_BASE, ID, GRID, DCT, KUP, KDN, SLOTS, UCH, DPD>>(),  \
                "TEMP_FP8_OFFSET must match offsetof(MoEGemmSpec<Dims>, temp_fp8); new fields go " \
                "at the tail of MoEGemmSpec<Dims>.");
MONO_ACTIVE_CONFIGS(X)
#undef X

// Top-K MoE entry point: dispatch on the runtime tensor shape, token count,
// and config_id to the matching hard-specialized Dims variant.  config_id < 0
// (or an id absent from this shape's table) selects config 0 (the shipped
// default).  BS8 serves M <= 8; the BS16 companion serves 8 < M <= 16.
void monomoe_topk(TensorView activations_in, TensorView router_logits, TensorView expert_weights_up,
                  TensorView expert_scales_up, TensorView expert_weights_down,
                  TensorView expert_scales_down, TensorView activations_out, TensorView scratchpad,
                  int64_t top_k, int64_t scoring_func, bool renormalize,
                  ffi::Optional<TensorView> expert_bias, double routed_scaling_factor,
                  int64_t config_id) {
  using Base = MONO_ACTIVE_BASE;
  const int64_t m = activations_in.ndim() == 2 ? activations_in.size(0) : -1;

  if (!shape_matches<Base>(activations_in, expert_weights_up, expert_weights_down)) {
    TVM_FFI_ICHECK(false) << "monomoe_topk: this module is built for E=" << Base::NUM_EXPERTS
                          << ", N=" << Base::N << ", K=" << Base::K << " but got activations_in=["
                          << m << ", " << (activations_in.ndim() == 2 ? activations_in.size(1) : -1)
                          << "], expert_weights_up.dim0="
                          << (expert_weights_up.ndim() == 3 ? expert_weights_up.size(0) : -1)
                          << ".";
    return;
  }

  // Common launcher call (identical args for BS8 and BS16).
#define MONO_LAUNCH(DIMS)                                                                         \
  monomoe_topk_launcher<DIMS>(activations_in, router_logits, expert_weights_up, expert_scales_up, \
                              expert_weights_down, expert_scales_down, activations_out,           \
                              scratchpad, top_k, scoring_func, renormalize, expert_bias,          \
                              routed_scaling_factor)

  if (m >= 1 && m <= static_cast<int64_t>(Base::BS)) {
    // BS8 arm: any config in this shape's table is valid.  The local `using`
    // shields the comma-bearing DimsForConfig<...> from the MONO_LAUNCH macro.
#define X(ID, GRID, DCT, KUP, KDN, SLOTS, UCH, DPD)                                \
  if (config_id == (ID)) {                                                         \
    using CfgDims = DimsForConfig<Base, ID, GRID, DCT, KUP, KDN, SLOTS, UCH, DPD>; \
    MONO_LAUNCH(CfgDims);                                                          \
    return;                                                                        \
  }
    MONO_ACTIVE_CONFIGS(X)
#undef X
    MONO_LAUNCH(Base);  // config_id < 0 or unknown -> config 0 default
    return;
  }

#if MONO_ACTIVE_HAS_BS16
  using Base16 = MONO_ACTIVE_BASE_BS16;
  if (m > static_cast<int64_t>(Base::BS) && m <= static_cast<int64_t>(Base16::BS)) {
    // BS16 arm: only the configs that fit the BS16 SHM budget are baked in
    // (MONO_CONFIGS_BS16_*); a config_id absent here falls to config 0.
#define X(ID, GRID, DCT, KUP, KDN, SLOTS, UCH, DPD)                                  \
  if (config_id == (ID)) {                                                           \
    using CfgDims = DimsForConfig<Base16, ID, GRID, DCT, KUP, KDN, SLOTS, UCH, DPD>; \
    MONO_LAUNCH(CfgDims);                                                            \
    return;                                                                          \
  }
    MONO_ACTIVE_CONFIGS_BS16(X)
#undef X
    MONO_LAUNCH(Base16);  // config_id not in the BS16 set -> config 0 default
    return;
  }
#endif  // MONO_ACTIVE_HAS_BS16

#undef MONO_LAUNCH

  TVM_FFI_ICHECK(false) << "monomoe_topk: token count M=" << m
                        << " out of range for E=" << Base::NUM_EXPERTS << ", N=" << Base::N
                        << ", K=" << Base::K << " (supported 1 <= M <= "
#if MONO_ACTIVE_HAS_BS16
                        << 16
#else
                        << Base::BS
#endif
                        << ").";
}

// Scratchpad size (bytes) for a specific config_id, sized to cover both the
// BS8 and (where present) BS16 variant of that config so one buffer serves any
// token count at that config.  config_id < 0 / unknown -> config 0.
int64_t monomoe_scratchpad_size(int64_t config_id) {
  using Base = MONO_ACTIVE_BASE;
  size_t s = sizeof(::monomoe::MoEGemmSpec<Base>);  // config-0 floor
  auto take = [&s](size_t c) { s = c > s ? c : s; };
  (void)take;
#define X(ID, GRID, DCT, KUP, KDN, SLOTS, UCH, DPD)                                              \
  if (config_id == ID) {                                                                         \
    take(sizeof(                                                                                 \
        ::monomoe::MoEGemmSpec<DimsForConfig<Base, ID, GRID, DCT, KUP, KDN, SLOTS, UCH, DPD>>)); \
  }
  MONO_ACTIVE_CONFIGS(X)
#undef X
#if MONO_ACTIVE_HAS_BS16
  using Base16 = MONO_ACTIVE_BASE_BS16;
#define X(ID, GRID, DCT, KUP, KDN, SLOTS, UCH, DPD)                                                \
  if (config_id == ID) {                                                                           \
    take(sizeof(                                                                                   \
        ::monomoe::MoEGemmSpec<DimsForConfig<Base16, ID, GRID, DCT, KUP, KDN, SLOTS, UCH, DPD>>)); \
  }
  MONO_ACTIVE_CONFIGS_BS16(X)
#undef X
#endif  // MONO_ACTIVE_HAS_BS16
  return static_cast<int64_t>(s);
}

// Scratchpad size (bytes) covering EVERY config of this shape (BS8 + BS16), so
// one buffer can be reused across config switches without reallocation.
int64_t monomoe_scratchpad_size_max() {
  using Base = MONO_ACTIVE_BASE;
  size_t s = 0;
  auto take = [&s](size_t c) { s = c > s ? c : s; };
#define X(ID, GRID, DCT, KUP, KDN, SLOTS, UCH, DPD) \
  take(sizeof(                                      \
      ::monomoe::MoEGemmSpec<DimsForConfig<Base, ID, GRID, DCT, KUP, KDN, SLOTS, UCH, DPD>>));
  MONO_ACTIVE_CONFIGS(X)
#undef X
#if MONO_ACTIVE_HAS_BS16
  using Base16 = MONO_ACTIVE_BASE_BS16;
#define X(ID, GRID, DCT, KUP, KDN, SLOTS, UCH, DPD) \
  take(sizeof(                                      \
      ::monomoe::MoEGemmSpec<DimsForConfig<Base16, ID, GRID, DCT, KUP, KDN, SLOTS, UCH, DPD>>));
  MONO_ACTIVE_CONFIGS_BS16(X)
#undef X
#endif  // MONO_ACTIVE_HAS_BS16
  return static_cast<int64_t>(s);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(monomoe_topk, monomoe_topk);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(monomoe_scratchpad_size, monomoe_scratchpad_size);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(monomoe_scratchpad_size_max, monomoe_scratchpad_size_max);
