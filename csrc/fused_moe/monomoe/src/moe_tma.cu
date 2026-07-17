/**
 * Host-side TMA `CUtensorMap` descriptor factories (declared in moe_tma.h).
 *
 * Pure host code: the only CUDA interaction is `cuTensorMapEncodeTiled`
 * (Driver API), which populates a 128-byte POD on the CPU.  Unlike the
 * other .cu files in this directory (which are #included into moe.cu),
 * this is a standalone TU compiled by the build system.
 *
 * FlashInfer is framework-agnostic through TVM-FFI, so error reporting
 * uses TVM_FFI_ICHECK.  This TU never touches a Tensor — only raw
 * pointers + dims.
 */

#include <cuda.h>

#include "moe_tma.h"
#include "tvm_ffi_utils.h"

namespace monomoe {

CUtensorMap create_up_weight_tma_desc(const void* weights_ptr, uint32_t num_experts, uint32_t N,
                                      uint32_t K) {
  // Zero-init so unfilled POD bytes have a defined value.
  CUtensorMap desc{};

  constexpr uint32_t kRank = 2;
  uint64_t global_dim[kRank] = {
      static_cast<uint64_t>(K),
      static_cast<uint64_t>(num_experts) * 2ULL * static_cast<uint64_t>(N),
  };
  uint64_t global_strides[kRank - 1] = {
      static_cast<uint64_t>(K),  // K bytes per row (fp8 = 1 B/element)
  };
  uint32_t box_dim[kRank] = {
      /*K-inner*/ 128u,
      /*rows   */ 128u,
  };
  uint32_t element_strides[kRank] = {1u, 1u};

  const CUresult res = cuTensorMapEncodeTiled(
      &desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, kRank,
      // The Driver API takes a non-const void*; the descriptor only reads
      // the pointer value.
      const_cast<void*>(weights_ptr), global_dim, global_strides, box_dim, element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  TVM_FFI_ICHECK(res == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for up-projection weights: CUresult="
      << static_cast<int>(res) << " (num_experts=" << num_experts << ", N=" << N << ", K=" << K
      << ")";

  return desc;
}

CUtensorMap create_activations_tma_desc(const void* activations_ptr, uint32_t batch_size_cap,
                                        uint32_t K_hidden, uint32_t box_rows) {
  TVM_FFI_ICHECK(box_rows == 8u || box_rows == 16u)
      << "create_activations_tma_desc: box_rows must be 8 (BS8) or 16 (BS16), got " << box_rows;
  CUtensorMap desc{};

  // Innermost-first ordering per cuTensorMapEncodeTiled: globalDim[0] = K,
  // globalDim[1] = batch rows; globalStrides is the byte stride between
  // rows (bf16 = 2 B/element).
  constexpr uint32_t kRank = 2;
  uint64_t global_dim[kRank] = {
      static_cast<uint64_t>(K_hidden),
      static_cast<uint64_t>(batch_size_cap),
  };
  uint64_t global_strides[kRank - 1] = {
      static_cast<uint64_t>(K_hidden) * 2ULL,
  };
  uint32_t box_dim[kRank] = {
      /*K-inner*/ 128u,
      /*tokens */ box_rows,
  };
  uint32_t element_strides[kRank] = {1u, 1u};

  const CUresult res = cuTensorMapEncodeTiled(
      &desc, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, kRank, const_cast<void*>(activations_ptr),
      global_dim, global_strides, box_dim, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  TVM_FFI_ICHECK(res == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for activations: CUresult=" << static_cast<int>(res)
      << " (batch_size_cap=" << batch_size_cap << ", K_hidden=" << K_hidden << ")";

  return desc;
}

CUtensorMap create_down_weight_tma_desc(const void* weights_ptr, uint32_t num_experts, uint32_t K,
                                        uint32_t N, uint32_t row_box) {
  TVM_FFI_ICHECK(row_box == 128u || row_box == 256u)
      << "create_down_weight_tma_desc: row_box must be 128 or 256, got " << row_box;
  TVM_FFI_ICHECK(K % row_box == 0)
      << "create_down_weight_tma_desc: K=" << K << " must be a multiple of row_box=" << row_box;

  // Innermost = N (the down-proj reduction dim), outer = flattened
  // expert_id * K + output_row; see moe_tma.h.
  CUtensorMap desc{};

  constexpr uint32_t kRank = 2;
  uint64_t global_dim[kRank] = {
      static_cast<uint64_t>(N),
      static_cast<uint64_t>(num_experts) * static_cast<uint64_t>(K),
  };
  uint64_t global_strides[kRank - 1] = {
      static_cast<uint64_t>(N),  // N bytes per row (fp8 = 1 B/element)
  };
  uint32_t box_dim[kRank] = {
      /*N-inner*/ 128u,
      /*rows   */ row_box,
  };
  uint32_t element_strides[kRank] = {1u, 1u};

  const CUresult res = cuTensorMapEncodeTiled(
      &desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, kRank, const_cast<void*>(weights_ptr), global_dim,
      global_strides, box_dim, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  TVM_FFI_ICHECK(res == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for down-projection weights: CUresult="
      << static_cast<int>(res) << " (num_experts=" << num_experts << ", K=" << K << ", N=" << N
      << ", row_box=" << row_box << ")";

  return desc;
}

CUtensorMap create_down_activation_tma_desc(const void* activations_ptr, uint32_t temp_rows,
                                            uint32_t N, uint32_t t_tile) {
  // SWZ128 preconditions: boxDim[0] * sizeof(fp8) = 128 B; the row stride
  // N must be a multiple of 128; the scratchpad base is 256-B aligned by
  // the PyTorch allocator.  t_tile is the per-issue row-box: 8 (one SWZ128
  // atom) on the BS8 path, 16 (two stacked atoms) on the BS16 path.
  TVM_FFI_ICHECK(t_tile == 8u || t_tile == 16u)
      << "create_down_activation_tma_desc: t_tile must be 8 (BS8) or 16 (BS16), got " << t_tile;
  CUtensorMap desc{};

  constexpr uint32_t kRank = 2;
  uint64_t global_dim[kRank] = {
      static_cast<uint64_t>(N),
      static_cast<uint64_t>(temp_rows),
  };
  uint64_t global_strides[kRank - 1] = {
      static_cast<uint64_t>(N),  // N bytes per row (fp8 = 1 B/element)
  };
  uint32_t box_dim[kRank] = {
      /*K-inner*/ 128u,
      /*rows   */ t_tile,
  };
  uint32_t element_strides[kRank] = {1u, 1u};

  const CUresult res = cuTensorMapEncodeTiled(
      &desc, CU_TENSOR_MAP_DATA_TYPE_UINT8, kRank, const_cast<void*>(activations_ptr), global_dim,
      global_strides, box_dim, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  TVM_FFI_ICHECK(res == CUDA_SUCCESS)
      << "cuTensorMapEncodeTiled failed for down-projection activations: CUresult="
      << static_cast<int>(res) << " (temp_rows=" << temp_rows << ", N=" << N << ")";

  return desc;
}

}  // namespace monomoe
