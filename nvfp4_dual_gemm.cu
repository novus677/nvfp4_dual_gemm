#include "tcgen05-interface.cuh"
#include "tma-interface.cuh"

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAUtils.h>
#include <torch/library.h>

#define WARP_SIZE 32
#define WARPGROUP_SIZE 128
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

#define UMMA_K 64
#define SF_VEC_SIZE 16

__device__ __forceinline__ float silu(float x) {
  return x * (1.0f / (1.0f + __expf(-x)));
}

__device__ __forceinline__ float silu_fast(float x) {
  return fmaf(0.5f * x, __tanhf(0.5f * x), 0.5f * x);
}

__device__ __forceinline__ float silu_fast2(float x) {
  return fmaf(0.5f * x, __tanhf(0.49995f * x), 0.5f * x);
}

__device__ __forceinline__ void sync_wg(int wg_id) {
  asm volatile("bar.sync %0, 128;\n" ::"r"(wg_id + 1) : "memory");
}

__device__ __forceinline__ uint32_t elect_one_sync() {
  uint32_t pred = 0;
  asm volatile("{\n\t"
               ".reg .pred %%px;\n\t"
               "elect.sync _|%%px, %1;\n\t"
               "@%%px mov.s32 %0, 1;\n\t"
               "}"
               : "+r"(pred)
               : "r"(0xFFFFFFFF));
  return pred;
}

template <int32_t K, int32_t BLOCK_M, int32_t BLOCK_N, int32_t BLOCK_K,
          int32_t PIPE_DEPTH, bool FAST_SILU, bool FAST_SILU2, bool TRANSPOSE>
__global__ void __launch_bounds__(WARPGROUP_SIZE + 2 * WARP_SIZE, 1)
    nvfp4_dual_gemm_cutlass(int M, int N,
                            __grid_constant__ const CUtensorMap a_map,
                            __grid_constant__ const CUtensorMap b1_map,
                            __grid_constant__ const CUtensorMap b2_map,
                            uint8_t *SFA_ptr, uint8_t *SFB1_ptr,
                            uint8_t *SFB2_ptr, half *C_ptr) {

  extern __shared__ __align__(1024) char shmem[];
  __shared__ __align__(8) uint64_t matmul_done[PIPE_DEPTH];
  __shared__ __align__(8) uint64_t tma_done[PIPE_DEPTH];
  __shared__ __align__(8) uint64_t final_matmul_done[1];

  constexpr int32_t TILE_SIZE_A = BLOCK_K * BLOCK_M;
  constexpr int32_t TILE_SIZE_B = BLOCK_K * BLOCK_N;
  constexpr int32_t TILE_SIZE_SFA = (BLOCK_K / SF_VEC_SIZE) * 128;
  constexpr int32_t TILE_SIZE_SFB = (BLOCK_K / SF_VEC_SIZE) * 128;
  constexpr int32_t STAGE_BYTES =
      (TILE_SIZE_A + 2 * TILE_SIZE_B) / 2 + (TILE_SIZE_SFA + 2 * TILE_SIZE_SFB);

  constexpr int32_t TMEM_WIDTH_SFA = 16;
  constexpr int32_t TMEM_WIDTH_SFB = 16;

  const int32_t num_rows = CEIL_DIV(M, BLOCK_M);
  const int32_t num_cols = CEIL_DIV(N, BLOCK_N);
  const int32_t outer_row = blockIdx.x / num_cols;
  const int32_t outer_col = blockIdx.x % num_cols;

  const int32_t warp_id = threadIdx.x / WARP_SIZE;
  const int32_t lane_id = threadIdx.x % WARP_SIZE;
  const int32_t wg_lane_id = threadIdx.x % WARPGROUP_SIZE;

  if (warp_id == 0 && elect_one_sync()) {
    for (int32_t i = 0; i < PIPE_DEPTH; i++) {
      init_barrier(&matmul_done[i], 1);
      init_barrier(&tma_done[i], 1);
    }
    init_barrier(&final_matmul_done[0], 1);
    fence_barrier_init();
  } else if (warp_id == 1) {
    tcgen05_alloc(&shmem[0], 512);
  }
  __syncthreads();

  if (warp_id < 4) { // epilogue warps
    warpgroup_reg_alloc<256>();

    constexpr uint32_t tmem_d1 = 0;
    constexpr uint32_t tmem_d2 = tmem_d1 + BLOCK_N;
    constexpr auto silu_func = FAST_SILU    ? silu_fast
                               : FAST_SILU2 ? silu_fast2
                                            : silu;

    wait(&final_matmul_done[0], 0);

    if constexpr (TRANSPOSE) {
      float acc1[BLOCK_N / 4];
      float acc2[BLOCK_N / 4];

      half *c_off = C_ptr + (outer_col * BLOCK_N) * M + (outer_row * BLOCK_M);
      for (int32_t r = 0; r < 4; r++) {
        if constexpr (BLOCK_N == 128) {
          tcgen05_ld_32x32b_x32(tmem_d1 + r * (BLOCK_N / 4), acc1);
          tcgen05_ld_32x32b_x32(tmem_d2 + r * (BLOCK_N / 4), acc2);
        } else if constexpr (BLOCK_N == 64) {
          tcgen05_ld_32x32b_x16(tmem_d1 + r * (BLOCK_N / 4), acc1);
          tcgen05_ld_32x32b_x16(tmem_d2 + r * (BLOCK_N / 4), acc2);
        }
        tcgen05_wait_ld();

        for (int32_t i = 0; i < (BLOCK_N / 4); i++) {
          float val = silu_func(acc1[i]) * acc2[i];
          __stwt(&c_off[(r * (BLOCK_N / 4) + i) * M + wg_lane_id],
                 __float2half_rn(val));
        }
      }
    } else {
      float acc1[BLOCK_N / 4];
      float acc2[BLOCK_N / 4];

      half *c_off = C_ptr + (outer_row * BLOCK_M) * N + (outer_col * BLOCK_N);
      for (int32_t r0 = 0; r0 < 2; r0++) {
        for (int32_t r1 = 0; r1 < 2; r1++) {
          if constexpr (BLOCK_N == 128) {
            tcgen05_ld_16x256b_x8(
                tmem_d1 + r0 * (BLOCK_N / 2) + r1 * (16 << 16), acc1);
            tcgen05_ld_16x256b_x8(
                tmem_d2 + r0 * (BLOCK_N / 2) + r1 * (16 << 16), acc2);
          } else if constexpr (BLOCK_N == 64) {
            tcgen05_ld_16x256b_x4(
                tmem_d1 + r0 * (BLOCK_N / 2) + r1 * (16 << 16), acc1);
            tcgen05_ld_16x256b_x4(
                tmem_d2 + r0 * (BLOCK_N / 2) + r1 * (16 << 16), acc2);
          }
          tcgen05_wait_ld();

          for (int32_t i = 0; i < (BLOCK_N / 4); i += 4) {
            const int32_t row = warp_id * 32 + r1 * 16 + lane_id / 4;
            const int32_t col = r0 * (BLOCK_N / 2) + (lane_id % 4) * 2 + 2 * i;

            float val1 = silu_func(acc1[i]) * acc2[i];
            float val2 = silu_func(acc1[i + 1]) * acc2[i + 1];
            float val3 = silu_func(acc1[i + 2]) * acc2[i + 2];
            float val4 = silu_func(acc1[i + 3]) * acc2[i + 3];

            __stwt(reinterpret_cast<half2 *>(&c_off[row * N + col]),
                   __float22half2_rn(make_float2(val1, val2)));
            __stwt(reinterpret_cast<half2 *>(&c_off[(row + 8) * N + col]),
                   __float22half2_rn(make_float2(val3, val4)));
          }
        }
      }
    }

    sync_wg(0);
    if (warp_id == 0) {
      tcgen05_dealloc(0, 512);
    }
  } else {
    if (warp_id == 4 && elect_one_sync()) { // issue TMAs
      CachePolicy cache_policy_a = CachePolicy::EVICT_LAST;
      CachePolicy cache_policy_b = CachePolicy::EVICT_FIRST;

      auto issue_tma = [&](int32_t k, int32_t pipe_idx) {
        expect_bytes_and_arrive(&tma_done[pipe_idx], STAGE_BYTES);

        char *a_shr = shmem + pipe_idx * STAGE_BYTES;
        char *b1_shr = a_shr + TILE_SIZE_A / 2;
        char *b2_shr = b1_shr + TILE_SIZE_B / 2;
        char *sfa_shr = b2_shr + TILE_SIZE_B / 2;
        char *sfb1_shr = sfa_shr + TILE_SIZE_SFA;
        char *sfb2_shr = sfb1_shr + TILE_SIZE_SFB;

        cp_async_bulk_tensor_2d_global_to_shared(
            a_shr, &a_map, k * BLOCK_K, outer_row * BLOCK_M,
            &tma_done[pipe_idx], cache_policy_a);
        cp_async_bulk_tensor_2d_global_to_shared(
            b1_shr, &b1_map, k * BLOCK_K, outer_col * BLOCK_N,
            &tma_done[pipe_idx], cache_policy_b);
        cp_async_bulk_tensor_2d_global_to_shared(
            b2_shr, &b2_map, k * BLOCK_K, outer_col * BLOCK_N,
            &tma_done[pipe_idx], cache_policy_b);

        cp_async_bulk_global_to_shared(
            sfa_shr,
            SFA_ptr + ((outer_row * BLOCK_M / 128) * ((K / SF_VEC_SIZE) / 4) +
                       k * 4) *
                          512,
            TILE_SIZE_SFA, &tma_done[pipe_idx], cache_policy_a);
        cp_async_bulk_global_to_shared(
            sfb1_shr,
            SFB1_ptr + ((outer_col * BLOCK_N / 128) * ((K / SF_VEC_SIZE) / 4) +
                        k * 4) *
                           512,
            TILE_SIZE_SFB, &tma_done[pipe_idx], cache_policy_b);
        cp_async_bulk_global_to_shared(
            sfb2_shr,
            SFB2_ptr + ((outer_col * BLOCK_N / 128) * ((K / SF_VEC_SIZE) / 4) +
                        k * 4) *
                           512,
            TILE_SIZE_SFB, &tma_done[pipe_idx], cache_policy_b);
      };

      for (int32_t k = 0; k < PIPE_DEPTH; k++) {
        issue_tma(k, k);
      }
      for (int32_t k = PIPE_DEPTH; k < (K / BLOCK_K); k++) {
        const int32_t pipe_idx = k % PIPE_DEPTH;
        const int32_t phase = (k / PIPE_DEPTH) % 2;
        wait(&matmul_done[pipe_idx], phase ^ 1);
        issue_tma(k, pipe_idx);
      }
    } else if (warp_id == 5 && elect_one_sync()) { // issue UMMAs
      constexpr uint32_t tmem_d1 = 0;
      constexpr uint32_t tmem_d2 = tmem_d1 + BLOCK_N;
      constexpr uint32_t tmem_sfa = tmem_d2 + BLOCK_N;
      constexpr uint32_t tmem_sfb1 = tmem_sfa + TMEM_WIDTH_SFA;
      constexpr uint32_t tmem_sfb2 = tmem_sfb1 + TMEM_WIDTH_SFB;

      constexpr uint32_t inst_desc =
          make_inst_desc<BLOCK_M, BLOCK_N, UMMA_K, 0, 0>();

      const uint32_t sfa_offset = (BLOCK_M == 64) ? (outer_row % 2) * 2 : 0;
      const uint32_t sfb_offset = (BLOCK_N == 64) ? (outer_col % 2) * 2 : 0;

#pragma unroll 4
      for (int32_t k = 0; k < (K / BLOCK_K); k++) {
        const int32_t pipe_idx = k % PIPE_DEPTH;
        const int32_t phase = (k / PIPE_DEPTH) % 2;

        char *a_shr = shmem + pipe_idx * STAGE_BYTES;
        char *b1_shr = a_shr + TILE_SIZE_A / 2;
        char *b2_shr = b1_shr + TILE_SIZE_B / 2;
        char *sfa_shr = b2_shr + TILE_SIZE_B / 2;
        char *sfb1_shr = sfa_shr + TILE_SIZE_SFA;
        char *sfb2_shr = sfb1_shr + TILE_SIZE_SFB;

        const uint64_t desc_a = make_smem_desc<SWIZZLE_128B>(a_shr, 1, 1024);
        const uint64_t desc_b1 = make_smem_desc<SWIZZLE_128B>(b1_shr, 1, 1024);
        const uint64_t desc_b2 = make_smem_desc<SWIZZLE_128B>(b2_shr, 1, 1024);
        const uint64_t desc_sfa = make_smem_desc<NO_SWIZZLE>(sfa_shr, 128, 128);
        const uint64_t desc_sfb1 =
            make_smem_desc<NO_SWIZZLE>(sfb1_shr, 128, 128);
        const uint64_t desc_sfb2 =
            make_smem_desc<NO_SWIZZLE>(sfb2_shr, 128, 128);

        wait(&tma_done[pipe_idx], phase);

        for (int32_t j = 0; j < 4; j++) {
          tcgen05_cp(desc_sfa + j * 32, tmem_sfa + j * 4);
          tcgen05_cp(desc_sfb1 + j * 32, tmem_sfb1 + j * 4);
          tcgen05_cp(desc_sfb2 + j * 32, tmem_sfb2 + j * 4);

          if (j == 0 && k == 0) {
            tcgen05_mma<0, inst_desc, CollectorUsage::FILL>(
                desc_a + j * 2, desc_b1 + j * 2, tmem_d1,
                tmem_sfa + j * (TMEM_WIDTH_SFA / 4) + sfa_offset,
                tmem_sfb1 + j * (TMEM_WIDTH_SFB / 4) + sfb_offset);
            tcgen05_mma<0, inst_desc, CollectorUsage::LASTUSE>(
                desc_a + j * 2, desc_b2 + j * 2, tmem_d2,
                tmem_sfa + j * (TMEM_WIDTH_SFA / 4) + sfa_offset,
                tmem_sfb2 + j * (TMEM_WIDTH_SFB / 4) + sfb_offset);
          } else {
            tcgen05_mma<1, inst_desc, CollectorUsage::FILL>(
                desc_a + j * 2, desc_b1 + j * 2, tmem_d1,
                tmem_sfa + j * (TMEM_WIDTH_SFA / 4) + sfa_offset,
                tmem_sfb1 + j * (TMEM_WIDTH_SFB / 4) + sfb_offset);
            tcgen05_mma<1, inst_desc, CollectorUsage::LASTUSE>(
                desc_a + j * 2, desc_b2 + j * 2, tmem_d2,
                tmem_sfa + j * (TMEM_WIDTH_SFA / 4) + sfa_offset,
                tmem_sfb2 + j * (TMEM_WIDTH_SFB / 4) + sfb_offset);
          }
        }
        tcgen05_commit(&matmul_done[pipe_idx]);
      }
      tcgen05_commit(&final_matmul_done[0]);
    }
  }
}

template <int32_t K, int32_t BLOCK_M, int32_t BLOCK_N, int32_t BLOCK_K,
          int32_t PIPE_DEPTH, bool FAST_SILU, bool FAST_SILU2, bool TRANSPOSE,
          int32_t GRID_DIM>
void launch_nvfp4_dual_gemm(int M, int N, uint8_t *A_ptr, uint8_t *B1_ptr,
                            uint8_t *B2_ptr, uint8_t *SFA_ptr,
                            uint8_t *SFB1_ptr, uint8_t *SFB2_ptr, half *C_ptr) {

  constexpr int32_t BLOCK_DIM = WARPGROUP_SIZE + 2 * WARP_SIZE;

  CUtensorMap tensorMapA;
  const cuuint64_t globalDimA[2] = {K, M};
  const cuuint64_t globalStridesA[1] = {K / 2};
  const cuuint32_t boxDimA[2] = {BLOCK_K, BLOCK_M};
  const cuuint32_t elementStridesA[2] = {1, 1};
  CUDA_CHECK(cuTensorMapEncodeTiled(
      &tensorMapA, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 2, A_ptr, globalDimA,
      globalStridesA, boxDimA, elementStridesA, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

  CUtensorMap tensorMapB1, tensorMapB2;
  const cuuint64_t globalDimB[2] = {K, N};
  const cuuint64_t globalStridesB[1] = {K / 2};
  const cuuint32_t boxDimB[2] = {BLOCK_K, BLOCK_N};
  const cuuint32_t elementStridesB[2] = {1, 1};
  CUDA_CHECK(cuTensorMapEncodeTiled(
      &tensorMapB1, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 2, B1_ptr, globalDimB,
      globalStridesB, boxDimB, elementStridesB, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  CUDA_CHECK(cuTensorMapEncodeTiled(
      &tensorMapB2, CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B, 2, B2_ptr, globalDimB,
      globalStridesB, boxDimB, elementStridesB, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));

  constexpr int32_t shmem_size_a = PIPE_DEPTH * (BLOCK_K * BLOCK_M) / 2;
  constexpr int32_t shmem_size_b = PIPE_DEPTH * (BLOCK_K * BLOCK_N) / 2;
  constexpr int32_t shmem_size_sfa = PIPE_DEPTH * (BLOCK_K / SF_VEC_SIZE) * 128;
  constexpr int32_t shmem_size_sfb = PIPE_DEPTH * (BLOCK_K / SF_VEC_SIZE) * 128;
  constexpr int32_t shmem_size =
      shmem_size_a + 2 * shmem_size_b + shmem_size_sfa + 2 * shmem_size_sfb;
  static_assert(shmem_size <= 227 * 1024, "Shared memory size exceeds 227 KB");
  cudaFuncSetAttribute(
      nvfp4_dual_gemm_cutlass<K, BLOCK_M, BLOCK_N, BLOCK_K, PIPE_DEPTH,
                              FAST_SILU, FAST_SILU2, TRANSPOSE>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size);

  // const int32_t num_rows = CEIL_DIV(M, BLOCK_M);
  // const int32_t num_cols = CEIL_DIV(N, BLOCK_N);

  nvfp4_dual_gemm_cutlass<K, BLOCK_M, BLOCK_N, BLOCK_K, PIPE_DEPTH, FAST_SILU,
                          FAST_SILU2, TRANSPOSE>
      <<<GRID_DIM, BLOCK_DIM, shmem_size>>>(M, N, tensorMapA, tensorMapB1,
                                            tensorMapB2, SFA_ptr, SFB1_ptr,
                                            SFB2_ptr, C_ptr);
}

#define LAUNCH(K, BLOCK_M, BLOCK_N, BLOCK_K, PIPE_DEPTH, FAST_SILU,            \
               FAST_SILU2, TRANSPOSE, GRID_DIM)                                \
  launch_nvfp4_dual_gemm<K, BLOCK_M, BLOCK_N, BLOCK_K, PIPE_DEPTH, FAST_SILU,  \
                         FAST_SILU2, TRANSPOSE, GRID_DIM>(                     \
      M, N, A_ptr, B1_ptr, B2_ptr, SFA_ptr, SFB1_ptr, SFB2_ptr, C_ptr)

torch::Tensor
cuda_nvfp4_dual_gemm(const torch::Tensor &A, const torch::Tensor &B1,
                     const torch::Tensor &B2, const torch::Tensor &SFA,
                     const torch::Tensor &SFB1, const torch::Tensor &SFB2,
                     torch::Tensor &C) {
  const int M = A.size(0);
  const int N = B1.size(0);
  const int K = A.size(1) * 2;

  uint8_t *A_ptr = reinterpret_cast<uint8_t *>(A.data_ptr());
  uint8_t *B1_ptr = reinterpret_cast<uint8_t *>(B1.data_ptr());
  uint8_t *B2_ptr = reinterpret_cast<uint8_t *>(B2.data_ptr());
  uint8_t *SFA_ptr = reinterpret_cast<uint8_t *>(SFA.data_ptr());
  uint8_t *SFB1_ptr = reinterpret_cast<uint8_t *>(SFB1.data_ptr());
  uint8_t *SFB2_ptr = reinterpret_cast<uint8_t *>(SFB2.data_ptr());
  half *C_ptr = reinterpret_cast<half *>(C.data_ptr());

  auto hash = [](int x, int y, int z) -> uint64_t {
    return static_cast<uint64_t>(x) << 32 | static_cast<uint64_t>(y) << 16 |
           static_cast<uint64_t>(z);
  };

  switch (hash(M, N, K)) {
  [[likely]] case hash(256, 4096, 7168):
    LAUNCH(7168, 128, 64, 256, 5, true, false, false, 128);
    return C;
  [[likely]] case hash(512, 4096, 7168):
    LAUNCH(7168, 128, 128, 256, 4, false, true, false, 128);
    return C;
  [[likely]] case hash(256, 3072, 4096):
    LAUNCH(4096, 128, 64, 256, 5, true, false, false, 96);
    return C;
  [[likely]] case hash(512, 3072, 7168):
    LAUNCH(7168, 128, 128, 256, 4, true, false, true, 96);
    return C.view({N, M, 1}).transpose(0, 1);
  [[unlikely]] case hash(1536, 512, 7168):
    LAUNCH(7168, 128, 128, 256, 4, false, false, false, 48);
    return C;
  [[unlikely]] case hash(256, 512, 256):
    LAUNCH(256, 128, 128, 256, 4, false, false, false, 8);
    return C;
  [[unlikely]] case hash(3072, 1024, 1536):
    LAUNCH(1536, 128, 128, 256, 4, false, false, false, 192);
    return C;
  [[unlikely]] case hash(7168, 1024, 256):
    LAUNCH(256, 128, 128, 256, 4, false, false, false, 448);
    return C;
  [[unlikely]] case hash(7168, 2304, 2048):
    LAUNCH(2048, 128, 128, 256, 4, false, false, false, 1008);
    return C;
  [[unlikely]] case hash(4608, 384, 7168):
    LAUNCH(7168, 128, 128, 256, 4, false, false, false, 108);
    return C;
  [[unlikely]] case hash(7168, 384, 2304):
    LAUNCH(2304, 128, 128, 256, 4, false, false, false, 168);
    return C;
  [[unlikely]] case hash(512, 768, 7168):
    LAUNCH(7168, 128, 128, 256, 4, false, false, false, 24);
    return C;
  [[unlikely]] case hash(4096, 768, 512):
    LAUNCH(512, 128, 128, 256, 4, false, false, false, 192);
    return C;
  [[unlikely]] default:
    return C;
  }
}
