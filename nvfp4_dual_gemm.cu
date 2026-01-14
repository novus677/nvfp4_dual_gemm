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

template <int32_t BLOCK_M, int32_t BLOCK_N, int32_t BLOCK_K, int32_t PIPE_DEPTH>
__global__ void __launch_bounds__(WARPGROUP_SIZE + 2 * WARP_SIZE)
    nvfp4_dual_gemm(int M, int N, int K,
                    __grid_constant__ const CUtensorMap a_map,
                    __grid_constant__ const CUtensorMap b1_map,
                    __grid_constant__ const CUtensorMap b2_map,
                    uint8_t *SFA_ptr, uint8_t *SFB1_ptr, uint8_t *SFB2_ptr,
                    uint8_t *C_ptr) {

  extern __shared__ __align__(1024) char shmem[];
  __shared__ __align__(8) uint64_t matmul_done[PIPE_DEPTH];
  __shared__ __align__(8) uint64_t tma_done[PIPE_DEPTH];
  __shared__ __align__(8) uint64_t final_matmul_done[1];
  __shared__ uint32_t tmem_base[1];

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

  if (warp_id == 0 && elect_one_sync()) {
    for (int32_t i = 0; i < PIPE_DEPTH; i++) {
      init_barrier(&matmul_done[i], 1);
      init_barrier(&tma_done[i], 1);
    }
    init_barrier(&final_matmul_done[0], 1);
    async_proxy_fence();
  } else if (warp_id == 1) {
    tcgen05_alloc(&tmem_base[0], 512);
  }
  __syncthreads();

  if (warp_id < 4) { // epilogue warps
    const uint32_t tmem_d1 = tmem_base[0];
    const uint32_t tmem_d2 = tmem_d1 + BLOCK_N;

    float acc1[BLOCK_N / 2];
    float acc2[BLOCK_N / 2];

    wait(&final_matmul_done[0], 0);

    half *c_off = reinterpret_cast<half *>(C_ptr) + (outer_row * BLOCK_M) * N +
                  (outer_col * BLOCK_N);
    for (int32_t r = 0; r < 2; r++) {
      if constexpr (BLOCK_N == 128) {
        tcgen05_ld_16x256b_x16(tmem_d1 + r * (16 << 16), acc1);
        tcgen05_ld_16x256b_x16(tmem_d2 + r * (16 << 16), acc2);
      } else if constexpr (BLOCK_N == 64) {
        tcgen05_ld_16x256b_x8(tmem_d1 + r * (16 << 16), acc1);
        tcgen05_ld_16x256b_x8(tmem_d2 + r * (16 << 16), acc2);
      }
      tcgen05_wait_ld();

      for (int32_t i = 0; i < (BLOCK_N / 2); i += 4) {
        const int32_t row = warp_id * 32 + r * 16 + lane_id / 4;
        const int32_t col = (lane_id % 4) * 2 + 2 * i;

        float val1 = silu(acc1[i]) * acc2[i];
        float val2 = silu(acc1[i + 1]) * acc2[i + 1];
        float val3 = silu(acc1[i + 2]) * acc2[i + 2];
        float val4 = silu(acc1[i + 3]) * acc2[i + 3];

        __stwt(reinterpret_cast<half2 *>(&c_off[row * N + col]),
               __float22half2_rn(make_float2(val1, val2)));
        __stwt(reinterpret_cast<half2 *>(&c_off[(row + 8) * N + col]),
               __float22half2_rn(make_float2(val3, val4)));
      }
    }

    sync_wg(0);
    if (warp_id == 0) {
      tcgen05_dealloc(tmem_d1, 512);
    }
  } else {
    if (warp_id == 4 && elect_one_sync()) { // issue TMAs
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
            &tma_done[pipe_idx], CachePolicy::EVICT_NORMAL);
        cp_async_bulk_tensor_2d_global_to_shared(
            b1_shr, &b1_map, k * BLOCK_K, outer_col * BLOCK_N,
            &tma_done[pipe_idx], CachePolicy::EVICT_NORMAL);
        cp_async_bulk_tensor_2d_global_to_shared(
            b2_shr, &b2_map, k * BLOCK_K, outer_col * BLOCK_N,
            &tma_done[pipe_idx], CachePolicy::EVICT_NORMAL);

        cp_async_bulk_global_to_shared(
            sfa_shr,
            SFA_ptr + ((outer_row * BLOCK_M / 128) * ((K / SF_VEC_SIZE) / 4) +
                       k * 4) *
                          512,
            TILE_SIZE_SFA, &tma_done[pipe_idx], CachePolicy::EVICT_NORMAL);
        cp_async_bulk_global_to_shared(
            sfb1_shr,
            SFB1_ptr + ((outer_col * BLOCK_N / 128) * ((K / SF_VEC_SIZE) / 4) +
                        k * 4) *
                           512,
            TILE_SIZE_SFB, &tma_done[pipe_idx], CachePolicy::EVICT_NORMAL);
        cp_async_bulk_global_to_shared(
            sfb2_shr,
            SFB2_ptr + ((outer_col * BLOCK_N / 128) * ((K / SF_VEC_SIZE) / 4) +
                        k * 4) *
                           512,
            TILE_SIZE_SFB, &tma_done[pipe_idx], CachePolicy::EVICT_NORMAL);
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
      const uint32_t tmem_d1 = tmem_base[0];
      const uint32_t tmem_d2 = tmem_d1 + BLOCK_N;
      const uint32_t tmem_sfa = tmem_d2 + BLOCK_N;
      const uint32_t tmem_sfb1 = tmem_sfa + TMEM_WIDTH_SFA;
      const uint32_t tmem_sfb2 = tmem_sfb1 + TMEM_WIDTH_SFB;

      constexpr uint32_t inst_desc =
          make_inst_desc<BLOCK_M, BLOCK_N, UMMA_K, 0, 0>();

      int32_t pipe_idx;
      uint32_t phase;
      for (int32_t k = 0; k < (K / BLOCK_K); k++) {
        pipe_idx = k % PIPE_DEPTH;
        phase = (k / PIPE_DEPTH) % 2;

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
        }

        const uint32_t sfa_offset = (BLOCK_M == 64) ? (outer_row % 2) * 2 : 0;
        const uint32_t sfb_offset = (BLOCK_N == 64) ? (outer_col % 2) * 2 : 0;
        for (int32_t j = 0; j < 4; j++) {
          if (j == 0 && k == 0) {
            tcgen05_mma<0, inst_desc>(
                desc_a + j * 2, desc_b1 + j * 2, tmem_d1,
                tmem_sfa + j * (TMEM_WIDTH_SFA / 4) + sfa_offset,
                tmem_sfb1 + j * (TMEM_WIDTH_SFB / 4) + sfb_offset);
            tcgen05_mma<0, inst_desc>(
                desc_a + j * 2, desc_b2 + j * 2, tmem_d2,
                tmem_sfa + j * (TMEM_WIDTH_SFA / 4) + sfa_offset,
                tmem_sfb2 + j * (TMEM_WIDTH_SFB / 4) + sfb_offset);
          } else {
            tcgen05_mma<1, inst_desc>(
                desc_a + j * 2, desc_b1 + j * 2, tmem_d1,
                tmem_sfa + j * (TMEM_WIDTH_SFA / 4) + sfa_offset,
                tmem_sfb1 + j * (TMEM_WIDTH_SFB / 4) + sfb_offset);
            tcgen05_mma<1, inst_desc>(
                desc_a + j * 2, desc_b2 + j * 2, tmem_d2,
                tmem_sfa + j * (TMEM_WIDTH_SFA / 4) + sfa_offset,
                tmem_sfb2 + j * (TMEM_WIDTH_SFB / 4) + sfb_offset);
          }
        }
        tcgen05_commit(&matmul_done[pipe_idx]);
      }
      wait(&matmul_done[pipe_idx], phase);
      tcgen05_commit(&final_matmul_done[0]);
    }
  }
}

template <int32_t BLOCK_M, int32_t BLOCK_N, int32_t BLOCK_K, int32_t PIPE_DEPTH>
void launch_nvfp4_dual_gemm(int M, int N, int K, uint8_t *A_ptr,
                            uint8_t *B1_ptr, uint8_t *B2_ptr, uint8_t *SFA_ptr,
                            uint8_t *SFB1_ptr, uint8_t *SFB2_ptr,
                            uint8_t *C_ptr) {
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

  constexpr int32_t BLOCK_DIM = WARPGROUP_SIZE + 2 * WARP_SIZE;

  constexpr int32_t shmem_size_a = PIPE_DEPTH * (BLOCK_K * BLOCK_M) / 2;
  constexpr int32_t shmem_size_b = PIPE_DEPTH * (BLOCK_K * BLOCK_N) / 2;
  constexpr int32_t shmem_size_sfa = PIPE_DEPTH * (BLOCK_K / SF_VEC_SIZE) * 128;
  constexpr int32_t shmem_size_sfb = PIPE_DEPTH * (BLOCK_K / SF_VEC_SIZE) * 128;
  constexpr int32_t shmem_size =
      shmem_size_a + 2 * shmem_size_b + shmem_size_sfa + 2 * shmem_size_sfb;
  static_assert(shmem_size <= 227 * 1024, "Shared memory size exceeds 227 KB");
  CUDA_CHECK(cudaFuncSetAttribute(
      nvfp4_dual_gemm<BLOCK_M, BLOCK_N, BLOCK_K, PIPE_DEPTH>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));

  const int32_t num_rows = CEIL_DIV(M, BLOCK_M);
  const int32_t num_cols = CEIL_DIV(N, BLOCK_N);

  nvfp4_dual_gemm<BLOCK_M, BLOCK_N, BLOCK_K, PIPE_DEPTH>
      <<<num_rows * num_cols, BLOCK_DIM, shmem_size>>>(
          M, N, K, tensorMapA, tensorMapB1, tensorMapB2, SFA_ptr, SFB1_ptr,
          SFB2_ptr, C_ptr);
}

#define LAUNCH(BLOCK_M, BLOCK_N, BLOCK_K, PIPE_DEPTH)                          \
  launch_nvfp4_dual_gemm<BLOCK_M, BLOCK_N, BLOCK_K, PIPE_DEPTH>(               \
      M, N, K, A_ptr, B1_ptr, B2_ptr, SFA_ptr, SFB1_ptr, SFB2_ptr, C_ptr)

torch::Tensor
cuda_nvfp4_dual_gemm(const torch::Tensor &A, const torch::Tensor &B1,
                     const torch::Tensor &B2, const torch::Tensor &SFA,
                     const torch::Tensor &SFB1, const torch::Tensor &SFB2,
                     torch::Tensor &C) {
  const int M = A.size(0);
  const int N = B1.size(0);
  const int K = A.size(1) * 2;
  const int L = A.size(2);

  if (L != 1) [[unlikely]]
    throw std::runtime_error("Batch size must be 1");

  uint8_t *A_ptr = reinterpret_cast<uint8_t *>(A.data_ptr());
  uint8_t *B1_ptr = reinterpret_cast<uint8_t *>(B1.data_ptr());
  uint8_t *B2_ptr = reinterpret_cast<uint8_t *>(B2.data_ptr());
  uint8_t *SFA_ptr = reinterpret_cast<uint8_t *>(SFA.data_ptr());
  uint8_t *SFB1_ptr = reinterpret_cast<uint8_t *>(SFB1.data_ptr());
  uint8_t *SFB2_ptr = reinterpret_cast<uint8_t *>(SFB2.data_ptr());
  uint8_t *C_ptr = reinterpret_cast<uint8_t *>(C.data_ptr());

  LAUNCH(128, 64, 256, 1);

  return C;
}
