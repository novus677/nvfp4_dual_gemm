#pragma once

#include <cstdio>
#include <cuda.h>

////////////////////////////////////////////////////////////////////////////////
// HELPER FUNCTION TO CHECK FOR ERRORS
////////////////////////////////////////////////////////////////////////////////
void cuda_check(CUresult code, const char *file, int line) {
  if (code != CUDA_SUCCESS) {
    char const *str;
    cuGetErrorString(code, &str);
    fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, str);
    exit(1);
  }
}

void cuda_check(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s:%n: %s\n", file, line,
            cudaGetErrorString(code));
    exit(1);
  }
}

// Macro for convenient CUDA error checking
#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cuda_check((x), __FILE__, __LINE__);                                       \
  } while (0)

////////////////////////////////////////////////////////////////////////////////
// ASYNC PROXY FENCE
////////////////////////////////////////////////////////////////////////////////

__device__ static __forceinline__ void async_proxy_fence() {
  asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
}

////////////////////////////////////////////////////////////////////////////////
// MBARRIER FUNCTIONS
////////////////////////////////////////////////////////////////////////////////

__device__ static __forceinline__ void init_barrier(uint64_t *bar,
                                                    int arrival_count) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar_ptr),
               "r"(arrival_count)
               : "memory");
}

__device__ static __forceinline__ void fence_barrier_init() {
  asm volatile("fence.mbarrier_init.release.cluster;\n");
}

__device__ static __forceinline__ void arrive(uint64_t *bar, uint32_t count) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0],  %1;\n"
               :
               : "r"(mbar_ptr), "r"(count)
               : "memory");
}

__device__ static __forceinline__ void wait(uint64_t *bar, int phaseParity) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile("{\n"
               ".reg .pred                P1;\n"
               "LAB_WAIT:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
               "@P1                       bra.uni DONE;\n"
               "bra.uni                   LAB_WAIT;\n"
               "DONE:\n"
               "}\n" ::"r"(mbar_ptr),
               "r"(phaseParity));
}

__device__ static __forceinline__ void expect_bytes_and_arrive(uint64_t *bar,
                                                               uint32_t bytes) {
  uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm("mbarrier.arrive.expect_tx.release.cta.shared.b64 _, [%0], %1;\n "
      :
      : "r"(bar_ptr), "r"(bytes)
      : "memory");
}

////////////////////////////////////////////////////////////////////////////////
// TMA GROUP OPERATIONS
////////////////////////////////////////////////////////////////////////////////

__device__ static __forceinline__ void tma_commit_group() {
  asm volatile("cp.async.bulk.commit_group;");
}

template <int N>
__device__ static __forceinline__ void tma_wait_until_pending() {
  asm volatile("cp.async.bulk.wait_group %0;" : : "n"(N) : "memory");
}

////////////////////////////////////////////////////////////////////////////////
// TMA CACHE POLICIES
////////////////////////////////////////////////////////////////////////////////

enum class CachePolicy : uint64_t {
  EVICT_NORMAL = 0x1000000000000000,
  EVICT_FIRST = 0x12F0000000000000,
  EVICT_LAST = 0x14F0000000000000,
};

////////////////////////////////////////////////////////////////////////////////
// GLOBAL -> SHARED
////////////////////////////////////////////////////////////////////////////////

__device__ static __forceinline__ void
cp_async_bulk_global_to_shared(void *smem_dest, const void *src, int size,
                               uint64_t *bar, CachePolicy cache_policy) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes."
      "L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
      :
      : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dest))),
        "l"(src), "r"(size), "r"(mbar_ptr), "l"(cache_policy)
      : "memory");
}

__device__ static __forceinline__ void cp_async_bulk_tensor_1d_global_to_shared(
    void *smem_dest, const CUtensorMap *tensor_map, int c0, uint64_t *bar,
    CachePolicy cache_policy) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "cp.async.bulk.tensor.1d.shared::cluster.global.tile.mbarrier::complete_"
      "tx::bytes.L2::cache_hint "
      "[%0], [%1, {%2}], [%3], %4;\n"
      :
      : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dest))),
        "l"(tensor_map), "r"(c0), "r"(mbar_ptr), "l"(cache_policy)
      : "memory");
}

__device__ static __forceinline__ void cp_async_bulk_tensor_2d_global_to_shared(
    void *smem_dest, const CUtensorMap *tensor_map, int c0, int c1,
    uint64_t *bar, CachePolicy cache_policy) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_"
      "tx::bytes.L2::cache_hint "
      "[%0], [%1, {%2, %3}], [%4], %5;\n"
      :
      : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dest))),
        "l"(tensor_map), "r"(c0), "r"(c1), "r"(mbar_ptr), "l"(cache_policy)
      : "memory");
}

__device__ static __forceinline__ void cp_async_bulk_tensor_3d_global_to_shared(
    void *smem_dest, const CUtensorMap *tensor_map, int c0, int c1, int c2,
    uint64_t *bar, CachePolicy cache_policy) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_"
      "tx::bytes.L2::cache_hint "
      "[%0], [%1, {%2, %3, %4}], [%5], %6;\n"
      :
      : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dest))),
        "l"(tensor_map), "r"(c0), "r"(c1), "r"(c2), "r"(mbar_ptr),
        "l"(cache_policy)
      : "memory");
}

__device__ static __forceinline__ void cp_async_bulk_tensor_4d_global_to_shared(
    void *smem_dest, const CUtensorMap *tensor_map, int c0, int c1, int c2,
    int c3, uint64_t *bar, CachePolicy cache_policy) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_"
      "tx::bytes.L2::cache_hint "
      "[%0], [%1, {%2, %3, %4, %5}], [%6], %7;\n"
      :
      : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dest))),
        "l"(tensor_map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(mbar_ptr),
        "l"(cache_policy)
      : "memory");
}

__device__ static __forceinline__ void cp_async_bulk_tensor_5d_global_to_shared(
    void *smem_dest, const CUtensorMap *tensor_map, int c0, int c1, int c2,
    int c3, int c4, uint64_t *bar, CachePolicy cache_policy) {
  uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
  asm volatile(
      "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_"
      "tx::bytes.L2::cache_hint "
      "[%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;\n"
      :
      : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(smem_dest))),
        "l"(tensor_map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4),
        "r"(mbar_ptr), "l"(cache_policy)
      : "memory");
}

////////////////////////////////////////////////////////////////////////////////
// SHARED -> GLOBAL
////////////////////////////////////////////////////////////////////////////////

__device__ static __forceinline__ void
cp_async_bulk_tensor_1d_shared_to_global(const CUtensorMap *tensor_map, int c0,
                                         const void *src) {
  asm volatile("cp.async.bulk.tensor.1d.global.shared::cta.tile.bulk_group "
               "[%0, {%1}], [%2];\n"
               :
               : "l"(tensor_map), "r"(c0),
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src)))
               : "memory");
}

__device__ static __forceinline__ void
cp_async_bulk_tensor_2d_shared_to_global(const CUtensorMap *tensor_map, int c0,
                                         int c1, const void *src) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group "
               "[%0, {%1, %2}], [%3];\n"
               :
               : "l"(tensor_map), "r"(c0), "r"(c1),
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src)))
               : "memory");
}

__device__ static __forceinline__ void
cp_async_bulk_tensor_3d_shared_to_global(const CUtensorMap *tensor_map, int c0,
                                         int c1, int c2, const void *src) {
  asm volatile("cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group "
               "[%0, {%1, %2, %3}], [%4];\n"
               :
               : "l"(tensor_map), "r"(c0), "r"(c1), "r"(c2),
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src)))
               : "memory");
}

__device__ static __forceinline__ void
cp_async_bulk_tensor_4d_shared_to_global(const CUtensorMap *tensor_map, int c0,
                                         int c1, int c2, int c3,
                                         const void *src) {
  asm volatile("cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group "
               "[%0, {%1, %2, %3, %4}], [%5];\n"
               :
               : "l"(tensor_map), "r"(c0), "r"(c1), "r"(c2), "r"(c3),
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src)))
               : "memory");
}

__device__ static __forceinline__ void
cp_async_bulk_tensor_5d_shared_to_global(const CUtensorMap *tensor_map, int c0,
                                         int c1, int c2, int c3, int c4,
                                         const void *src) {
  asm volatile("cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group "
               "[%0, {%1, %2, %3, %4, %5}], [%6];\n"
               :
               : "l"(tensor_map), "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(c4),
                 "r"(static_cast<uint32_t>(__cvta_generic_to_shared(src)))
               : "memory");
}
