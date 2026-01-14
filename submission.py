#!POPCORN leaderboard nvfp4_dual_gemm

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

CPP_SRC = r"""
#include <torch/extension.h>

torch::Tensor cuda_nvfp4_dual_gemm(const torch::Tensor &A, const torch::Tensor &B1,
                                   const torch::Tensor &B2, const torch::Tensor &SFA,
                                   const torch::Tensor &SFB1, const torch::Tensor &SFB2,
                                   torch::Tensor &C);
"""

CUDA_HEADERS = [
    r"""
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
    """,
    r"""
    #include <cuda_fp16.h>

    ////////////////////////////////////////////////////////////////////////////////
    // WARP GROUP REGISTER ALLOCATION
    ////////////////////////////////////////////////////////////////////////////////

    template <uint32_t RegCount> __device__ void warpgroup_reg_alloc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
    }

    template <uint32_t RegCount> __device__ void warpgroup_reg_dealloc() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
    }

    ////////////////////////////////////////////////////////////////////////////////
    // tcgen05 COMMIT GROUP FUNCTIONS
    ////////////////////////////////////////////////////////////////////////////////

    __device__ void tcgen05_commit(uint64_t *bar) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\n"
                :
                : "r"(mbar_ptr)
                : "memory");
    }

    ////////////////////////////////////////////////////////////////////////////////
    // TMEM FUNCTIONS
    ////////////////////////////////////////////////////////////////////////////////

    __device__ void tcgen05_alloc(uint32_t *dst, uint32_t cols) {
    uint32_t smem_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n"
        :
        : "r"(smem_ptr), "r"(cols));
    }

    __device__ void tcgen05_dealloc(uint32_t tmem, uint32_t cols) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n"
                :
                : "r"(tmem), "r"(cols));
    }

    __device__ void tcgen05_ld_16x256b_x8(uint32_t tmem, float d[32]) {
    asm volatile("{\n"
                "tcgen05.ld.sync.aligned.16x256b.x8.b32 "
                "{%0, %1, %2, %3, %4, %5, %6, %7, "
                " %8, %9, %10, %11, %12, %13, %14, %15, "
                " %16, %17, %18, %19, %20, %21, %22, %23, "
                " %24, %25, %26, %27, %28, %29, %30, %31},"
                " [%32];\n"
                "}\n"
                : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]),
                    "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]),
                    "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]),
                    "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]),
                    "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]),
                    "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]),
                    "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]),
                    "=f"(d[30]), "=f"(d[31])
                : "r"(tmem));
    }

    __device__ void tcgen05_ld_16x256b_x16(uint32_t tmem, float d[64]) {
    asm volatile(
        "{\n"
        "tcgen05.ld.sync.aligned.16x256b.x16.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        " %8, %9, %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47, "
        " %48, %49, %50, %51, %52, %53, %54, %55, "
        " %56, %57, %58, %59, %60, %61, %62, %63},"
        " [%64];\n"
        "}\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]),
            "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]),
            "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]),
            "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]),
            "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]),
            "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]),
            "=f"(d[31]), "=f"(d[32]), "=f"(d[33]), "=f"(d[34]), "=f"(d[35]),
            "=f"(d[36]), "=f"(d[37]), "=f"(d[38]), "=f"(d[39]), "=f"(d[40]),
            "=f"(d[41]), "=f"(d[42]), "=f"(d[43]), "=f"(d[44]), "=f"(d[45]),
            "=f"(d[46]), "=f"(d[47]), "=f"(d[48]), "=f"(d[49]), "=f"(d[50]),
            "=f"(d[51]), "=f"(d[52]), "=f"(d[53]), "=f"(d[54]), "=f"(d[55]),
            "=f"(d[56]), "=f"(d[57]), "=f"(d[58]), "=f"(d[59]), "=f"(d[60]),
            "=f"(d[61]), "=f"(d[62]), "=f"(d[63])
        : "r"(tmem));
    }

    __device__ void tcgen05_ld_16x256b_x32(uint32_t tmem, float d[128]) {
    asm volatile(
        "{\n"
        "tcgen05.ld.sync.aligned.16x256b.x32.b32 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103,  "
        " %104, %105, %106, %107, %108, %109, %110, %111,  "
        " %112, %113, %114, %115, %116, %117, %118, %119,  "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        " [%128];\n"
        "}\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]),
            "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]),
            "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]),
            "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]),
            "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]),
            "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]),
            "=f"(d[31]), "=f"(d[32]), "=f"(d[33]), "=f"(d[34]), "=f"(d[35]),
            "=f"(d[36]), "=f"(d[37]), "=f"(d[38]), "=f"(d[39]), "=f"(d[40]),
            "=f"(d[41]), "=f"(d[42]), "=f"(d[43]), "=f"(d[44]), "=f"(d[45]),
            "=f"(d[46]), "=f"(d[47]), "=f"(d[48]), "=f"(d[49]), "=f"(d[50]),
            "=f"(d[51]), "=f"(d[52]), "=f"(d[53]), "=f"(d[54]), "=f"(d[55]),
            "=f"(d[56]), "=f"(d[57]), "=f"(d[58]), "=f"(d[59]), "=f"(d[60]),
            "=f"(d[61]), "=f"(d[62]), "=f"(d[63]), "=f"(d[64]), "=f"(d[65]),
            "=f"(d[66]), "=f"(d[67]), "=f"(d[68]), "=f"(d[69]), "=f"(d[70]),
            "=f"(d[71]), "=f"(d[72]), "=f"(d[73]), "=f"(d[74]), "=f"(d[75]),
            "=f"(d[76]), "=f"(d[77]), "=f"(d[78]), "=f"(d[79]), "=f"(d[80]),
            "=f"(d[81]), "=f"(d[82]), "=f"(d[83]), "=f"(d[84]), "=f"(d[85]),
            "=f"(d[86]), "=f"(d[87]), "=f"(d[88]), "=f"(d[89]), "=f"(d[90]),
            "=f"(d[91]), "=f"(d[92]), "=f"(d[93]), "=f"(d[94]), "=f"(d[95]),
            "=f"(d[96]), "=f"(d[97]), "=f"(d[98]), "=f"(d[99]), "=f"(d[100]),
            "=f"(d[101]), "=f"(d[102]), "=f"(d[103]), "=f"(d[104]), "=f"(d[105]),
            "=f"(d[106]), "=f"(d[107]), "=f"(d[108]), "=f"(d[109]), "=f"(d[110]),
            "=f"(d[111]), "=f"(d[112]), "=f"(d[113]), "=f"(d[114]), "=f"(d[115]),
            "=f"(d[116]), "=f"(d[117]), "=f"(d[118]), "=f"(d[119]), "=f"(d[120]),
            "=f"(d[121]), "=f"(d[122]), "=f"(d[123]), "=f"(d[124]), "=f"(d[125]),
            "=f"(d[126]), "=f"(d[127])
        : "r"(tmem));
    }

    __device__ void tcgen05_ld_32x32b_x32(uint32_t tmem, float d[32]) {
    asm volatile("{\n"
                "tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                "{%0, %1, %2, %3, %4, %5, %6, %7, "
                " %8, %9, %10, %11, %12, %13, %14, %15, "
                " %16, %17, %18, %19, %20, %21, %22, %23, "
                " %24, %25, %26, %27, %28, %29, %30, %31},"
                " [%32];\n"
                "}\n"
                : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]),
                    "=f"(d[5]), "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]),
                    "=f"(d[10]), "=f"(d[11]), "=f"(d[12]), "=f"(d[13]),
                    "=f"(d[14]), "=f"(d[15]), "=f"(d[16]), "=f"(d[17]),
                    "=f"(d[18]), "=f"(d[19]), "=f"(d[20]), "=f"(d[21]),
                    "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]),
                    "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]),
                    "=f"(d[30]), "=f"(d[31])
                : "r"(tmem));
    }

    __device__ void tcgen05_ld_32x32b_x64(uint32_t tmem, float d[64]) {
    asm volatile(
        "{\n"
        "tcgen05.ld.sync.aligned.32x32b.x64.b32 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, "
        " %8, %9, %10, %11, %12, %13, %14, %15, "
        " %16, %17, %18, %19, %20, %21, %22, %23, "
        " %24, %25, %26, %27, %28, %29, %30, %31, "
        " %32, %33, %34, %35, %36, %37, %38, %39, "
        " %40, %41, %42, %43, %44, %45, %46, %47, "
        " %48, %49, %50, %51, %52, %53, %54, %55, "
        " %56, %57, %58, %59, %60, %61, %62, %63},"
        " [%64];\n"
        "}\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]),
            "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]),
            "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]),
            "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]),
            "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]),
            "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]),
            "=f"(d[31]), "=f"(d[32]), "=f"(d[33]), "=f"(d[34]), "=f"(d[35]),
            "=f"(d[36]), "=f"(d[37]), "=f"(d[38]), "=f"(d[39]), "=f"(d[40]),
            "=f"(d[41]), "=f"(d[42]), "=f"(d[43]), "=f"(d[44]), "=f"(d[45]),
            "=f"(d[46]), "=f"(d[47]), "=f"(d[48]), "=f"(d[49]), "=f"(d[50]),
            "=f"(d[51]), "=f"(d[52]), "=f"(d[53]), "=f"(d[54]), "=f"(d[55]),
            "=f"(d[56]), "=f"(d[57]), "=f"(d[58]), "=f"(d[59]), "=f"(d[60]),
            "=f"(d[61]), "=f"(d[62]), "=f"(d[63])
        : "r"(tmem));
    }

    __device__ void tcgen05_ld_32x32b_x128(uint32_t tmem, float d[128]) {
    asm volatile(
        "{\n"
        "tcgen05.ld.sync.aligned.32x32b.x128.b32 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7,   "
        " %8,   %9,   %10,  %11,  %12,  %13,  %14,  %15,  "
        " %16,  %17,  %18,  %19,  %20,  %21,  %22,  %23,  "
        " %24,  %25,  %26,  %27,  %28,  %29,  %30,  %31,  "
        " %32,  %33,  %34,  %35,  %36,  %37,  %38,  %39,  "
        " %40,  %41,  %42,  %43,  %44,  %45,  %46,  %47,  "
        " %48,  %49,  %50,  %51,  %52,  %53,  %54,  %55,  "
        " %56,  %57,  %58,  %59,  %60,  %61,  %62,  %63,  "
        " %64,  %65,  %66,  %67,  %68,  %69,  %70,  %71,  "
        " %72,  %73,  %74,  %75,  %76,  %77,  %78,  %79,  "
        " %80,  %81,  %82,  %83,  %84,  %85,  %86,  %87,  "
        " %88,  %89,  %90,  %91,  %92,  %93,  %94,  %95,  "
        " %96,  %97,  %98,  %99,  %100, %101, %102, %103,  "
        " %104, %105, %106, %107, %108, %109, %110, %111,  "
        " %112, %113, %114, %115, %116, %117, %118, %119,  "
        " %120, %121, %122, %123, %124, %125, %126, %127},"
        " [%128];\n"
        "}\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3]), "=f"(d[4]), "=f"(d[5]),
            "=f"(d[6]), "=f"(d[7]), "=f"(d[8]), "=f"(d[9]), "=f"(d[10]),
            "=f"(d[11]), "=f"(d[12]), "=f"(d[13]), "=f"(d[14]), "=f"(d[15]),
            "=f"(d[16]), "=f"(d[17]), "=f"(d[18]), "=f"(d[19]), "=f"(d[20]),
            "=f"(d[21]), "=f"(d[22]), "=f"(d[23]), "=f"(d[24]), "=f"(d[25]),
            "=f"(d[26]), "=f"(d[27]), "=f"(d[28]), "=f"(d[29]), "=f"(d[30]),
            "=f"(d[31]), "=f"(d[32]), "=f"(d[33]), "=f"(d[34]), "=f"(d[35]),
            "=f"(d[36]), "=f"(d[37]), "=f"(d[38]), "=f"(d[39]), "=f"(d[40]),
            "=f"(d[41]), "=f"(d[42]), "=f"(d[43]), "=f"(d[44]), "=f"(d[45]),
            "=f"(d[46]), "=f"(d[47]), "=f"(d[48]), "=f"(d[49]), "=f"(d[50]),
            "=f"(d[51]), "=f"(d[52]), "=f"(d[53]), "=f"(d[54]), "=f"(d[55]),
            "=f"(d[56]), "=f"(d[57]), "=f"(d[58]), "=f"(d[59]), "=f"(d[60]),
            "=f"(d[61]), "=f"(d[62]), "=f"(d[63]), "=f"(d[64]), "=f"(d[65]),
            "=f"(d[66]), "=f"(d[67]), "=f"(d[68]), "=f"(d[69]), "=f"(d[70]),
            "=f"(d[71]), "=f"(d[72]), "=f"(d[73]), "=f"(d[74]), "=f"(d[75]),
            "=f"(d[76]), "=f"(d[77]), "=f"(d[78]), "=f"(d[79]), "=f"(d[80]),
            "=f"(d[81]), "=f"(d[82]), "=f"(d[83]), "=f"(d[84]), "=f"(d[85]),
            "=f"(d[86]), "=f"(d[87]), "=f"(d[88]), "=f"(d[89]), "=f"(d[90]),
            "=f"(d[91]), "=f"(d[92]), "=f"(d[93]), "=f"(d[94]), "=f"(d[95]),
            "=f"(d[96]), "=f"(d[97]), "=f"(d[98]), "=f"(d[99]), "=f"(d[100]),
            "=f"(d[101]), "=f"(d[102]), "=f"(d[103]), "=f"(d[104]), "=f"(d[105]),
            "=f"(d[106]), "=f"(d[107]), "=f"(d[108]), "=f"(d[109]), "=f"(d[110]),
            "=f"(d[111]), "=f"(d[112]), "=f"(d[113]), "=f"(d[114]), "=f"(d[115]),
            "=f"(d[116]), "=f"(d[117]), "=f"(d[118]), "=f"(d[119]), "=f"(d[120]),
            "=f"(d[121]), "=f"(d[122]), "=f"(d[123]), "=f"(d[124]), "=f"(d[125]),
            "=f"(d[126]), "=f"(d[127])
        : "r"(tmem));
    }

    __device__ void tcgen05_wait_ld() {
    asm volatile("tcgen05.wait::ld.sync.aligned;");
    }

    __device__ void tcgen05_wait_st() {
    asm volatile("tcgen05.wait::st.sync.aligned;");
    }

    __device__ void tcgen05_cp(uint64_t desc, uint32_t tmem) {
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;\n"
                :
                : "r"(tmem), "l"(desc));
    }

    ////////////////////////////////////////////////////////////////////////////////
    // SHARED MEMORY DESCRIPTORS
    ////////////////////////////////////////////////////////////////////////////////

    enum wgmmaSwizzle {
    NO_SWIZZLE,
    SWIZZLE_128B_ATOM_32B,
    SWIZZLE_128B,
    SWIZZLE_64B,
    SWIZZLE_32B,
    };

    __device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) {
    return (((x) & 0x3FFFF) >> 0x4);
    }

    template <wgmmaSwizzle Swizzle>
    __device__ uint64_t make_smem_desc(char *ptr, uint64_t lbo, uint64_t sbo) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode(lbo) << 16;
    desc |= matrix_descriptor_encode(sbo) << 32;
    desc |= 1llu << 46;

    uint64_t swizzle_val;
    if constexpr (Swizzle == NO_SWIZZLE) {
        swizzle_val = 0llu;
    } else if constexpr (Swizzle == SWIZZLE_128B_ATOM_32B) {
        swizzle_val = 1llu;
    } else if constexpr (Swizzle == SWIZZLE_128B) {
        swizzle_val = 2llu;
    } else if constexpr (Swizzle == SWIZZLE_64B) {
        swizzle_val = 4llu;
    } else if constexpr (Swizzle == SWIZZLE_32B) {
        swizzle_val = 6llu;
    } else {
        static_assert(true, "Invalid wgmmaSwizzle value");
    }

    desc |= swizzle_val << 61;
    return desc;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // INSTRUCTION DESCRIPTORS
    ////////////////////////////////////////////////////////////////////////////////

    // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=tcgen05%2520mma#tcgen05-instuction-desc-kind-mxf4-mxf4nvf4

    template <int M, int N, int K, int NegateA, int NegateB>
    __device__ constexpr uint32_t make_inst_desc() {
    uint32_t desc = 0x00000000;
    desc |= 1lu << 7;
    desc |= 1lu << 10;
    desc |= uint32_t(NegateA) << 13;
    desc |= uint32_t(NegateB) << 14;
    desc |= ((N >> 3) & 0x3F) << 17;
    desc |= ((M >> 7) & 0x03) << 27;

    if constexpr (K == 64 || K == 128) {
        desc |= 0lu << 31;
    } else if constexpr (K == 96) {
        desc |= 1lu << 31;
    } else {
        static_assert(true, "Invalid K value");
    }

    return desc;
    }

    ////////////////////////////////////////////////////////////////////////////////
    // tcgen05 Intrinsic Calls
    ////////////////////////////////////////////////////////////////////////////////

    template <int ScaleD, uint32_t InstDesc>
    __device__ void tcgen05_mma(uint64_t desc_a, uint64_t desc_b, uint32_t tmem_d,
                                uint32_t tmem_sfa, uint32_t tmem_sfb) {
    asm volatile("{\n"
                "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::"
                "4X [%0], %1, %2, %3, [%4], [%5], %6;\n"
                "}\n"
                :
                : "r"(tmem_d), "l"(desc_a), "l"(desc_b), "r"(InstDesc),
                    "r"(tmem_sfa), "r"(tmem_sfb), "n"(int32_t(ScaleD)));
    }
    """,
]

CUDA_SRC = r"""
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

  LAUNCH(128, 64, 256, 4);

  return C;
}
"""

my_module = load_inline(
    name="nvfp4_dual_gemm",
    cpp_sources=[CPP_SRC],
    cuda_sources=CUDA_HEADERS + [CUDA_SRC],
    functions=["cuda_nvfp4_dual_gemm"],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-gencode=arch=compute_100a,code=sm_100a",
        "--expt-relaxed-constexpr",
    ],
    extra_ldflags=["-lcuda", "-lcublas"],
    verbose=True,
)


def custom_kernel(data: input_t) -> output_t:
    a, b1, b2, _, _, _, sfa_blocked, sfb1_blocked, sfb2_blocked, c = data
    return my_module.cuda_nvfp4_dual_gemm(
        a, b1, b2, sfa_blocked, sfb1_blocked, sfb2_blocked, c
    )
