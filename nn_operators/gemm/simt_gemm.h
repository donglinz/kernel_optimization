//
// Created by donglin on 5/22/22.
//

#ifndef KERNEL_OPTIMIZATION_SIMT_GEMM_H
#define KERNEL_OPTIMIZATION_SIMT_GEMM_H

#include "../../common/common.h"

constexpr int divide_up(const int &a, const int &b) {
    return (a + b - 1) / b;
}

constexpr int round_up_to(const int &a, const int &b) {
    return divide_up(a, b) * b;
}

namespace thread_mapping {
    template<
            typename ThreadBlockShape,
            typename WarpShape>
    struct ThreadBlockThreadMap {
        static const int warp_count_m = ThreadBlockShape::kM / WarpShape::kM;
        static const int warp_count_n = ThreadBlockShape::kN / WarpShape::kN;
    };

    template<typename WarpShape>
    struct WarpThreadMap {
        static const int thread_count_m = 8;
        static const int thread_count_n = 4;

        static_assert(WarpShape::kM % thread_count_m == 0, "");
        static_assert(WarpShape::kN & thread_count_n == 0, "");

        static const int element_per_thread_m = WarpShape::kM / thread_count_m;
        static const int element_per_thread_n = WarpShape::kN / thread_count_n;
    };
}

template<
        typename ElementA,
        typename ElementB,
        typename ElementC,
        typename ThreadBlockShape,
        typename WarpShape,
        int AlignmentA,
        int AlignmentB,
        int AlignmentC>
struct GemmKernelSimt {
    static_assert(ThreadBlockShape::kK % AlignmentA == 0, "");
    static_assert(ThreadBlockShape::kN % AlignmentB == 0, "");
    static_assert(ThreadBlockShape::kN % AlignmentC == 0, "");

    static const int smem_A_total_size_in_bytes = ThreadBlockShape::kMK * sizeof(ElementA) * 2;
    static const int smem_B_total_size_in_bytes = ThreadBlockShape::kKN * sizeof(ElementB) * 2;
    static const int smem_total_size_in_bytes = smem_A_total_size_in_bytes + smem_B_total_size_in_bytes;

    static const int warp_count_m = ThreadBlockShape::kK / WarpShape::kK;
    static const int warp_count_n = ThreadBlockShape::kN / WarpShape::kN;
    static const int warp_count = warp_count_m * warp_count_n;
    static const int thread_per_block = warp_count * 32;

    using AccessTypeA = AlignedArray<ElementA, AlignmentA>;
    using AccessTypeB = AlignedArray<ElementB, AlignmentB>;
    using AccessTypeC = AlignedArray<ElementC, AlignmentC>;
    using AccessTypeD = AccessTypeC;

    static_assert(ThreadBlockShape::kMK % thread_per_block == 0, "");
    static_assert(ThreadBlockShape::kKN % thread_per_block == 0, "");
    static_assert(ThreadBlockShape::kMN % thread_per_block == 0, "");

    using FragmentA = AlignedArray<ElementA, ThreadBlockShape::kMK / thread_per_block>;
    using FragmentB = AlignedArray<ElementB, ThreadBlockShape::kKN / thread_per_block>;
    using FragmentC = AlignedArray<ElementC, ThreadBlockShape::kMN / thread_per_block>;
    using FragmentD = FragmentC;

    using WarpThreadMap = thread_mapping::WarpThreadMap<WarpShape>;
    using BlockThreadMap = thread_mapping::ThreadBlockThreadMap<ThreadBlockShape, WarpShape>;

    using WarpFragmentA = AlignedArray<ElementA, WarpThreadMap::element_per_thread_m>;
    using WarpFragmentB = AlignedArray<ElementB, WarpThreadMap::element_per_thread_n>;

    __device__ __forceinline__
    void load_A_from_global(const shape::GemmCoord &problem_size, ElementA *ptr, FragmentA &frag_a) {
        #pragma unroll
        for (int idx = threadIdx.x; idx < ThreadBlockShape::kMK / AlignmentA; idx += thread_per_block) {
            int tb_m = idx / (ThreadBlockShape::kK / AlignmentA);
            int tb_n = (idx % ThreadBlockShape::kK) * AlignmentA;

            reinterpret_cast<AccessTypeA *>(&frag_a)[idx / thread_per_block] =
                    *reinterpret_cast<AccessTypeA *>(ptr + tb_m * problem_size.k() + tb_n);
        }
    }

    __device__ __forceinline__
    void load_B_from_global(const shape::GemmCoord &problem_size, ElementB *ptr, FragmentB &frag_b) {
        #pragma unroll
        for (int idx = threadIdx.x; idx < ThreadBlockShape::kKN / AlignmentB; idx += thread_per_block) {
            int tb_m = idx / (ThreadBlockShape::kN / AlignmentB);
            int tb_n = (idx / ThreadBlockShape::kN) * AlignmentB;

            reinterpret_cast<AccessTypeB *>(&frag_b)(idx / thread_per_block) =
                    *reinterpret_cast<AccessTypeB *>(ptr + tb_m * problem_size.n() + tb_n);
        }
    }

    __device__ __forceinline__
    void store_A_to_smem(const shape::GemmCoord &problem_size, ElementA *smem_ptr, const FragmentA &frag_a) {
        // note that A in smem is transposed
        #pragma unroll
        for (int idx = threadIdx.x; idx < ThreadBlockShape::kMK / AlignmentA; idx += thread_per_block) {

        }
    }

    __device__ __forceinline__
    void store_B_to_smem(const shape::GemmCoord &problem_size, ElementB *smem_ptr, const FragmentB &frag_b) {
        #pragma unroll
        for (int idx = threadIdx.x; idx < ThreadBlockShape::kKN / AlignmentB; idx += thread_per_block) {
            AccessTypeB &data = reinterpret_cast<AccessTypeB *>(&frag_b) + idx / thread_per_block;

        }
    }

    __device__ __forceinline__
    void load_warp_frag_A_from_smem(ElementA *smem_ptr, WarpFragmentA &warp_frag_a) {

    }

    __device__ __forceinline__
    void load_warp_frag_B_from_smem(ElementB *smem_ptr, WarpFragmentB &warp_frag_b) {

    }

    __device__ __forceinline__
    void warp_mma(WarpFragmentA &warp_frag_a, WarpFragmentB &warp_frag_b, FragmentC &accum) {

    }

    __device__ __forceinline__
    void gemm_main_loop(const shape::GemmCoord &problem_size, ElementA *ptr_A, ElementB *ptr_B, FragmentC &accum) {
        extern __shared__ uint8_t __cache[];

        // Transposed smem of A tile
        ElementA (*As)[ThreadBlockShape::kK][ThreadBlockShape::kM] =
                reinterpret_cast<ElementA (*)[ThreadBlockShape::kK][ThreadBlockShape::kM]>(__cache);
        ElementA (*Bs)[ThreadBlockShape::kK][ThreadBlockShape::kN] =
                reinterpret_cast<ElementA (*)[ThreadBlockShape::kK][ThreadBlockShape::kN]>(__cache) + smem_A_total_size_in_bytes;

        int gemm_main_loop_steps = (problem_size.k() + ThreadBlockShape::kK - 1) / ThreadBlockShape::kK;

        FragmentA frag_a;
        FragmentB frag_b;
        WarpFragmentA warp_frag_a[2];
        WarpFragmentB warp_frag_b[2];
        int ptr_A_offset = 0;
        int ptr_B_offset = 0;

        this->load_A_from_global(problem_size, ptr_A, frag_a);
        this->load_B_from_global(problem_size, ptr_B, frag_b);
        ptr_A_offset += ThreadBlockShape::kK;
        ptr_B_offset += ThreadBlockShape::kK * problem_size.n();

        this->store_A_to_smem(problem_size, As[0], frag_a);
        this->store_B_to_smem(problem_size, Bs[0], frag_b);

        const int lane_id = threadIdx.x % 32;
        const int warp_id = threadIdx.x / 32;

        #pragma unroll(1)
        for (int gemm_main_loop_id = 0; gemm_main_loop_id < gemm_main_loop_steps; ++gemm_main_loop_id) {
            int warp_id_m = warp_id / BlockThreadMap::warp_count_n;
            int warp_id_n = warp_id % BlockThreadMap::warp_count_n;

            // transposed smem A
            int smem_ptr_A_offset = warp_id_m * WarpShape::kM;
            int smem_ptr_B_offset = warp_id_n * WarpShape::kN;

            #pragma unroll
            for (int warp_loop_id = 0; warp_loop_id < ThreadBlockShape::kK; ++warp_loop_id) {
                if (warp_loop_id == ThreadBlockShape::kK - 1) {
                    this->store_A_to_smem(problem_size, As[(gemm_main_loop_id + 1) % 2], frag_a);
                    this->store_B_to_smem(problem_size, Bs[(gemm_main_loop_id + 1) % 2], frag_b);
                }

                this->load_warp_frag_A_from_smem(As[gemm_main_loop_id % 2] + smem_ptr_A_offset, warp_frag_a[(warp_loop_id + 1) % 2]);
                this->load_warp_frag_B_from_smem(Bs[gemm_main_loop_id % 2] + smem_ptr_B_offset, warp_frag_b[(warp_loop_id + 1) % 2]);
                smem_ptr_A_offset += ThreadBlockShape::kM;
                smem_ptr_B_offset += ThreadBlockShape::kN;

                if (warp_loop_id == 0) {
                    this->load_A_from_global(problem_size, ptr_A + ptr_A_offset, frag_a);
                    this->load_B_from_global(problem_size, ptr_B + ptr_B_offset, frag_b);
                    ptr_A_offset += ThreadBlockShape::kK;
                    ptr_B_offset += ThreadBlockShape::kK * problem_size.n();
                }

                this->warp_mma(warp_frag_a[warp_loop_id % 2], warp_frag_b[warp_loop_id % 2], accum);
            }
        }
    }

    __device__ __forceinline__
    void operator () (const shape::GemmCoord &problem_size, ElementA *ptr_A, ElementB *ptr_B, ElementC *ptr_C, ElementC *ptr_D) {
        const int tb_per_row = divide_up(problem_size.m(), ThreadBlockShape::kK);
        const int tb_per_column = divide_up(problem_size.n(), ThreadBlockShape::kN);

        const int block_offset_m = blockIdx.x / tb_per_column;
        const int block_offset_n = blockIdx.x % tb_per_column;

        FragmentC accum;
        gemm_main_loop(problem_size,
                       ptr_A + block_offset_m * ThreadBlockShape::kM * problem_size.k(),
                        ptr_B + block_offset_n * ThreadBlockShape::kN,
                        accum);
    }
};
#endif //KERNEL_OPTIMIZATION_SIMT_GEMM_H
