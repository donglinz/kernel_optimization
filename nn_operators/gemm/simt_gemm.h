//
// Created by donglin on 5/22/22.
//

#ifndef KERNEL_OPTIMIZATION_SIMT_GEMM_H
#define KERNEL_OPTIMIZATION_SIMT_GEMM_H

#include "../../common/common.h"
#include "../../common/functors.h"

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
        static const int thread_count_m = 4;
        static const int thread_count_n = 8;

        static const int element_per_thread_m = 4;
        static const int element_per_thread_n = 4;

        static_assert(WarpShape::kM % (thread_count_m * element_per_thread_m) == 0, "");
        static_assert(WarpShape::kN % (thread_count_n * element_per_thread_n) == 0, "");

        using WarpStepShape = shape::MatrixShape<thread_count_m * element_per_thread_m, thread_count_n * element_per_thread_n>;

        static const int steps_m = WarpShape::kM / WarpStepShape::kM;
        static const int steps_n = WarpShape::kN / WarpStepShape::kN;

        __device__
        static void print() {
            printf("warp shape %d:%d, thread count %d:%d element_per_thread %d:%d warp step shape %d:%d steps %d:%d\n", WarpShape::kM, WarpShape::kN, thread_count_m, thread_count_n, element_per_thread_m, element_per_thread_n,
                   WarpStepShape::kM, WarpStepShape::kN, steps_m, steps_n);
        }
    };
}

template<
        typename _ElementA,
        typename _ElementB,
        typename _ElementC,
        typename _ThreadBlockShape,
        typename _WarpShape,
        int AlignmentA,
        int AlignmentB,
        int AlignmentC>
struct GemmKernelSimt {
    using ElementA = _ElementA;
    using ElementB = _ElementB;
    using ElementC = _ElementC;
    using ThreadBlockShape = _ThreadBlockShape;
    using WarpShape = _WarpShape;

    static_assert(ThreadBlockShape::kK % AlignmentA == 0, "");
    static_assert(ThreadBlockShape::kN % AlignmentB == 0, "");
    static_assert(ThreadBlockShape::kN % AlignmentC == 0, "");

    static const int smem_A_total_size_in_bytes = ThreadBlockShape::kM * (ThreadBlockShape::kK + 1) * sizeof(ElementA) * 2;
    static const int smem_B_total_size_in_bytes = ThreadBlockShape::kKN * sizeof(ElementB) * 2;

    static const int smem_total_size_in_bytes = smem_A_total_size_in_bytes + smem_B_total_size_in_bytes;

    static const int warp_count_m = ThreadBlockShape::kM / WarpShape::kM;
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

    static_assert(WarpThreadMap::element_per_thread_m * WarpThreadMap::element_per_thread_n * WarpThreadMap::steps_m * WarpThreadMap::steps_n == FragmentC::kElements, "");

    static_assert(WarpThreadMap::element_per_thread_m % AccessTypeA::kElements == 0, "");
    static_assert(WarpThreadMap::element_per_thread_n % AccessTypeB::kElements == 0, "");

    using WarpFragmentA = AlignedArray<ElementA, WarpThreadMap::element_per_thread_m>;
    using WarpFragmentB = AlignedArray<ElementB, WarpThreadMap::element_per_thread_n>;
    using WarpAggregateFragmentA = AlignedArray<ElementA, WarpThreadMap::element_per_thread_m * WarpThreadMap::steps_m>;
    using WarpAggregateFragmentB = AlignedArray<ElementA, WarpThreadMap::element_per_thread_n * WarpThreadMap::steps_n>;

    using SmemAPointerType = ElementA (*)[ThreadBlockShape::kK][ThreadBlockShape::kM];
    using SmemBPointerType = ElementB (*)[ThreadBlockShape::kK][ThreadBlockShape::kN];

    __device__ __forceinline__
    void load_A_from_global(const shape::GemmCoord &problem_size, ElementA *ptr, FragmentA &frag_a) {
        #pragma unroll
        for (int idx = threadIdx.x; idx < ThreadBlockShape::kMK / AlignmentA; idx += thread_per_block) {
            int tb_m = idx / (ThreadBlockShape::kK / AlignmentA);
            int tb_n = (idx % (ThreadBlockShape::kK / AlignmentA)) * AlignmentA;

            reinterpret_cast<AccessTypeA *>(&frag_a)[idx / thread_per_block] =
                    *reinterpret_cast<AccessTypeA *>(ptr + tb_m * problem_size.k() + tb_n);
        }
    }

    __device__ __forceinline__
    void load_B_from_global(const shape::GemmCoord &problem_size, ElementB *ptr, FragmentB &frag_b) {
        #pragma unroll
        for (int idx = threadIdx.x; idx < ThreadBlockShape::kKN / AlignmentB; idx += thread_per_block) {
            int tb_m = idx / (ThreadBlockShape::kN / AlignmentB);
            int tb_n = (idx % (ThreadBlockShape::kN / AlignmentB)) * AlignmentB;

            reinterpret_cast<AccessTypeB *>(&frag_b)[idx / thread_per_block] =
                    *reinterpret_cast<AccessTypeB *>(ptr + tb_m * problem_size.n() + tb_n);
        }
    }

    __device__ __forceinline__
    void store_A_to_smem(const shape::GemmCoord &problem_size, ElementA *smem_ptr, const FragmentA &frag_a) {
        ElementA (*smem_array_ptr)[ThreadBlockShape::kK + 1] =
                reinterpret_cast<ElementA (*)[ThreadBlockShape::kK + 1]>(smem_ptr);

//        // Note that A in smem is transposed
//        #pragma unroll
//        for (int idx = threadIdx.x; idx < ThreadBlockShape::kMK / AlignmentA; idx += thread_per_block) {
//            const AccessTypeA &data = reinterpret_cast<const AccessTypeA *>(&frag_a)[idx / thread_per_block];
//
//            int tb_m = idx / (ThreadBlockShape::kK / AlignmentA);
//            int tb_n = (idx % (ThreadBlockShape::kK / AlignmentA)) * AlignmentA;
//
//            #pragma unroll
//            for (int offset = 0; offset < AccessTypeA::kElements; ++offset) {
//                smem_array_ptr[tb_n + offset][tb_m] = data[offset];
//            }
//        }
        #pragma unroll
        for (int idx = threadIdx.x; idx < ThreadBlockShape::kMK / AlignmentA; idx += thread_per_block) {
            const AccessTypeA &data = reinterpret_cast<const AccessTypeA *>(&frag_a)[idx / thread_per_block];
            int idx_m = idx / (ThreadBlockShape::kK / AlignmentA);
            int idx_n = (idx % (ThreadBlockShape::kK / AlignmentA)) * AlignmentA;
            #pragma unroll
            for (int offset = 0; offset < AlignmentA; ++offset) {
                smem_array_ptr[idx_m][idx_n + offset] = data[offset];
            }
        }
    }

    __device__ __forceinline__
    void store_B_to_smem(const shape::GemmCoord &problem_size, ElementB *smem_ptr, const FragmentB &frag_b) {
        #pragma unroll
        for (int idx = threadIdx.x; idx < ThreadBlockShape::kKN / AlignmentB; idx += thread_per_block) {
            const AccessTypeB &data = reinterpret_cast<const AccessTypeB *>(&frag_b)[idx / thread_per_block];
            reinterpret_cast<AccessTypeB *>(smem_ptr)[idx] = data;
        }
    }

    __device__ __forceinline__
    void load_warp_aggregate_frag_A_from_smem(ElementA *smem_ptr, WarpAggregateFragmentA &warp_aggregate_frag_a) {
        ElementA (*smem_array_ptr)[ThreadBlockShape::kK + 1] =
                reinterpret_cast<ElementA (*)[ThreadBlockShape::kK + 1]>(smem_ptr);

        #pragma unroll
        for (int idx_m = 0; idx_m < WarpThreadMap::steps_m; ++idx_m) {
            WarpFragmentA &warp_frag_a = reinterpret_cast<WarpFragmentA *>(&warp_aggregate_frag_a)[idx_m];
            #pragma unroll
            for (int idx = 0; idx < WarpFragmentA::kElements / AccessTypeA::kElements; ++idx) {
                #pragma unroll
                for (int offset = 0; offset < AccessTypeA::kElements; ++offset) {
                    reinterpret_cast<AccessTypeA *>(&warp_frag_a)[idx][offset] = smem_array_ptr[idx_m * WarpThreadMap::WarpStepShape::kM + idx * AccessTypeA::kElements + offset][0];
                }
            }
        }
//        #pragma unroll
//        for (int idx = 0; idx < WarpFragmentA::kElements / AccessTypeA::kElements; ++idx) {
//            reinterpret_cast<AccessTypeA *>(&warp_frag_a)[idx] = reinterpret_cast<AccessTypeA *>(smem_ptr)[idx];
//        }
    }

    __device__ __forceinline__
    void load_warp_aggregate_frag_B_from_smem(ElementB *smem_ptr, WarpAggregateFragmentB &warp_aggregate_frag_b) {
        #pragma unroll
        for (int idx_n = 0; idx_n < WarpThreadMap::steps_n; ++idx_n) {
            WarpFragmentB &warp_frag_b = reinterpret_cast<WarpFragmentB *>(&warp_aggregate_frag_b)[idx_n];
            #pragma unroll
            for (int idx = 0; idx < WarpFragmentB::kElements / AccessTypeB::kElements; ++idx) {
                reinterpret_cast<AccessTypeB *>(&warp_frag_b)[idx] = reinterpret_cast<AccessTypeB *>(smem_ptr + idx_n * WarpThreadMap::WarpStepShape::kN)[idx];
            }
        }
//        #pragma unroll
//        for (int idx = 0; idx < WarpFragmentB::kElements / AccessTypeB::kElements; ++idx) {
//            reinterpret_cast<AccessTypeB *>(&warp_frag_b)[idx] = reinterpret_cast<AccessTypeB *>(smem_ptr)[idx];
//        }
    }

//    __device__ __forceinline__
//    void warp_mma(WarpFragmentA &warp_frag_a, WarpFragmentB &warp_frag_b, FragmentC &accum) {
//        #pragma unroll
//        for (int idx_m = 0; idx_m < WarpFragmentA::kElements; ++idx_m) {
//            #pragma unroll
//            for (int idx_n = 0; idx_n < WarpFragmentB::kElements; ++idx_n) {
//                accum[idx_m * WarpFragmentB::kElements + idx_n] += warp_frag_a[idx_m] * warp_frag_b[idx_n];
//            }
//        }
//    }
    __device__ __forceinline__
    void warp_mma(WarpAggregateFragmentA &warp_aggregate_frag_a, WarpAggregateFragmentB &warp_aggregate_frag_b, FragmentC &accum) {
        #pragma unroll
        for (int idx_m = 0; idx_m < WarpThreadMap::steps_m; ++idx_m) {
            WarpFragmentA &warp_frag_a = reinterpret_cast<WarpFragmentA *>(&warp_aggregate_frag_a)[idx_m];
            #pragma unroll
            for (int idx_n = 0; idx_n < WarpThreadMap::steps_n; ++idx_n) {
                WarpFragmentB &warp_frag_b = reinterpret_cast<WarpFragmentB *>(&warp_aggregate_frag_b)[idx_n];
                #pragma unroll
                for (int offset_m = 0; offset_m < WarpFragmentA::kElements; ++offset_m) {
                    #pragma unroll
                    for (int offset_n = 0; offset_n < WarpFragmentB::kElements; ++offset_n) {
                        accum[
                                idx_m * WarpThreadMap::element_per_thread_m * WarpThreadMap::element_per_thread_n * WarpThreadMap::steps_n +
                                idx_n * WarpThreadMap::element_per_thread_n +
                                offset_m * WarpThreadMap::element_per_thread_n * WarpThreadMap::steps_n +
                                offset_n
                                ] += warp_frag_a[offset_m] * warp_frag_b[offset_n];
                    }
                }
            }
        }
    }

    __device__ __forceinline__
    void gemm_main_loop(const shape::GemmCoord &problem_size, ElementA *ptr_A, ElementB *ptr_B, FragmentC &accum) {
        extern __shared__ uint8_t __cache[];

        // Transposed smem of A tile
        ElementA (*As)[ThreadBlockShape::kM][ThreadBlockShape::kK + 1] =
                reinterpret_cast<ElementA (*)[ThreadBlockShape::kM][ThreadBlockShape::kK + 1]>(__cache);
        ElementB (*Bs)[ThreadBlockShape::kK][ThreadBlockShape::kN] =
                reinterpret_cast<ElementB (*)[ThreadBlockShape::kK][ThreadBlockShape::kN]>(__cache + smem_A_total_size_in_bytes);

        int gemm_main_loop_steps = (problem_size.k() + ThreadBlockShape::kK - 1) / ThreadBlockShape::kK;

        FragmentA frag_a;
        FragmentB frag_b;
        WarpAggregateFragmentA warp_aggregate_frag_a[2];
        WarpAggregateFragmentB warp_aggregate_frag_b[2];
        int ptr_A_offset = 0;
        int ptr_B_offset = 0;

        this->load_A_from_global(problem_size, ptr_A, frag_a);
        this->load_B_from_global(problem_size, ptr_B, frag_b);
        ptr_A_offset += ThreadBlockShape::kK;
        ptr_B_offset += ThreadBlockShape::kK * problem_size.n();

        this->store_A_to_smem(problem_size, reinterpret_cast<ElementA *>(As[0]), frag_a);
        this->store_B_to_smem(problem_size, reinterpret_cast<ElementB *>(Bs[0]), frag_b);

        const int lane_id = threadIdx.x % 32;
        const int warp_id = threadIdx.x / 32;

        #pragma unroll(1)
        for (int gemm_main_loop_id = 0; gemm_main_loop_id < gemm_main_loop_steps; ++gemm_main_loop_id) {
            const int warp_id_m = warp_id / BlockThreadMap::warp_count_n;
            const int warp_id_n = warp_id % BlockThreadMap::warp_count_n;
            const int lane_id_m = lane_id / WarpThreadMap::thread_count_n;
            const int lane_id_n = lane_id % WarpThreadMap::thread_count_n;

//            // transposed smem A
//            int smem_ptr_A_offset = warp_id_m * WarpShape::kM + lane_id_m * WarpThreadMap::element_per_thread_m;
            int smem_ptr_A_offset = (warp_id_m * WarpShape::kM + lane_id_m * WarpThreadMap::element_per_thread_m) * (ThreadBlockShape::kK + 1);
            int smem_ptr_B_offset = warp_id_n * WarpShape::kN + lane_id_n * WarpThreadMap::element_per_thread_n;

            __syncthreads();

            this->load_warp_aggregate_frag_A_from_smem(reinterpret_cast<ElementA *>(As[gemm_main_loop_id % 2]) + smem_ptr_A_offset, warp_aggregate_frag_a[0]);
            this->load_warp_aggregate_frag_B_from_smem(reinterpret_cast<ElementB *>(Bs[gemm_main_loop_id % 2]) + smem_ptr_B_offset, warp_aggregate_frag_b[0]);
//            smem_ptr_A_offset += ThreadBlockShape::kM;
            smem_ptr_A_offset += 1;
            smem_ptr_B_offset += ThreadBlockShape::kN;

            #pragma unroll
	        for (int warp_loop_id = 0; warp_loop_id < ThreadBlockShape::kK; ++warp_loop_id) {
                if (warp_loop_id == ThreadBlockShape::kK - 1) { // register->shared memory
                    this->store_A_to_smem(problem_size, reinterpret_cast<ElementA *>(As[(gemm_main_loop_id + 1) % 2]), frag_a);
                    this->store_B_to_smem(problem_size, reinterpret_cast<ElementB *>(Bs[(gemm_main_loop_id + 1) % 2]), frag_b);
                }

                this->load_warp_aggregate_frag_A_from_smem(reinterpret_cast<ElementA *>(As[gemm_main_loop_id % 2]) + smem_ptr_A_offset, warp_aggregate_frag_a[(warp_loop_id + 1) % 2]);
                this->load_warp_aggregate_frag_B_from_smem(reinterpret_cast<ElementB *>(Bs[gemm_main_loop_id % 2]) + smem_ptr_B_offset, warp_aggregate_frag_b[(warp_loop_id + 1) % 2]);
//                smem_ptr_A_offset += ThreadBlockShape::kM;
                smem_ptr_A_offset += 1;
                smem_ptr_B_offset += ThreadBlockShape::kN;

                if (warp_loop_id == 0) { // global -> register
                    this->load_A_from_global(problem_size, ptr_A + ptr_A_offset, frag_a);
                    this->load_B_from_global(problem_size, ptr_B + ptr_B_offset, frag_b);
                    ptr_A_offset += ThreadBlockShape::kK;
                    ptr_B_offset += ThreadBlockShape::kK * problem_size.n();
                }
 
                this->warp_mma(warp_aggregate_frag_a[warp_loop_id % 2], warp_aggregate_frag_b[warp_loop_id % 2], accum);
            }
        }
    }

    __device__ __forceinline__
    void operator () (const shape::GemmCoord &problem_size, ElementA *ptr_A, ElementB *ptr_B, ElementC *ptr_C, ElementC *ptr_D) {
        const int tb_per_row = (problem_size.m() + ThreadBlockShape::kM - 1) / ThreadBlockShape::kM;
        const int tb_per_column = (problem_size.n() + ThreadBlockShape::kN - 1) / ThreadBlockShape::kN;

        const int block_id_m = blockIdx.x / tb_per_column;
        const int block_id_n = blockIdx.x % tb_per_column;

        FragmentC accum = 0;
        gemm_main_loop(problem_size,
                       ptr_A + block_id_m * ThreadBlockShape::kM * problem_size.k(),
                        ptr_B + block_id_n * ThreadBlockShape::kN,
                        accum);

        const int lane_id = threadIdx.x % 32;
        const int warp_id = threadIdx.x / 32;

        const int warp_id_m = warp_id / BlockThreadMap::warp_count_n;
        const int warp_id_n = warp_id % BlockThreadMap::warp_count_n;
        const int lane_id_m = lane_id / WarpThreadMap::thread_count_n;
        const int lane_id_n = lane_id % WarpThreadMap::thread_count_n;
        const int thread_offset_m = block_id_m * ThreadBlockShape::kM + warp_id_m * WarpShape::kM + lane_id_m * WarpThreadMap::element_per_thread_m;
        const int thread_offset_n = block_id_n * ThreadBlockShape::kN + warp_id_n * WarpShape::kN + lane_id_n * WarpThreadMap::element_per_thread_n;
        using AccumulatorAccessType = AlignedArray<ElementC, WarpThreadMap::element_per_thread_m * WarpThreadMap::element_per_thread_n>;

        if (ptr_C) {
            #pragma unroll
            for (int warp_step_id_m = 0; warp_step_id_m < WarpThreadMap::steps_m; ++warp_step_id_m) { // 4
                int warp_step_id_m_offset = warp_step_id_m * WarpThreadMap::WarpStepShape::kM;
                #pragma unroll
                for (int warp_step_id_n = 0; warp_step_id_n < WarpThreadMap::steps_n; ++warp_step_id_n) { // 1
                    int warp_step_id_n_offset = warp_step_id_n * WarpThreadMap::WarpStepShape::kN;
                    AccumulatorAccessType source_accum;
                    #pragma unroll
                    for (int element_id_m = 0; element_id_m < WarpThreadMap::element_per_thread_m; ++element_id_m) { // 4
                        #pragma unroll
                        for (int element_id_n = 0; element_id_n < WarpThreadMap::element_per_thread_n; element_id_n += AlignmentC) { // 4
                            *reinterpret_cast<AccessTypeC *>(&source_accum[element_id_m * WarpThreadMap::element_per_thread_n + element_id_n])
                             = *reinterpret_cast<AccessTypeC *>(ptr_C + (thread_offset_m + warp_step_id_m_offset + element_id_m) * problem_size.n() +
                                    (thread_offset_n + warp_step_id_n_offset + element_id_n));
                        }
                    }

                    #pragma unroll
                    for (int element_id_m = 0; element_id_m < WarpThreadMap::element_per_thread_m; ++element_id_m) {
                        #pragma unroll
                        for (int element_id_n = 0; element_id_n < WarpThreadMap::element_per_thread_n; element_id_n += AlignmentC) {
                            AccessTypeC &source_accum_ref = *reinterpret_cast<AccessTypeC *>(&source_accum[element_id_m * WarpThreadMap::element_per_thread_n + element_id_n]);
                            AccessTypeC &accum_ref = *reinterpret_cast<AccessTypeC *>(&accum[(warp_step_id_m * WarpThreadMap::steps_n + warp_step_id_n) * AccumulatorAccessType::kElements + element_id_m * WarpThreadMap::element_per_thread_n + element_id_n]);
                            *reinterpret_cast<AccessTypeC *>(ptr_D + (thread_offset_m + warp_step_id_m_offset + element_id_m) * problem_size.n() +
                                                                       (thread_offset_n + warp_step_id_n_offset + element_id_n)) = Add<AccessTypeC>()(accum_ref, source_accum_ref);
                        }
                    }
                }
            }
        } else {
            #pragma unroll
            for (int warp_step_id_m = 0; warp_step_id_m < WarpThreadMap::steps_m; ++warp_step_id_m) {
                int warp_step_id_m_offset = warp_step_id_m * WarpThreadMap::WarpStepShape::kM;
                #pragma unroll
                for (int warp_step_id_n = 0; warp_step_id_n < WarpThreadMap::steps_n; ++warp_step_id_n) {
                    int warp_step_id_n_offset = warp_step_id_n * WarpThreadMap::WarpStepShape::kN;
                    #pragma unroll
                    for (int element_id_m = 0; element_id_m < WarpThreadMap::element_per_thread_m; ++element_id_m) {
                        #pragma unroll
                        for (int element_id_n = 0; element_id_n < WarpThreadMap::element_per_thread_n; element_id_n += AlignmentC) {
                            AccessTypeC &accum_ref = *reinterpret_cast<AccessTypeC *>(&accum[(warp_step_id_m * WarpThreadMap::steps_n + warp_step_id_n) * AccumulatorAccessType::kElements + element_id_m * WarpThreadMap::element_per_thread_n + element_id_n]);
                            *reinterpret_cast<AccessTypeC *>(ptr_D + (thread_offset_m + warp_step_id_m_offset + element_id_m) * problem_size.n() +
                                                             (thread_offset_n + warp_step_id_n_offset + element_id_n)) = accum_ref;
                        }
                    }
                }
            }
        }
    }
};


#endif //KERNEL_OPTIMIZATION_SIMT_GEMM_H
