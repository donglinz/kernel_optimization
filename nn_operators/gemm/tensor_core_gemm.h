//
// Created by donglin on 5/23/22.
//

#ifndef KERNEL_OPTIMIZATION_TENSOR_CORE_GEMM_H
#define KERNEL_OPTIMIZATION_TENSOR_CORE_GEMM_H


#include "../../common/common.h"
#include "../../common/functors.h"

namespace {
    template<
            typename ThreadBlockShape,
            typename WarpShape>
    struct ThreadBlockThreadMap {
        static const int warp_count_m = ThreadBlockShape::kM / WarpShape::kM;
        static const int warp_count_n = ThreadBlockShape::kN / WarpShape::kN;
    };

    template<typename WarpShape,
            typename InstructionShape>
    struct WarpThreadMap {
        static const int steps_m = WarpShape::kM / InstructionShape::kM;
        static const int steps_n = WarpShape::kN / InstructionShape::kN;
        static const int steps_k = WarpShape::kK / InstructionShape::kK;
    };
}

template<int M, int N, int K>
struct MMA;

template<>
struct MMA<16, 8, 8> {
    __device__ __forceinline__
    void operator() (void *ptr_A, void *ptr_B, void *ptr_C, void *ptr_D) {
        AlignedArray<uint32_t, 2> &D = *reinterpret_cast<AlignedArray<uint32_t, 2> *>(ptr_D);
        AlignedArray<uint32_t, 2> &A = *reinterpret_cast<AlignedArray<uint32_t, 2> *>(ptr_A);
        uint32_t &B = *reinterpret_cast<uint32_t *>(ptr_B);
        AlignedArray<uint32_t, 2> &C = *reinterpret_cast<AlignedArray<uint32_t, 2> *>(ptr_C);

        asm volatile ("mma.sync.aligned.m16n8k8.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3}, {%4}, {%5, %6};\n"
                : "=r"(D[0]), "=r"(D[1])
                : "r"(A[0]), "r"(A[1]),
                "r"(B),
                "r"(C[0]), "r"(C[1]));
    }
};

template<int N, bool transpose>
struct LDMatrix;

template<>
struct LDMatrix<1, true> {
    __device__ __forceinline__
    void operator() (void *data_ptr, void *smem_ptr) {
        uint32_t &data = *reinterpret_cast<uint32_t *>(data_ptr);

        asm volatile ("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n"
                : "=r"(data)
                : "r"(static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr))));
    }
};

template<>
struct LDMatrix<1, false> {
    __device__ __forceinline__
    void operator() (void *data_ptr, void *smem_ptr) {
        uint32_t &data = *reinterpret_cast<uint32_t *>(data_ptr);

        asm volatile ("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
                : "=r"(data)
                : "r"(static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr))));
    }
};

template<>
struct LDMatrix<2, true> {
    __device__ __forceinline__
    void operator() (void *data_ptr, void *smem_ptr) {
        AlignedArray<uint32_t, 2> &data = *reinterpret_cast<AlignedArray<uint32_t, 2> *>(data_ptr);

        asm volatile ("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"
                : "=r"(data[0]), "=r"(data[1])
                : "r"(static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr))));
    }
};

template<>
struct LDMatrix<2, false> {
    __device__ __forceinline__
    void operator() (void *data_ptr, void *smem_ptr) {
        AlignedArray<uint32_t, 2> &data = *reinterpret_cast<AlignedArray<uint32_t, 2> *>(data_ptr);

        asm volatile ("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
                : "=r"(data[0]), "=r"(data[1])
                : "r"(static_cast<unsigned>(__cvta_generic_to_shared(smem_ptr))));
    }
};

template<
        typename _ElementA,
        typename _ElementB,
        typename _ElementC,
        typename _ThreadBlockShape,
        typename _WarpShape,
        typename InstructionShape,
        int AlignmentA,
        int AlignmentB,
        int AlignmentC>
struct GemmKernelTensorCore {
    using ElementA = _ElementA;
    using ElementB = _ElementB;
    using ElementC = _ElementC;
    using ThreadBlockShape = _ThreadBlockShape;
    using WarpShape = _WarpShape;

    static_assert(ThreadBlockShape::kK % AlignmentA == 0, "");
    static_assert(ThreadBlockShape::kN % AlignmentB == 0, "");
    static_assert(ThreadBlockShape::kN % AlignmentC == 0, "");

    static const int smem_A_total_size_in_bytes = ThreadBlockShape::kMK * sizeof(ElementA);
    static const int smem_B_total_size_in_bytes = ThreadBlockShape::kKN * sizeof(ElementB);

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

    using WarpThreadMap = ::WarpThreadMap<WarpShape, InstructionShape>;
    using BlockThreadMap = ::ThreadBlockThreadMap<ThreadBlockShape, WarpShape>;

    using WarpFragmentA = AlignedArray<uint32_t, 2>;
    using WarpFragmentB = AlignedArray<uint32_t, 1>;
    using WarpFragmentC = AlignedArray<float, 4>;
    using WarpFragmentD = AlignedArray<float, 4>;
    using WarpAggregateFragmentA = AlignedArray<uint32_t, WarpFragmentA::kElements * WarpThreadMap::steps_m>;
    using WarpAggregateFragmentB = AlignedArray<uint32_t, WarpFragmentB::kElements * WarpThreadMap::steps_n>;

    static_assert(FragmentC::kElements == WarpThreadMap::steps_m * WarpThreadMap::steps_n * WarpFragmentC::kElements, "");

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
        #pragma unroll
        for (int idx = threadIdx.x; idx < ThreadBlockShape::kMK / AlignmentA; idx += thread_per_block) {
            const AccessTypeA &data = reinterpret_cast<const AccessTypeA *>(&frag_a)[idx / thread_per_block];
            reinterpret_cast<AccessTypeA *>(smem_ptr)[idx] = data;
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
        const int lane_id = threadIdx.x % 32;

        #pragma unroll
        for (int idx_m = 0; idx_m < WarpThreadMap::steps_m; ++idx_m) {
            WarpFragmentA &warp_frag_a = reinterpret_cast<WarpFragmentA *>(&warp_aggregate_frag_a)[idx_m];
            LDMatrix<2, false>()(&warp_frag_a, smem_ptr + ((lane_id % 16) + idx_m * InstructionShape::kM) * ThreadBlockShape::kK);
        }
    }

    __device__ __forceinline__
    void load_warp_aggregate_frag_B_from_smem(ElementB *smem_ptr, WarpAggregateFragmentB &warp_aggregate_frag_b) {
        const int lane_id = threadIdx.x % 32;

        #pragma unroll
        for (int idx_n = 0; idx_n < WarpThreadMap::steps_n; ++idx_n) {
            WarpFragmentB &warp_frag_b = reinterpret_cast<WarpFragmentB *>(&warp_aggregate_frag_b)[idx_n];
            LDMatrix<1, true>()(&warp_frag_b, smem_ptr + ((lane_id % 8) + idx_n * InstructionShape::kN));
        }
    }

    __device__ __forceinline__
    void warp_mma(WarpAggregateFragmentA &warp_aggregate_frag_a, WarpAggregateFragmentB &warp_aggregate_frag_b, FragmentC &accum) {
        #pragma unroll
        for (int idx_m = 0; idx_m < WarpThreadMap::steps_m; ++idx_m) {
            WarpFragmentA &warp_frag_a = reinterpret_cast<WarpFragmentA *>(&warp_aggregate_frag_a)[idx_m];
            #pragma unroll
            for (int idx_n = 0; idx_n < WarpThreadMap::steps_n; ++idx_n) {
                WarpFragmentB &warp_frag_b = reinterpret_cast<WarpFragmentB *>(&warp_aggregate_frag_b)[idx_n];
                WarpFragmentC &warp_frag_accum = reinterpret_cast<WarpFragmentC *>(&accum)[idx_m * WarpThreadMap::steps_n + idx_n];
                MMA<InstructionShape::kM, InstructionShape::kN, InstructionShape::kK>()(&warp_frag_a, &warp_frag_b, &warp_frag_accum, &warp_frag_accum);
            }
        }
    }

    __device__ __forceinline__
    void gemm_main_loop(const shape::GemmCoord &problem_size, ElementA *ptr_A, ElementB *ptr_B, FragmentC &accum) {
        extern __shared__ uint8_t __cache[];

        // Transposed smem of A tile
        ElementA (*As)[ThreadBlockShape::kM][ThreadBlockShape::kK] =
                reinterpret_cast<ElementA (*)[ThreadBlockShape::kM][ThreadBlockShape::kK]>(__cache);
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

        const int warp_id = threadIdx.x / 32;

        #pragma unroll(1)
        for (int gemm_main_loop_id = 0; gemm_main_loop_id < gemm_main_loop_steps; ++gemm_main_loop_id) {
            const int warp_id_m = warp_id / BlockThreadMap::warp_count_n;
            const int warp_id_n = warp_id % BlockThreadMap::warp_count_n;

            int smem_ptr_A_offset = warp_id_m * WarpShape::kM * ThreadBlockShape::kK;
            int smem_ptr_B_offset = warp_id_n * WarpShape::kN;

            __syncthreads();

            this->load_warp_aggregate_frag_A_from_smem(reinterpret_cast<ElementA *>(As[gemm_main_loop_id % 2]) + smem_ptr_A_offset, warp_aggregate_frag_a[0]);
            this->load_warp_aggregate_frag_B_from_smem(reinterpret_cast<ElementB *>(Bs[gemm_main_loop_id % 2]) + smem_ptr_B_offset, warp_aggregate_frag_b[0]);
//            smem_ptr_A_offset += ThreadBlockShape::kM;
            smem_ptr_A_offset += InstructionShape::kK;
            smem_ptr_B_offset += InstructionShape::kK * ThreadBlockShape::kN;

            int warp_loop_steps = ThreadBlockShape::kK / InstructionShape::kK;
            #pragma unroll
            for (int warp_loop_id = 0; warp_loop_id < warp_loop_steps; ++warp_loop_id) {
                if (warp_loop_id == warp_loop_steps - 1) { // register->shared memory
                    this->store_A_to_smem(problem_size, reinterpret_cast<ElementA *>(As[(gemm_main_loop_id + 1) % 2]), frag_a);
                    this->store_B_to_smem(problem_size, reinterpret_cast<ElementB *>(Bs[(gemm_main_loop_id + 1) % 2]), frag_b);
                }

                this->load_warp_aggregate_frag_A_from_smem(reinterpret_cast<ElementA *>(As[gemm_main_loop_id % 2]) + smem_ptr_A_offset, warp_aggregate_frag_a[(warp_loop_id + 1) % 2]);
                this->load_warp_aggregate_frag_B_from_smem(reinterpret_cast<ElementB *>(Bs[gemm_main_loop_id % 2]) + smem_ptr_B_offset, warp_aggregate_frag_b[(warp_loop_id + 1) % 2]);

                smem_ptr_A_offset += InstructionShape::kK;
                smem_ptr_B_offset += InstructionShape::kK * ThreadBlockShape::kN;

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

        FragmentC accum = ElementC(0);
        gemm_main_loop(problem_size,
                       ptr_A + block_id_m * ThreadBlockShape::kM * problem_size.k(),
                       ptr_B + block_id_n * ThreadBlockShape::kN,
                       accum);

        const int warp_id = threadIdx.x / 32;

        const int warp_id_m = warp_id / BlockThreadMap::warp_count_n;
        const int warp_id_n = warp_id % BlockThreadMap::warp_count_n;

        
    }
};

#endif //KERNEL_OPTIMIZATION_TENSOR_CORE_GEMM_H
