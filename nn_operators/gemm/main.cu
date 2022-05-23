//
// Created by dongl on 5/13/2022.
//
#include <random>

#include "gemm.h"
#include "../../common/common.h"
#include "../../common/tensor.h"

template<
        typename ElementA,
        typename ElementB,
        typename ElementC,
        typename ThreadBlockShape,
        typename WarpShape,
        int AlignmentA,
        int AlignmentB,
        int AlignmentC>
void benchmark(const shape::GemmCoord &problem_size) {
    using GemmKernel = GemmKernelSimt<
            ElementA,
            ElementB,
            ElementC,
            ThreadBlockShape,
            WarpShape,
            AlignmentA,
            AlignmentB,
            AlignmentC>;
    Tensor A(problem_size.mk());
    Tensor B(problem_size.kn());
    Tensor C(problem_size.mn());
    Tensor D(problem_size.mn());

    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::default_random_engine generator;
    std::normal_distribution<double> dist(0, 1);

    for (int idx = 0; idx < problem_size.mk(); ++idx) {
        A.template host_ref<ElementA>()[idx] = ElementA(dist(generator));
    }

    for (int idx = 0; idx < problem_size.kn(); ++idx) {
        B.template host_ref<ElementB>()[idx] = ElementB(dist(generator));
    }


    A.host_to_device_async(stream);
    B.host_to_device_async(stream);

    if (GemmKernel::smem_total_size_in_bytes >= (48 << 10)) {
//        checkCudaErrors(cudaFuncSetAttribute(
//                gemm<GemmKernel>,
//                cudaFuncAttributeMaxDynamicSharedMemorySize,
//                GemmKernel::smem_total_size_in_bytes));
    }

    int grid_dim = (problem_size.m() / ThreadBlockShape::kM) * (problem_size.n() / ThreadBlockShape::kN);

    // warp up
    for (int idx = 0; idx < 10; ++idx) {
        gemm<GemmKernel>
                <<<grid_dim, GemmKernel::thread_per_block, GemmKernel::smem_total_size_in_bytes, stream>>>
        (problem_size,
        A.template device_ref<ElementA>(),
        B.template device_ref<ElementB>(),
        C.template device_ref<ElementC>(),
        D.template device_ref<ElementC>());
    }

    checkCudaErrors(cudaStreamSynchronize(stream));

    cudaEvent_t _start, _stop;
    checkCudaErrors(cudaEventCreate(&_start));
    checkCudaErrors(cudaEventCreate(&_stop));

    float time_in_ms = 0;

    for (int idx = 0; idx < 10; ++idx) {
        A.host_to_device_async(stream);
        B.host_to_device_async(stream);

        checkCudaErrors(cudaEventRecord(_start, stream));

        gemm<GemmKernel>
        <<<grid_dim, GemmKernel::thread_per_block, GemmKernel::smem_total_size_in_bytes, stream>>>
                (problem_size,
                 A.template device_ref<ElementA>(),
                 B.template device_ref<ElementB>(),
                 C.template device_ref<ElementC>(),
                 D.template device_ref<ElementC>());

        checkCudaErrors(cudaEventRecord(_stop, stream));

        checkCudaErrors(cudaEventSynchronize(_stop));

        float ms;
        checkCudaErrors(cudaEventElapsedTime(&ms, _start, _stop));
        time_in_ms += ms;
    }

    checkCudaErrors(cudaStreamSynchronize(stream));

    printf("======Problem size %s block shape: %s warp shape %s======\n",
           problem_size.to_string().c_str(),
           ThreadBlockShape::to_string().c_str(),
           WarpShape::to_string().c_str());

    printf("Average time %.2fms\n", time_in_ms / 10);
}

int main () {
    benchmark<float, float, float, shape::GemmShape<128, 128, 8>, shape::GemmShape<64, 32, 8>, 4, 4, 4>({1024, 1024, 1024});
    return 0;
}