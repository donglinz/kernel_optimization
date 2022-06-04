//
// Created by dongl on 5/13/2022.
//
#include <random>
#include <cublas_v2.h>

#include "gemm.h"
#include "../../common/common.h"
#include "../../common/tensor.h"

__global__
void empty_kernel() {

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
void benchmark(const shape::GemmCoord &problem_size) {
    empty_kernel<<<10, 128>>>();
    using GemmKernel = GemmKernelSimt<
            ElementA,
            ElementB,
            ElementC,
            ThreadBlockShape,
            WarpShape,
            AlignmentA,
            AlignmentB,
            AlignmentC>;

    Tensor A(problem_size.mk() * sizeof(ElementA));
    Tensor B(problem_size.kn() * sizeof(ElementB));
    Tensor C(problem_size.mn() * sizeof(ElementC));
    Tensor C_transpose(problem_size.mn() * sizeof(ElementC));
    Tensor D(problem_size.mn() * sizeof(ElementC));

    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::default_random_engine generator;
    std::normal_distribution<double> dist(0, 1);
    for (int idx = 0; idx < problem_size.mk(); ++idx) {
        A.template host_ref<ElementA>()[idx] = ElementA(dist(generator));
//        A.template host_ref<ElementA>()[idx] = ElementA(1);
    }

    for (int idx = 0; idx < problem_size.kn(); ++idx) {
        B.template host_ref<ElementB>()[idx] = ElementB(dist(generator));
//        int r = idx / problem_size.n();
//        int c = idx % problem_size.n();
//        B.template host_ref<ElementB>()[idx] = (r == c ? 1 : 0);
    }

    for (int idx = 0; idx < problem_size.mn(); ++idx) {
        int r = idx / problem_size.n();
        int c = idx % problem_size.n();
        C.template host_ref<ElementC>()[idx] = ElementC(dist(generator));
//        C.template host_ref<ElementC>()[idx] = 0;
        C_transpose.template host_ref<ElementC>()[c * problem_size.m() + r] = C.template host_ref<ElementC>()[idx];
    }

    A.host_to_device_async(stream);
    B.host_to_device_async(stream);
    C.host_to_device_async(stream);
    C_transpose.host_to_device_async(stream);

    std::cout << "SmemA bytes: " << GemmKernel::smem_A_total_size_in_bytes << ". SmemB bytes: " << GemmKernel::smem_B_total_size_in_bytes << std::endl;
    std::cout << "Total smem bytes: " << GemmKernel::smem_total_size_in_bytes << std::endl;

    if (GemmKernel::smem_total_size_in_bytes >= (48 << 10)) {
        std::cout << "large smem" << std::endl;
        checkCudaErrors(cudaFuncSetAttribute(
                gemm<GemmKernel>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                GemmKernel::smem_total_size_in_bytes));
    }

    int grid_dim = (problem_size.m() / ThreadBlockShape::kM) * (problem_size.n() / ThreadBlockShape::kN);
    int n_iter = 1;
//    // warp up
//    for (int idx = 0; idx < n_iter; ++idx) {
//        gemm<GemmKernel>
//                <<<grid_dim, GemmKernel::thread_per_block, GemmKernel::smem_total_size_in_bytes, stream>>>
//        (problem_size,
//        A.template device_ref<ElementA>(),
//        B.template device_ref<ElementB>(),
//        C.template device_ref<ElementC>(),
//        D.template device_ref<ElementC>());
//    }

    checkCudaErrors(cudaStreamSynchronize(stream));

    cudaEvent_t _start, _stop;
    checkCudaErrors(cudaEventCreate(&_start));
    checkCudaErrors(cudaEventCreate(&_stop));

    float time_in_ms = 0;

    for (int idx = 0; idx < n_iter; ++idx) {
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

    printf("Average time %.2fms\n", time_in_ms / n_iter);

    D.device_to_host_async(stream);
    checkCudaErrors(cudaStreamSynchronize(stream));
    ElementC alpha = 1.0, beta = 1.0;


    checkCudaErrors(cudaStreamSynchronize(stream));

    cublasHandle_t blas_handle;
    cublasCreate(&blas_handle);
    cublasSgemm (blas_handle, CUBLAS_OP_T, CUBLAS_OP_T,
                 problem_size.m(), problem_size.n(), problem_size.k(), &alpha,
                 A.template device_ref<ElementA>(), problem_size.k(),
                 B.template device_ref<ElementB>(), problem_size.n(), &beta,
                 C_transpose.template device_ref<ElementC>(), problem_size.m());

    checkCudaErrors(cudaDeviceSynchronize());
    C.device_to_host_async(stream);
    C_transpose.device_to_host_async(stream);
    checkCudaErrors(cudaStreamSynchronize(stream));

    for (int idx = 0; idx < problem_size.mn(); ++idx) {
        int row = idx / problem_size.n();
        int col = idx % problem_size.n();
        auto cal_value = D.template host_ref<ElementC>()[idx];
        auto ref_value = C_transpose.template host_ref<ElementC>()[col * problem_size.m() + row];
//        ElementC ref_value = 0;
//        for (int k = 0; k < problem_size.k(); ++k) {
//            ref_value += A.template host_ref<ElementA>()[row * problem_size.k() + k] * B.template host_ref<ElementB>()[k * problem_size.n() + col];
//        }
//
//        ref_value += C.template host_ref<ElementC>()[row * problem_size.n() + col];

        double absolute_err = fabs(cal_value - ref_value);
        if (absolute_err > 1e-3) {
            printf("Error at row %d, col %d, reference=%.2f, got %.2f\n", row, col, ref_value, cal_value);
            exit(-1);
        }
    }

    printf("Passed.\n");
}

template<
        typename ElementA,
        typename ElementB,
        typename ElementC,
        typename ThreadBlockShape,
        typename WarpShape,
        typename InstructionShape,
        int AlignmentA,
        int AlignmentB,
        int AlignmentC>
void tensor_core_benchmark(const shape::GemmCoord &problem_size) {
    static_assert(std::is_same<ElementA, half>::value, "");
    static_assert(std::is_same<ElementB, half>::value, "");
    static_assert(std::is_same<ElementC, half>::value, "");

    empty_kernel<<<10, 128>>>();
    using GemmKernel = GemmKernelTensorCore<
            ElementA,
            ElementB,
            ElementC,
            ThreadBlockShape,
            WarpShape,
            InstructionShape,
            AlignmentA,
            AlignmentB,
            AlignmentC>;

    Tensor A(problem_size.mk() * sizeof(ElementA));
    Tensor B(problem_size.kn() * sizeof(ElementB));
    Tensor C(problem_size.mn() * sizeof(ElementC));
    Tensor C_transpose(problem_size.mn() * sizeof(ElementC));
    Tensor D(problem_size.mn() * sizeof(ElementC));

    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::default_random_engine generator;
    std::normal_distribution<double> dist(0, 1);
    for (int idx = 0; idx < problem_size.mk(); ++idx) {
//        A.template host_ref<ElementA>()[idx] = ElementA(dist(generator));
        A.template host_ref<ElementA>()[idx] = ElementA(idx / 1024);
    }

    for (int idx = 0; idx < problem_size.kn(); ++idx) {
//        B.template host_ref<ElementB>()[idx] = ElementB(dist(generator));
        int r = idx / problem_size.n();
        int c = idx % problem_size.n();
        B.template host_ref<ElementB>()[idx] = ElementB(r == c ? 1 : 0);
    }

    for (int idx = 0; idx < problem_size.mn(); ++idx) {
        int r = idx / problem_size.n();
        int c = idx % problem_size.n();
//        C.template host_ref<ElementC>()[idx] = ElementC(dist(generator));
        C.template host_ref<ElementC>()[idx] = ElementC(0);
        C_transpose.template host_ref<ElementC>()[c * problem_size.m() + r] = C.template host_ref<ElementC>()[idx];
    }

    A.host_to_device_async(stream);
    B.host_to_device_async(stream);
    C.host_to_device_async(stream);
    C_transpose.host_to_device_async(stream);

    std::cout << "SmemA bytes: " << GemmKernel::smem_A_total_size_in_bytes << ". SmemB bytes: " << GemmKernel::smem_B_total_size_in_bytes << std::endl;
    std::cout << "Total smem bytes: " << GemmKernel::smem_total_size_in_bytes << std::endl;

    if (GemmKernel::smem_total_size_in_bytes >= (48 << 10)) {
        std::cout << "large smem" << std::endl;
        checkCudaErrors(cudaFuncSetAttribute(
                gemm<GemmKernel>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                GemmKernel::smem_total_size_in_bytes));
    }

    int grid_dim = (problem_size.m() / ThreadBlockShape::kM) * (problem_size.n() / ThreadBlockShape::kN);
    int n_iter = 1;
//    // warp up
//    for (int idx = 0; idx < n_iter; ++idx) {
//        gemm<GemmKernel>
//                <<<grid_dim, GemmKernel::thread_per_block, GemmKernel::smem_total_size_in_bytes, stream>>>
//        (problem_size,
//        A.template device_ref<ElementA>(),
//        B.template device_ref<ElementB>(),
//        C.template device_ref<ElementC>(),
//        D.template device_ref<ElementC>());
//    }

    checkCudaErrors(cudaStreamSynchronize(stream));

    cudaEvent_t _start, _stop;
    checkCudaErrors(cudaEventCreate(&_start));
    checkCudaErrors(cudaEventCreate(&_stop));

    float time_in_ms = 0;

    for (int idx = 0; idx < n_iter; ++idx) {
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

    printf("Average time %.2fms\n", time_in_ms / n_iter);

    D.device_to_host_async(stream);
    checkCudaErrors(cudaStreamSynchronize(stream));
    ElementC alpha = ElementC(1.0), beta = ElementC(1.0);

    checkCudaErrors(cudaStreamSynchronize(stream));

    cublasHandle_t blas_handle;
    cublasCreate(&blas_handle);
    cublasHgemm (blas_handle, CUBLAS_OP_T, CUBLAS_OP_T,
                 problem_size.m(), problem_size.n(), problem_size.k(), &alpha,
                 A.template device_ref<ElementA>(), problem_size.k(),
                 B.template device_ref<ElementB>(), problem_size.n(), &beta,
                 C_transpose.template device_ref<ElementC>(), problem_size.m());

    checkCudaErrors(cudaDeviceSynchronize());
    C.device_to_host_async(stream);
    C_transpose.device_to_host_async(stream);
    checkCudaErrors(cudaStreamSynchronize(stream));

    for (int idx = 0; idx < problem_size.mn(); ++idx) {
        int row = idx / problem_size.n();
        int col = idx % problem_size.n();
        auto cal_value = double(D.template host_ref<ElementC>()[idx]);
        auto ref_value = double(C_transpose.template host_ref<ElementC>()[col * problem_size.m() + row]);

        double absolute_err = fabs(cal_value - ref_value);
        if (absolute_err > 1e-3) {
            printf("Error at row %d, col %d, reference=%.2f, got %.2f\n", row, col, ref_value, cal_value);
            exit(-1);
        }
    }

    printf("Passed.\n");
}

int main () {
    // 60% cublas performance
    benchmark<float, float, float, shape::GemmShape<64, 64, 8>, shape::GemmShape<32, 32, 8>, 4, 4, 4>({1024, 1024, 1024});
//    tensor_core_benchmark<half, half, half, shape::GemmShape<128, 128, 32>, shape::GemmShape<64, 64, 32>, shape::GemmShape<16, 8, 8>, 8, 8, 8>({1024, 1024, 1024});
    return 0;
}

