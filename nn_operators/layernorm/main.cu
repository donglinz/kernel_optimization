//
// Created by dongl on 5/13/2022.
//

#include <cuda.h>
#include <vector>
#include <cuda_runtime.h>
#include "layernorm.h"
#include "../../common/tensor.h"
#include "../../common/common.h"
#include <random>

template<
        int grid_dim,
        int block_dim,
        typename T,
        template<typename, int, int, int, bool> typename LayerNorm,
        int vec_len,
        int num_row,
        int num_column>
void run_benchmark() {
    using LayerNormKernel = LayerNorm<T, vec_len, num_row, num_column, false>;

    Tensor in_data(num_row * num_column * sizeof(T));
    Tensor out_data(num_row * num_column * sizeof(T));

    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::default_random_engine generator;
    std::normal_distribution<double> dist(0, 1);

    for (int idx = 0; idx < num_row * num_column; ++idx) {
        in_data.template host_ref<T>()[idx] = T(dist(generator));
    }

    in_data.host_to_device_async(stream);

    if (LayerNormKernel::smem_in_bytes >= (48 << 10)) {
        checkCudaErrors(cudaFuncSetAttribute(layer_normalization<LayerNormKernel>, cudaFuncAttributeMaxDynamicSharedMemorySize, LayerNormKernel::smem_in_bytes));
    }

    int n_iter = 1;

    checkCudaErrors(cudaStreamSynchronize(stream));

    cudaEvent_t _start, _stop;
    checkCudaErrors(cudaEventCreate(&_start));
    checkCudaErrors(cudaEventCreate(&_stop));

    float time_in_ms = 0;

    for (int idx = 0; idx < n_iter; ++idx) {
        in_data.host_to_device_async(stream);

        checkCudaErrors(cudaEventRecord(_start, stream));
        layer_normalization<LayerNormKernel>
        <<<grid_dim, block_dim, LayerNormKernel::smem_in_bytes, stream>>>
        (in_data.template device_ref<T>(), out_data.template device_ref<T>(), 0.0001);
        checkCudaErrors(cudaEventRecord(_stop, stream));

        checkCudaErrors(cudaEventSynchronize(_stop));

        float ms;
        checkCudaErrors(cudaEventElapsedTime(&ms, _start, _stop));
        time_in_ms += ms;
    }

    checkCudaErrors(cudaStreamSynchronize(stream));

    printf("======Num row %d, num column %d======\n", num_row, num_column);
    printf("Average time %.2fms\n", time_in_ms / n_iter);
}
int main() {
    run_benchmark<40*8, 128, float, LayerNormWarpImpl, 4, 128 * 128, 768>();
    run_benchmark<40*16, 64, float, LayerNormBlockImpl, 4, 128 * 128, 768>();
//    run_benchmark<float, LayerNormBlockNoCacheImpl, 4, 49152, 1024>();
    return 0;
}