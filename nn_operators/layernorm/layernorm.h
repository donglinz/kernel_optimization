//
// Created by dongl on 5/13/2022.
//

#ifndef KERNEL_OPTIMIZATION_LAYERNORM_H
#define KERNEL_OPTIMIZATION_LAYERNORM_H

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include "../../common/common.h"
#include "../../common/functors.h"

template<typename T>
struct WelfordUpdate {
    __device__ __forceinline__
    void operator () (T &count, T &mean, T &m2, const T &new_value) {
        count += 1;
        T delta = new_value - mean;
        mean += delta / count;
        T delta2 = new_value - mean;
        m2 += delta * delta2;
    }
};

template<typename T, int N>
struct WelfordUpdate<AlignedArray<T, N>> {
    __device__ __forceinline__
    void operator () (AlignedArray<T, N> &count, AlignedArray<T, N> &mean, AlignedArray<T, N> &m2, const AlignedArray<T, N> &new_value) {
        #pragma unroll
        for (int idx = 0; idx < N; ++idx) {
            WelfordUpdate<T>()(count[idx], mean[idx], m2[idx], new_value[idx]);
        }
    }
};

template<typename T>
struct WelfordMerge {
    __device__ __forceinline__
    void operator () (T &count_a, T &mean_a, T &m2_a, const T &count_b, const T &mean_b, const T &m2_b) {
        T count_ab = count_a + count_b;
        T delta = mean_b - mean_a;
        T count_b_over_count_ab = count_b / count_ab;
        T m2 = m2_a + m2_b + delta * delta * count_a * count_b_over_count_ab;

        // update merge result
        count_a = count_ab;
        mean_a += delta * count_b_over_count_ab;
        m2_a = m2 / (count_ab - 1);
    }
};

template<typename T>
__device__ __forceinline__
void WelfordWarpReduce(T &result_count, T &result_mean, T &result_m2) {
    #pragma unroll
    for (int offset = 32 / 2; offset > 0; offset >>= 1) {
        WelfordMerge<T>()(result_count, result_mean, result_m2,
                          __shfl_down_sync(0xffffffff, result_count, offset),
                          __shfl_down_sync(0xffffffff, result_mean, offset),
                          __shfl_down_sync(0xffffffff, result_m2, offset));
    }
}

template<typename T, int N>
__device__ __forceinline__
void WelfordWarpReduce(const AlignedArray<T, N> &count, const AlignedArray<T, N> &mean, const AlignedArray<T, N> & m2,
                       T &result_count, T &result_mean, T &result_m2) {
    result_count = count[0];
    result_mean = mean[0];
    result_m2 = m2[0];

    #pragma unroll
    for (int idx = 1; idx < N; ++idx) {
        WelfordMerge<T>()(result_count, result_mean, result_m2, count[idx], mean[idx], m2[idx]);
    }

    WelfordWarpReduce<T>(result_count, result_mean, result_m2);
}

template<typename T, int N>
__device__ __forceinline__
void WelfordWarpAllReduce(const AlignedArray<T, N> &count, const AlignedArray<T, N> &mean, const AlignedArray<T, N> & m2,
                          T &result_count, T &result_mean, T &result_m2) {

    WelfordWarpReduce<T, N>(count, mean, m2, result_count, result_mean, result_m2);

    // broadcast
//    result_count = __shfl_sync(0xffffffff, result_count, 0);
    result_mean = __shfl_sync(0xffffffff, result_mean, 0);
    result_m2 = __shfl_sync(0xffffffff, result_m2, 0);
}

template<typename T, int N>
__device__ __forceinline__
void WelfordBlockReduce(const AlignedArray<T, N> &count, const AlignedArray<T, N> &mean, const AlignedArray<T, N> &m2,
                           T &result_count, T &result_mean, T &result_m2, T *s_reduce_cache) {
    T *result_count_cache = reinterpret_cast<T *>(s_reduce_cache);
    T *result_mean_cache = reinterpret_cast<T *>(s_reduce_cache) + 32;
    T *result_m2_cache = reinterpret_cast<T *>(s_reduce_cache) + 32 * 2;

    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    WelfordWarpReduce<T, N>(count, mean, m2, result_count, result_mean, result_m2);

    if (lane_id == 0) {
        result_count_cache[warp_id] = result_count;
        result_mean_cache[warp_id] = result_mean;
        result_m2_cache[warp_id] = result_m2;
    }

    __syncthreads();

    if (threadIdx.x < blockDim.x / 32) {
        result_count = result_count_cache[lane_id];
        result_mean = result_mean_cache[lane_id];
        result_m2 = result_m2_cache[lane_id];
    }

    if (warp_id == 0) {
        WelfordWarpReduce<T>(result_count, result_mean, result_m2);
    }
}

template<typename T, int N>
__device__ __forceinline__
void WelfordBlockAllReduce(const AlignedArray<T, N> &count, const AlignedArray<T, N> &mean, const AlignedArray<T, N> &m2,
                           T &result_count, T &result_mean, T &result_m2, T *s_reduce_cache) {
    WelfordBlockReduce<T, N>(count, mean, m2, result_count, result_mean, result_m2, s_reduce_cache);

    // broadcast
    if (threadIdx.x == 0) {
        s_reduce_cache[0] = result_count;
        s_reduce_cache[1] = result_mean;
        s_reduce_cache[2] = result_m2;
    }

    __syncthreads();

    result_count = s_reduce_cache[0];
    result_mean = s_reduce_cache[1];
    result_m2 = s_reduce_cache[2];
}

template<typename _Element, int _vec_len, int _num_row, int _num_column, bool _elementwise_affine>
struct LayerNormWarpImpl {
    using Element = _Element;
    static const int vec_len = _vec_len;
    static const int num_row = _num_row;
    static const int num_column = _num_column;
    static const bool elementwise_affine = _elementwise_affine;
    static const int smem_in_bytes = 0;

    __device__ __forceinline__
    static void run(Element *in_data, Element *out_data, Element epsilon, Element *gamma = nullptr, Element *beta = nullptr) {
        static_assert(num_column % vec_len == 0, "Unaligned vectorized access");

        using AccessType = AlignedArray<Element, vec_len>;
        static const int column_per_thread = ((num_column / vec_len) + 32 - 1) / 32;
        AccessType in_data_buffer[column_per_thread];

        const int lane_id = threadIdx.x % 32;
        const int warp_id_block = threadIdx.x / 32;
        const int warp_cnt_block = blockDim.x / 32;
        const int warp_id_global = warp_id_block + blockIdx.x * warp_cnt_block;
        const int warp_cnt_global = warp_cnt_block * gridDim.x;

        for (int row_id = warp_id_global; row_id < num_row; row_id += warp_cnt_global) {
            AccessType thread_count = 0;
            AccessType thread_mean = 0;
            AccessType thread_m2 = 0;

            for (int col_id = lane_id; col_id < num_column / vec_len; col_id += 32) {
                AccessType in_data_element = *reinterpret_cast<AccessType *>(in_data + row_id * num_column + col_id * vec_len);
                in_data_buffer[col_id / 32] = in_data_element;
//                in_data_buffer[0] // AccessType;
//                in_data_buffer[0][0] // float
                WelfordUpdate<AccessType>()(thread_count, thread_mean, thread_m2, in_data_element);
            }

            Element count, mean, m2;
            WelfordWarpAllReduce(thread_count, thread_mean, thread_m2, count, mean, m2);

            Element div = Rsqrt<Element>()(m2 + epsilon);
            for (int col_id = lane_id; col_id < num_column / vec_len; col_id += 32) {
                AccessType &in_data_element = in_data_buffer[col_id / 32];
                AccessType result = Multiply<AccessType>()(Subtract<AccessType>()(in_data_element, mean), div);

                if (elementwise_affine) {
                    AccessType gamma_val = *reinterpret_cast<AccessType *>(gamma + col_id * vec_len);
                    AccessType beta_val = *reinterpret_cast<AccessType *>(beta + col_id * vec_len);

                    result = Multiply<AccessType>()(result, gamma_val);
                    result = Add<AccessType>()(result, beta_val);
                }

                *reinterpret_cast<AccessType *>(out_data + row_id * num_column + col_id * vec_len) = result;
            }
        }
    }
};


template<typename _Element, int _vec_len, int _num_row, int _num_column, bool _elementwise_affine>
struct LayerNormBlockImpl {
    using Element = _Element;
    static const int vec_len = _vec_len;
    static const int num_row = _num_row;
    static const int num_column = _num_column;
    static const bool elementwise_affine = _elementwise_affine;
    static const int smem_in_bytes = num_column * sizeof(Element) + 32 * 3 * sizeof(Element);

    __device__ __forceinline__
    static void run(Element *in_data, Element *out_data, Element epsilon, Element *gamma = nullptr, Element *beta = nullptr) {
        static_assert(num_column % vec_len == 0, "Unaligned vectorized access");
        extern __shared__ int8_t __cache[];
        Element *s_data_cache = reinterpret_cast<Element *>(__cache);
        Element *s_reduce_cache = reinterpret_cast<Element *>(__cache) + num_column;
        using AccessType = AlignedArray<Element, vec_len>;

        for (int row_id = blockIdx.x; row_id < num_row; row_id += gridDim.x) {
            AccessType thread_count = 0;
            AccessType thread_mean = 0;
            AccessType thread_m2 = 0;

            for (int col_id = threadIdx.x; col_id < num_column / vec_len; col_id += blockDim.x) {
                AccessType in_data_element = *reinterpret_cast<AccessType *>(in_data + row_id * num_column + col_id * vec_len);
                reinterpret_cast<AccessType *>(s_data_cache)[col_id] = in_data_element;
                WelfordUpdate<AccessType>()(thread_count, thread_mean, thread_m2, in_data_element);
            }

            Element count, mean, m2;
            WelfordBlockAllReduce(thread_count, thread_mean, thread_m2, count, mean, m2, s_reduce_cache);

            Element div = Rsqrt<Element>()(m2 + epsilon);
            for (int col_id = threadIdx.x; col_id < num_column / vec_len; col_id += blockDim.x) {
                AccessType &in_data_element = reinterpret_cast<AccessType *>(s_data_cache)[col_id];
                AccessType result = Multiply<AccessType>()(Subtract<AccessType>()(in_data_element, mean), div);

                if (elementwise_affine) {
                    AccessType gamma_val = *reinterpret_cast<AccessType *>(gamma + col_id * vec_len);
                    AccessType beta_val = *reinterpret_cast<AccessType *>(beta + col_id * vec_len);

                    result = Multiply<AccessType>()(result, gamma_val);
                    result = Add<AccessType>()(result, beta_val);
                }

                *reinterpret_cast<AccessType *>(out_data + row_id * num_column + col_id * vec_len) = result;
            }
        }
    }
};

template<typename _Element, int _vec_len, int _num_row, int _num_column, bool _elementwise_affine>
struct LayerNormBlockNoCacheImpl {
    using Element = _Element;
    static const int vec_len = _vec_len;
    static const int num_row = _num_row;
    static const int num_column = _num_column;
    static const bool elementwise_affine = _elementwise_affine;
    static const int smem_in_bytes = 32 * 3 * sizeof(Element);

    __device__ __forceinline__
    static void run(Element *in_data, Element *out_data, Element epsilon, Element *gamma = nullptr, Element *beta = nullptr) {
        static_assert(num_column % vec_len == 0, "Unaligned vectorized access");
        extern __shared__ int8_t __cache[];
        Element *s_reduce_cache = reinterpret_cast<Element *>(__cache);
        using AccessType = AlignedArray<Element, vec_len>;

        for (int row_id = blockIdx.x; row_id < num_row; row_id += gridDim.x) {
            AccessType thread_count = 0;
            AccessType thread_mean = 0;
            AccessType thread_m2 = 0;

            for (int col_id = threadIdx.x; col_id < num_column / vec_len; col_id += blockDim.x) {
                AccessType in_data_element = *reinterpret_cast<AccessType *>(in_data + row_id * num_column + col_id * vec_len); // read global
                WelfordUpdate<AccessType>()(thread_count, thread_mean, thread_m2, in_data_element);
            }

            Element count, mean, m2;
            WelfordBlockAllReduce(thread_count, thread_mean, thread_m2, count, mean, m2, s_reduce_cache);

            Element div = Rsqrt<Element>()(m2 + epsilon);
            for (int col_id = threadIdx.x; col_id < num_column / vec_len; col_id += blockDim.x) {
                AccessType in_data_element = *reinterpret_cast<AccessType *>(in_data + row_id * num_column + col_id * vec_len); // read global
                AccessType result = Multiply<AccessType>()(Subtract<AccessType>()(in_data_element, mean), div);

                if (elementwise_affine) {
                    AccessType gamma_val = *reinterpret_cast<AccessType *>(gamma + col_id * vec_len);
                    AccessType beta_val = *reinterpret_cast<AccessType *>(beta + col_id * vec_len);

                    result = Multiply<AccessType>()(result, gamma_val);
                    result = Add<AccessType>()(result, beta_val);
                }

                *reinterpret_cast<AccessType *>(out_data + row_id * num_column + col_id * vec_len) = result;
            }
        }
    }
};

template<typename LayerNormKernel>
__global__
void layer_normalization(
    typename LayerNormKernel::Element *in_data,
    typename LayerNormKernel::Element *out_data,
    typename LayerNormKernel::Element epsilon,
    typename LayerNormKernel::Element *gamma = nullptr,
    typename LayerNormKernel::Element *beta = nullptr) {

    LayerNormKernel::run(in_data, out_data, epsilon, gamma, beta);
}

#endif //KERNEL_OPTIMIZATION_LAYERNORM_H
