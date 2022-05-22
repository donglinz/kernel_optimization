//
// Created by dongl on 5/13/2022.
//

#ifndef KERNEL_OPTIMIZATION_SOFTMAX_H
#define KERNEL_OPTIMIZATION_SOFTMAX_H

#include "../../common/common.h"
#include "../../common/functors.h"

template<typename _T, int _vec_len, int _num_row, int _num_column>
struct SoftMaxWarpImpl {
    static const int smem_in_bytes = 0;
    using T = _T;
    static const int vec_len = _vec_len;
    static const int num_row = _num_row;
    static const int num_column = _num_column;

    __device__ __forceinline__
    static void run(T *in_data, T *out_data) {
        static_assert(num_column % vec_len == 0, "Unaligned vectorized access");
        using AccessType = AlignedArray<T, vec_len>;
        static const int column_per_thread = ((num_column / vec_len) + 32 - 1) / 32;
        AccessType  in_data_buffer[column_per_thread];

        const int lane_id = threadIdx.x % 32;
        const int warp_id_block = threadIdx.x / 32;
        const int warp_cnt_block = blockDim.x / 32;
        const int warp_id_global = warp_id_block + blockIdx.x * warp_cnt_block;
        const int warp_cnt_global = warp_cnt_block * gridDim.x;

        for (int row_id = warp_id_global; row_id < num_row; row_id += warp_cnt_global) {
            AccessType tmp = -limits::numeric_limits<T>::max();

            for (int col_id = lane_id; col_id < num_column / vec_len; col_id += 32) {
                AccessType in_data_element = *reinterpret_cast<AccessType *>(in_data + row_id * num_column + col_id * vec_len);
                in_data_buffer[col_id / 32] = in_data_element;
                tmp = ReduceMax<AccessType>()(tmp, in_data_element);
            }

            T reduce_max = WarpReduce<ReduceMax, T>(to_scalar<ReduceMax<T>, T, vec_len>(tmp));

            reduce_max = __shfl_sync(0xffffffff, reduce_max, 0);

            tmp = T(0);
            for (int col_id = lane_id; col_id < num_column / vec_len; col_id += 32) {
                AccessType &in_data_element = in_data_buffer[col_id / 32];
                in_data_element = Exp<AccessType>()(Subtract<AccessType>()(in_data_element, reduce_max));

                tmp = ReduceSum<AccessType>()(tmp, in_data_element);
            }

            T reduce_sum = WarpReduce<ReduceSum, T>(to_scalar<ReduceSum<T>, T, vec_len>(tmp));
            reduce_sum = __shfl_sync(0xffffffff, reduce_sum, 0);

            for (int col_id = lane_id; col_id < num_column / vec_len; col_id += 32) {
                *reinterpret_cast<AccessType * >(out_data + row_id * num_column + col_id * vec_len) =
                        Divide<AccessType>()(in_data_buffer[col_id / 32], reduce_sum);
            }
        }
    }
};

template<typename _T, int _vec_len, int _num_row, int _num_column>
struct SoftMaxBlockImpl {
    using T = _T;
    static const int vec_len = _vec_len;
    static const int num_row = _num_row;
    static const int num_column = _num_column;
    static const int smem_in_bytes = num_column * sizeof(T) + 32 * sizeof(T);

    __device__ __forceinline__
    static void run(T *in_data, T *out_data) {
        static_assert(num_column % vec_len == 0, "Unaligned vectorized access");
        extern __shared__ int8_t __cache[];
        T *s_data_cache = reinterpret_cast<T *>(__cache);
        T *s_reduce_cache = reinterpret_cast<T *>(__cache) + num_column;
        using AccessType = AlignedArray<T, vec_len>;

        for (int row_id = blockIdx.x; row_id < num_row; row_id += gridDim.x) {
            AccessType tmp = -limits::numeric_limits<T>::max();

            for (int col_id = threadIdx.x; col_id < num_column / vec_len; col_id += blockDim.x) {
                AccessType in_data_element = *reinterpret_cast<AccessType *>(in_data + row_id * num_column + col_id * vec_len);
                reinterpret_cast<AccessType *>(s_data_cache)[col_id] = in_data_element;
                tmp = ReduceMax<AccessType>()(tmp, in_data_element);
            }

            T reduce_max = BlockReduce<ReduceMax, T>(to_scalar<ReduceMax<T>, T, vec_len>(tmp), s_reduce_cache);

            if (threadIdx.x == 0) s_reduce_cache[0] = reduce_max;
            __syncthreads();
            reduce_max = s_reduce_cache[0];

            tmp = T(0);
            for (int col_id = threadIdx.x; col_id < num_column / vec_len; col_id += blockDim.x) {
                AccessType &in_data_element = reinterpret_cast<AccessType *>(s_data_cache)[col_id];
                in_data_element = Exp<AccessType>()(Subtract<AccessType>()(in_data_element, reduce_max));

                tmp = ReduceSum<AccessType>()(tmp, in_data_element);
            }

            T reduce_sum = BlockReduce<ReduceSum, T>(to_scalar<ReduceSum<T>, T, vec_len>(tmp), s_reduce_cache);

            if (threadIdx.x == 0) s_reduce_cache[0] = reduce_sum;
            __syncthreads();
            reduce_sum = s_reduce_cache[0];

            for (int col_id = threadIdx.x; col_id < num_column / vec_len; col_id += blockDim.x) {
                *reinterpret_cast<AccessType * >(out_data + row_id * num_column + col_id * vec_len) =
                        Divide<AccessType>()(reinterpret_cast<AccessType *>(s_data_cache)[col_id], reduce_sum);
            }
        }
    }
};


template<typename _T, int _vec_len, int _num_row, int _num_column>
struct SoftMaxBlockNoCacheImpl {
    using T = _T;
    static const int vec_len = _vec_len;
    static const int num_row = _num_row;
    static const int num_column = _num_column;
    static const int smem_in_bytes = 32 * sizeof(T);

    __device__ __forceinline__
    static void run(T *in_data, T *out_data) {
        static_assert(num_column % vec_len == 0, "Unaligned vectorized access");
        extern __shared__ int8_t __cache[];
        T *s_reduce_cache = reinterpret_cast<T *>(__cache);
        using AccessType = AlignedArray<T, vec_len>;

        for (int row_id = blockIdx.x; row_id < num_row; row_id += gridDim.x) {
            AccessType tmp = -limits::numeric_limits<T>::max();

            for (int col_id = threadIdx.x; col_id < num_column / vec_len; col_id += blockDim.x) {
                AccessType in_data_element = *reinterpret_cast<AccessType *>(in_data + row_id * num_column + col_id * vec_len);
                tmp = ReduceMax<AccessType>()(tmp, in_data_element);
            }

            T reduce_max = BlockReduce<ReduceMax, T>(to_scalar<ReduceMax<T>, T, vec_len>(tmp), s_reduce_cache);

            if (threadIdx.x == 0) s_reduce_cache[0] = reduce_max;
            __syncthreads();
            reduce_max = s_reduce_cache[0];

            tmp = T(0);
            for (int col_id = threadIdx.x; col_id < num_column / vec_len; col_id += blockDim.x) {
                AccessType in_data_element = *reinterpret_cast<AccessType *>(in_data + row_id * num_column + col_id * vec_len);
                in_data_element = Exp<AccessType>()(Subtract<AccessType>()(in_data_element, reduce_max));

                tmp = ReduceSum<AccessType>()(tmp, in_data_element);
            }

            T reduce_sum = BlockReduce<ReduceSum, T>(to_scalar<ReduceSum<T>, T, vec_len>(tmp), s_reduce_cache);

            if (threadIdx.x == 0) s_reduce_cache[0] = reduce_sum;
            __syncthreads();
            reduce_sum = s_reduce_cache[0];

            for (int col_id = threadIdx.x; col_id < num_column / vec_len; col_id += blockDim.x) {
                AccessType in_data_element = *reinterpret_cast<AccessType *>(in_data + row_id * num_column + col_id * vec_len);
                *reinterpret_cast<AccessType * >(out_data + row_id * num_column + col_id * vec_len) =
                        Divide<AccessType>()(in_data_element, reduce_sum);
            }
        }
    }
};

template<typename SoftMaxKernel>
__global__
void softmax(
        typename SoftMaxKernel::T *in_data,
        typename SoftMaxKernel::T *out_data) {
    SoftMaxKernel::run(in_data, out_data);
}

#endif //KERNEL_OPTIMIZATION_SOFTMAX_H
