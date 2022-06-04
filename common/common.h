//
// Created by dongl on 5/13/2022.
//

#ifndef KERNEL_OPTIMIZATION_COMMON_H
#define KERNEL_OPTIMIZATION_COMMON_H

#include <iostream>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

static const char *_cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorName(error);
}

template<typename T>
void check(T result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

__device__ __forceinline__
int divide_up(const int &a, const int &b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__
constexpr int const_max(const int &a, const int &b) {
    return a > b ? a : b;
}

template<template<typename> typename ReductionOp, typename T>
__device__ __forceinline__
T WarpReduce(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = ReductionOp<T>()(val, __shfl_down_sync(0xffffffff, val, offset));
    }

    return val;
}

template<template<typename> typename ReductionOp, typename T>
__device__ __forceinline__
T BlockReduce(T val, T *s_reduce_cache) {
    val = WarpReduce<ReductionOp, T>(val);

    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;

    if (lane_id == 0) s_reduce_cache[warp_id] = val;
    __syncthreads();

    if (threadIdx.x < blockDim.x / 32) {
        val = s_reduce_cache[threadIdx.x];
    }

    if (warp_id == 0) val = WarpReduce<ReductionOp, T>(val);
    return val;
}

template<typename T, int N>
struct alignas(sizeof(T) * N) AlignedArray {
    using Element = T;
    static const int kElements = N;

    __device__ __host__
    AlignedArray() {}

    __device__ __host__
    AlignedArray(const T &rhs) {
        #pragma unroll
        for (int idx = 0; idx < kElements; ++idx) {
            this->at(idx) = rhs;
        }
    }
//
//    __device__ __host__
//    AlignedArray(const AlignedArray<T, N> &rhs) {
//        #pragma unroll
//        for (int idx = 0; idx < kElements; ++idx) {
//            this->at(idx) = rhs[idx];
//        }
//    }

//    __device__ __host__
//    AlignedArray<T, N> &operator= (const AlignedArray<T, N> &rhs) {
//        #pragma unroll
//        for (int idx = 0; idx < kElements; ++idx) {
//            this->at(idx) = rhs[idx];
//        }
//
//        return *this;
//    }

    __device__ __host__
    T &operator [] (int offset) {
        return reinterpret_cast<T &>(this->buffer[offset]);
    }

    __device__ __host__
    const T &operator [] (int offset) const {
        return reinterpret_cast<const T &>(this->buffer[offset]);
    }

    __device__ __host__
    T &at(int offset) {
        return reinterpret_cast<T &>(this->buffer[offset]);
    }

    __device__ __host__
    const T &at(int offset) const {
        return reinterpret_cast<const T &>(this->buffer[offset]);
    }

    __device__ __forceinline__
    void clear() {
        #pragma unroll
        for (int idx = 0; idx < kElements; ++idx) {
            this->at(idx) = Element(0);
        }
    }

    Element buffer[N];
};

//#include <cutlass/array.h>
//
//template<typename T, int N>
//using AlignedArray = cutlass::AlignedArray<T, N>;


template<typename ReductionOp, typename T, int N>
__device__ __forceinline__
T to_scalar(const AlignedArray<T, N> &data) {
    T ret = data[0];

    #pragma unroll
    for (int idx = 1; idx < N; ++idx) {
        ret = ReductionOp()(ret, data[idx]);
    }

    return ret;
}

namespace limits {
    template<typename T>
    struct numeric_limits;

    template<>
    struct numeric_limits<float> {
        __device__ __forceinline__
        static float max() {
            return __FLT_MAX__;
        }
    };

    template<>
    struct numeric_limits<int32_t> {
        __device__ __forceinline__
        static int32_t max() {
            return __INT32_MAX__;
        }
    };

    template<>
    struct numeric_limits<half> {
        __device__ __forceinline__
        static half max() {
            return half(65504.0);
        }
    };
}

namespace shape {
    class GemmCoord {
    public:
        __device__ __host__
        GemmCoord(size_t _m, size_t _n, size_t _k) : _m(_m), _n(_n), _k(_k) {}

        __device__ __host__
        size_t m() const { return this->_m; }
        __device__ __host__
        size_t n() const { return this->_n; }
        __device__ __host__
        size_t k() const { return this->_k; }
        __device__ __host__
        size_t mk() const { return this->_m * this->_k; }
        __device__ __host__
        size_t kn() const { return this->_k * this->_n; }
        __device__ __host__
        size_t mn() const { return this->_m * this->_n; }
        __device__ __host__
        size_t mnk() const { return this->_m * this->_n * this->_k; }

        std::string to_string() const {
            return std::string("{") + std::to_string(this->m()) + "," + std::to_string(this->n()) + "," + std::to_string(this->k()) + "}";
        }

    private:
        size_t _m, _n, _k;
    };

    template<int _M, int _N>
    struct MatrixShape {
        static const int kM = _M;
        static const int kN = _N;
    };

    template<int _M, int _N, int _K>
    struct GemmShape {
        static const int kM = _M;
        static const int kN = _N;
        static const int kK = _K;

        static const int kMN = kM * kN;
        static const int kMK = kM * kK;
        static const int kKN = kK * kN;

        static const int kMNK = kM * kN * kK;

        static std::string to_string() {
            return std::string("{") + std::to_string(kM) + "," + std::to_string(kN) + "," + std::to_string(kK) + "}";
        }
    };
}

#endif //KERNEL_OPTIMIZATION_COMMON_H
