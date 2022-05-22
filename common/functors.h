//
// Created by dongl on 5/13/2022.
//

#ifndef KERNEL_OPTIMIZATION_FUNCTORS_H
#define KERNEL_OPTIMIZATION_FUNCTORS_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "common.h"

template<typename T>
struct ReduceMax {
    __device__ __forceinline__
    T operator () (const T &a, const T& b) const {
        return a > b ? a : b;
    }
};

template<typename T, int N>
struct ReduceMax<AlignedArray<T, N>> {
    __device__ __forceinline__
    AlignedArray<T, N> operator () (const AlignedArray<T, N> &a, const AlignedArray<T, N> &b) const {
        AlignedArray<T, N> ret;

        #pragma unroll
        for (int idx = 0; idx < N; ++idx) {
            ret[idx] = ReduceMax<T>()(a[idx], b[idx]);
        }

        return ret;
    }
};


template<typename T>
struct ReduceSum {
    __device__ __forceinline__
    T operator () (const T &a, const T& b) const {
        return a + b;
    }
};

template<typename T, int N>
struct ReduceSum<AlignedArray<T, N>> {
    __device__ __forceinline__
    AlignedArray<T, N> operator () (const AlignedArray<T, N> &a, const AlignedArray<T, N> &b) const {
        AlignedArray<T, N> ret;

        #pragma unroll
        for (int idx = 0; idx < N; ++idx) {
            ret[idx] = ReduceSum<T>()(a[idx], b[idx]);
        }

        return ret;
    }
};

template<typename T>
struct Exp;

template<>
struct Exp<float> {
    __device__ __forceinline__
    float operator () (const float &a) const {
        return expf(a);
    }
};

template<>
struct Exp<half> {
    __device__ __forceinline__
    half operator () (const half &a) const {
        return hexp(a);
    }
};

template<typename T, int N>
struct Exp<AlignedArray<T, N>> {
    __device__ __forceinline__
    AlignedArray<T, N> operator () (const AlignedArray<T, N> &a) const {
        AlignedArray<T, N> ret;

        #pragma unroll
        for (int idx = 0; idx < N; ++idx) {
            ret[idx] = Exp<T>()(a[idx]);
        }

        return ret;
    }
};

template<typename T>
struct Rsqrt;

template<>
struct Rsqrt<float> {
    __device__ __forceinline__
    float operator () (const float &a) const {
        return rsqrtf(a);
    }
};

template<>
struct Rsqrt<half> {
    __device__ __forceinline__
    half operator () (const half &a) const {
        return hrsqrt(a);
    }
};

template<typename T, int N>
struct Rsqrt<AlignedArray<T, N>> {
    __device__ __forceinline__
    AlignedArray<T, N> operator () (const AlignedArray<T, N> &a) const {
        AlignedArray<T, N> ret;

        #pragma unroll
        for (int idx = 0; idx < N; ++idx) {
            ret[idx] = Rsqrt<T>()(a[idx]);
        }

        return ret;
    }
};


template<typename T>
struct Add {
    __device__ __forceinline__
    T operator () (const T &a, const T &b) const {
        return a + b;
    }
};

template<typename T, int N>
struct Add<AlignedArray<T, N>> {
    __device__ __forceinline__
    AlignedArray<T, N> operator () (const AlignedArray<T, N> &a, const AlignedArray<T, N> &b) const {
        AlignedArray<T, N> ret;

        #pragma unroll
        for (int idx = 0; idx < N; ++idx) {
            ret[idx] = Add<T>()(a[idx], b[idx]);
        }

        return ret;
    }

    __device__ __forceinline__
    AlignedArray<T, N> operator () (const AlignedArray<T, N> &a, const T &b) const {
        AlignedArray<T, N> ret;

        #pragma unroll
        for (int idx = 0; idx < N; ++idx) {
            ret[idx] = Add<T>()(a[idx], b);
        }

        return ret;
    }
};

template<typename T>
struct Subtract {
    __device__ __forceinline__
    T operator () (const T &a, const T &b) const {
        return a + b;
    }
};

template<typename T, int N>
struct Subtract<AlignedArray<T, N>> {
    __device__ __forceinline__
    AlignedArray<T, N> operator () (const AlignedArray<T, N> &a, const AlignedArray<T, N> &b) const {
        AlignedArray<T, N> ret;

        #pragma unroll
        for (int idx = 0; idx < N; ++idx) {
            ret[idx] = Subtract<T>()(a[idx], b[idx]);
        }

        return ret;
    }

    __device__ __forceinline__
    AlignedArray<T, N> operator () (const AlignedArray<T, N> &a, const T &b) const {
        AlignedArray<T, N> ret;

    #pragma unroll
        for (int idx = 0; idx < N; ++idx) {
            ret[idx] = Subtract<T>()(a[idx], b);
        }

        return ret;
    }
};

template<typename T>
struct Multiply {
    __device__ __forceinline__
    T operator () (const T &a, const T &b) const {
        return a * b;
    }
};

template<typename T, int N>
struct Multiply<AlignedArray<T, N>> {
    __device__ __forceinline__
    AlignedArray<T, N> operator () (const AlignedArray<T, N> &a, const AlignedArray<T, N> &b) const {
        AlignedArray<T, N> ret;

        #pragma unroll
        for (int idx = 0; idx < N; ++idx) {
            ret[idx] = Multiply<T>()(a[idx], b[idx]);
        }

        return ret;
    }

    __device__ __forceinline__
    AlignedArray<T, N> operator () (const AlignedArray<T, N> &a, const T &b) const {
        AlignedArray<T, N> ret;

        #pragma unroll
        for (int idx = 0; idx < N; ++idx) {
            ret[idx] = Multiply<T>()(a[idx], b);
        }

        return ret;
    }
};

template<typename T>
struct Divide {
    __device__ __forceinline__
    T operator () (const T &a, const T &b) const {
        return a / b;
    }
};

template<typename T, int N>
struct Divide<AlignedArray<T, N>> {
    __device__ __forceinline__
    AlignedArray<T, N> operator () (const AlignedArray<T, N> &a, const AlignedArray<T, N> &b) const {
        AlignedArray<T, N> ret;

        #pragma unroll
        for (int idx = 0; idx < N; ++idx) {
            ret[idx] = Divide<T>()(a[idx], b[idx]);
        }

        return ret;
    }

    __device__ __forceinline__
    AlignedArray<T, N> operator () (const AlignedArray<T, N> &a, const T &b) const {
        AlignedArray<T, N> ret;

        #pragma unroll
        for (int idx = 0; idx < N; ++idx) {
            ret[idx] = Divide<T>()(a[idx], b);
        }

        return ret;
    }
};

template<typename T>
struct MultiplyAdd;

template<>
struct MultiplyAdd<float> {
    __device__ __forceinline__
    float operator () (const float &a, const float &b, const float &c) const {
        return __fmaf_rd(a, b, c);
    }
};

template<>
struct MultiplyAdd<half> {
    __device__ __forceinline__
    half operator () (const half &a, const half &b, const half &c) const {
        return __hfma(a, b, c);
    }
};

template<typename T, int N>
struct MultiplyAdd<AlignedArray<T, N>> {
    __device__ __forceinline__
    AlignedArray<T, N> operator () (const AlignedArray<T, N> &a, const AlignedArray<T, N> &b, const AlignedArray<T, N> &c) const {
        AlignedArray<T, N> ret;

        #pragma unroll
        for (int idx = 0; idx < N; ++idx) {
            ret[idx] = MultiplyAdd<T>()(a[idx], b[idx], c[idx]);
        }

        return ret;
    }
};

#endif //KERNEL_OPTIMIZATION_FUNCTORS_H
