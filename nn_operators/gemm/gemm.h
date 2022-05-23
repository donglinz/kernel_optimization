//
// Created by donglin on 5/23/22.
//

#ifndef KERNEL_OPTIMIZATION_GEMM_H
#define KERNEL_OPTIMIZATION_GEMM_H

#include "simt_gemm.h"
#include "tensor_core_gemm.h"

template<typename GemmKernel>
__global__
void gemm(
        const shape::GemmCoord problem_size,
        typename GemmKernel::ElementA *ptr_A,
        typename GemmKernel::ElementB *ptr_B,
        typename GemmKernel::ElementC *ptr_C,
        typename GemmKernel::ElementC *ptr_D) {
    GemmKernel gemm_kernel;
    gemm_kernel(problem_size, ptr_A, ptr_B, ptr_C, ptr_D);
}

#endif //KERNEL_OPTIMIZATION_GEMM_H
