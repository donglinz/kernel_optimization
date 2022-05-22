//
// Created by dongl on 5/13/2022.
//

#include "tensor.h"
#include "common.h"

Tensor::Tensor(size_t _size_in_bytes) : size_in_bytes(_size_in_bytes) {
    this->host_ptr = new uint8_t[this->size_in_bytes];
    checkCudaErrors(cudaMalloc(&this->device_ptr, this->size_in_bytes));
}

Tensor &Tensor::operator= (Tensor &&rhs) {
    this->host_ptr = rhs.host_ptr;
    this->device_ptr = rhs.device_ptr;

    rhs.host_ptr = nullptr;
    rhs.device_ptr = nullptr;

    return *this;
}

Tensor::~Tensor() {
    if (this->host_ptr) delete reinterpret_cast<uint8_t *>(this->host_ptr);
    if (this->device_ptr) checkCudaErrors(cudaFree(this->device_ptr));
}

void Tensor::host_to_device() {
    checkCudaErrors(cudaMemcpy(this->device_ptr, this->host_ptr, this->size_in_bytes, cudaMemcpyHostToDevice));
}

void Tensor::host_to_device_async(cudaStream_t &stream) {
    checkCudaErrors(cudaMemcpyAsync(this->device_ptr, this->host_ptr, this->size_in_bytes, cudaMemcpyHostToDevice, stream));
}

void Tensor::device_to_host() {
    checkCudaErrors(cudaMemcpy(this->host_ptr, this->device_ptr, this->size_in_bytes, cudaMemcpyDeviceToHost));
}

void Tensor::device_to_host_async(cudaStream_t &stream) {
    checkCudaErrors(cudaMemcpyAsync(this->host_ptr, this->device_ptr, this->size_in_bytes, cudaMemcpyDeviceToHost, stream));
}