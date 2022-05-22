//
// Created by dongl on 5/13/2022.
//

#ifndef KERNEL_OPTIMIZATION_TENSOR_H
#define KERNEL_OPTIMIZATION_TENSOR_H

#include <vector>
#include <numeric>

class Tensor {
public:
    Tensor(size_t _size_in_bytes);
    Tensor(std::vector<size_t> _shape, size_t byte_per_element)
        : Tensor((size_t)byte_per_element * std::accumulate(_shape.begin(), _shape.end(), 1, std::multiplies<size_t>())) {}
    Tensor &operator= (Tensor &&rhs);
    Tensor(const Tensor &) = delete;
    Tensor &operator =(const Tensor &) = delete;

    template<typename T>
    T *host_ref() {
        return reinterpret_cast<T *>(this->host_ptr);
    }

    template<typename T>
    T *device_ref() {
        return reinterpret_cast<T *>(this->device_ptr);
    }

    void host_to_device();
    void host_to_device_async(cudaStream_t &stream);
    void device_to_host();
    void device_to_host_async(cudaStream_t &stream);

    ~Tensor();
private:
    void *host_ptr;
    void *device_ptr;
    size_t size_in_bytes;
};


#endif //KERNEL_OPTIMIZATION_TENSOR_H
