#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <iostream>
#include <memory>
#include <type_traits>

// 一些CUDA辅助函数和宏
namespace cuda {

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) ::cuda::check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line)
{
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":"
                  << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

template <class T>
T *cuMallocManaged()
{
    T *ptr;
    checkCudaErrors(cudaMallocManaged(&ptr, sizeof(T)));
    return ptr;
}

template <typename T>
class shared_ptr
{
public:
    __host__ shared_ptr(T *ptr) : p(ptr), ref_count(cuMallocManaged<size_t>()) { *ref_count = 1; }
    template <typename U,
              typename = typename std::enable_if_t<std::is_convertible<U *, T *>::value>>
    __host__ shared_ptr(const shared_ptr<U> &sp) : p(sp.p)
                                                 , ref_count(sp.ref_count)
    {
        *ref_count++;
    }
    __host__ ~shared_ptr()
    {
        if (--ref_count == 0) {
            p->~T();
            checkCudaErrors(cudaFree(p));
            checkCudaErrors(cudaFree(ref_count));
        }
    }

    __host__ __device__ T *operator->() const noexcept { return p; }
    __host__ __device__ T &operator*() const noexcept { return *p; }
    __host__ __device__ T *get() const noexcept { return p; }

private:
    template <typename U>
    friend class shared_ptr;

    T      *p;
    size_t *ref_count;
};

template <class T, typename... Args>
shared_ptr<T> make_shared(Args &&...args)
{
    T *ptr = cuMallocManaged<T>();
    new (ptr) T(::std::forward<Args>(args)...);
    return shared_ptr<T>(ptr);
}

}  // namespace cuda

#endif
