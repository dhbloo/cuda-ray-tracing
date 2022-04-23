//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

//#include "aarect.h"
//#include "box.h"
#include "camera.h"
#include "color.h"
#include "external/window.h"
//#include "hittable_list.h"
//#include "material.h"
#include "rtweekend.h"
//#include "sphere.h"

#include <atomic>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <iomanip>
#include <iostream>
#include <thread>

// ======================================================

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

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

// ======================================================
// 这里使用一个“全局”函数，来获取每个线程的随机数

// 表示随机数状态的全局数组
__device__ curandState *dev_rand_state;
__device__ int          rand_width, rand_height;
__device__ inline float random_float()
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= rand_width || j >= rand_height)
        return 0.0f;

    int   id = j * rand_width + i;
    float x  = curand_uniform(&dev_rand_state[id]);

    return x;
}

// ======================================================

__global__ void render_init(int width, int height, curandState *rand_state)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= width || j >= height)
        return;

    int id = j * width + i;
    if (id == 0) {
        dev_rand_state = rand_state;
        rand_width     = width;
        rand_height    = height;
    }
    curand_init(42, id, 0, &rand_state[id]);
}

__global__ void render(color *fb, int width, int height)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= width || j >= height)
        return;

    // fb[j * width + i] = color(float(i) / width, float(j) / height, 0.2f);
    fb[j * width + i] = color::random();
}

int main()
{
    // Image

    const int  image_width       = 600;
    const int  image_height      = 600;
    const int  thread_width      = 8;
    const int  thread_height     = 8;
    const auto aspect_ratio      = static_cast<float>(image_width) / image_height;
    const int  samples_per_pixel = 10;
    const int  max_depth         = 50;

    // Render Init

    Window window(image_width, image_height, "Ray tracing (GPU)");
    size_t num_pixels = image_width * image_height;
    color *frame_buffer;
    checkCudaErrors(cudaMallocManaged((void **)&frame_buffer, num_pixels * sizeof(color)));
    curandState *rand_state;
    checkCudaErrors(cudaMalloc((void **)&rand_state, num_pixels * sizeof(curandState)));

    dim3 blocks((image_width + thread_width - 1) / thread_width,
                (image_height + thread_height - 1) / thread_height);
    dim3 threads(thread_width, thread_height);

    render_init<<<blocks, threads>>>(image_width, image_height, rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Render

    auto start_time = std::chrono::high_resolution_clock::now();

    render<<<blocks, threads>>>(frame_buffer, image_width, image_height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    auto  end_time   = std::chrono::high_resolution_clock::now();
    auto  elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    float elapsed_seconds = elapsed_ms.count() * 0.001f;
    int64_t total_rays    = (int64_t)samples_per_pixel * image_height * image_width;
    std::cout << std::fixed << std::setprecision(3) << "\nDone after " << elapsed_seconds
              << " seconds, " << total_rays / elapsed_seconds / 1000000.f << " Mrays per second.\n";

    // Copy frame buffer to window
    for (int j = image_height - 1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            size_t pixel_index               = j * image_width + i;
            *window(i, image_height - 1 - j) = color_to_rgb_integer(frame_buffer[pixel_index]);
        }
    }

    window.update();
    while (window.is_run()) {
        window.dispatch();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Cleanup

    checkCudaErrors(cudaFree(frame_buffer));
    checkCudaErrors(cudaFree(rand_state));
}
