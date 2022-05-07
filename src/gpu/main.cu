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

#define rstate_t curandState *
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

// common headers

#include "camera.h"
#include "color.h"
#include "hittable_list.h"
#include "rtweekend.h"
#include "scene.h"

// window
#include "external/window.h"

#include <atomic>
#include <iomanip>
#include <iostream>
#include <thread>

// 一些CUDA辅助函数和宏

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

__device__ float random_float(rstate_t state)
{
    return curand_uniform((curandState *)state);
}

// ======================================================

// 构造场景

__global__ void setup_secne(hittable_list **scene_ptr)
{
    scene_ptr[0] = new hittable_list;
    scene_ptr[1] = new hittable_list;
    cornell_box(*scene_ptr[0], *scene_ptr[1]);
}

// 构造场景

__global__ void cleanup_secne(hittable_list **scene_ptr)
{
    delete scene_ptr[0];
    delete scene_ptr[1];
}

// 渲染

__global__ void ray_radiance(color          *fb,
                             int             width,
                             int             height,
                             int             samples_per_pixel,
                             int             max_depth,
                             camera         *cam,
                             hittable_list **scene_ptr)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= width || j >= height)
        return;

    curandState rand_state;
    curand_init(42 + j * width + i, 0, 0, &rand_state);

    const hittable &world  = *scene_ptr[0];
    const hittable &lights = *scene_ptr[1];

    color          radiance(0, 0, 0);
    hit_record     rec;
    scatter_record srec;
    color          accumL, accumR;
    ray            r;

    for (int s = 0; s < samples_per_pixel; ++s) {
        r      = cam->get_ray(&rand_state, i, j, width, height);
        accumL = color();
        accumR = color(1.0f, 1.0f, 1.0f);

        for (int depth = 0; depth < max_depth; depth++) {
            // If the ray hits nothing, return the background color.
            if (!world.hit(r, 0.001f, infinity, rec)) {
                accumL += accumR * background_radiance(r);
                break;
            }

            accumL += accumR * rec.mat_ptr->emitted(r, rec, rec.u, rec.v, rec.p);

            if (!rec.mat_ptr->scatter(r, rec, srec, &rand_state))
                break;

            accumR = accumR * srec.attenuation;

            if (srec.is_specular) {
                r = srec.specular_ray;
            }
            else {
                hittable_pdf light(lights, rec.p);
                mixture_pdf  p(light, *srec.pdf_ptr);
                ray          scattered = ray(rec.p, p.generate(&rand_state));
                auto         pdf_val   = p.value(scattered.direction());

                accumR *= rec.mat_ptr->scattering_pdf(r, rec, scattered, &rand_state) / pdf_val;
                r = scattered;
            }
        }

        radiance += accumL;
    }

    fb[j * width + i] = radiance;
}

int main()
{
    // Image

    const int  image_width       = 800;
    const int  image_height      = 800;
    const int  thread_width      = 32;
    const int  thread_height     = 16;
    const int  samples_per_pixel = 5;
    const int  max_depth         = 5;
    const auto aspect_ratio      = static_cast<float>(image_width) / image_height;

    // World

    hittable_list **scene_ptr;
    checkCudaErrors(cudaMalloc((void **)&scene_ptr, 2 * sizeof(hittable_list *)));

    setup_secne<<<1, 1>>>(scene_ptr);

    // Camera

    point3 lookfrom(278, 278, -800);
    point3 lookat(278, 278, 0);
    vec3   vup(0, 1, 0);
    auto   dist_to_focus = 10.0f;
    auto   aperture      = 0.001f;
    auto   vfov          = 40.0f;
    auto   time0         = 0.0f;
    auto   time1         = 1.0f;

    camera *cam;
    checkCudaErrors(cudaMallocManaged((void **)&cam, sizeof(camera)));
    *cam = camera(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus, time0, time1);

    // Render Init

    Window window(image_width, image_height, "Ray tracing (GPU)");
    size_t num_pixels = image_width * image_height;
    color *frame_buffer;
    checkCudaErrors(cudaMallocManaged((void **)&frame_buffer, num_pixels * sizeof(color)));

    dim3 blocks((image_width + thread_width - 1) / thread_width,
                (image_height + thread_height - 1) / thread_height);
    dim3 threads(thread_width, thread_height);

    // Render

    auto start_time = std::chrono::high_resolution_clock::now();

    ray_radiance<<<blocks, threads>>>(frame_buffer,
                                      image_width,
                                      image_height,
                                      samples_per_pixel,
                                      max_depth,
                                      cam,
                                      scene_ptr);
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
            size_t pixel_index = j * image_width + i;
            color  c           = radiance_to_color(frame_buffer[pixel_index], samples_per_pixel);
            *window(i, image_height - 1 - j) = color_to_rgb_integer(c);
        }
    }

    window.update();
    while (window.is_run()) {
        window.dispatch();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Cleanup

    cleanup_secne<<<1, 1>>>(scene_ptr);

    checkCudaErrors(cudaFree(scene_ptr));
    checkCudaErrors(cudaFree(cam));
    checkCudaErrors(cudaFree(frame_buffer));
}
