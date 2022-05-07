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

struct sample
{
    color radiance;
    int   num_samples;
};

struct pixel_data
{
    int         x, y;
    int         depth;
    curandState rand_state;
    ray         r;
    color       accumL, accumR;
    hit_record  rec;
};

__global__ void
init_pixel_data(pixel_data *pixel_buffer, sample *frame_buffer, int width, int height)

{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= width || j >= height)
        return;

    int id           = j * width + i;
    frame_buffer[id] = {color(), 0};

    pixel_data &pd = pixel_buffer[id];
    pd.x           = i;
    pd.y           = j;
    pd.depth       = 0;
    curand_init(42 + id, 0, 0, &pd.rand_state);
}

__global__ void generate_rays(pixel_data *pixel_buffer,
                              sample     *frame_buffer,
                              int         num_pixels,
                              int         width,
                              int         height,
                              int         max_depth,
                              camera     *cam)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_pixels)
        return;

    pixel_data pd = pixel_buffer[idx];

    if (pd.depth <= 0) {
        float u = (pd.x + random_float(&pd.rand_state)) / (width - 1);
        float v = (pd.y + random_float(&pd.rand_state)) / (height - 1);

        sample &s = frame_buffer[pd.y * width + pd.x];
        s.radiance += pd.accumL;
        s.num_samples++;

        pd.r      = cam->get_ray(&pd.rand_state, u, v);
        pd.accumL = color();
        pd.accumR = color(1.0f, 1.0f, 1.0f);
        pd.depth  = max_depth;

        pixel_buffer[idx] = pd;
    }
}

__global__ void hit_world(pixel_data *pixel_buffer, int num_pixels, hittable_list **scene)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_pixels)
        return;

    pixel_data &pd = pixel_buffer[idx];
    hit_record  hit_rec;

    if (scene[0]->hit(pd.r, 0.001f, infinity, hit_rec)) {
        pd.rec = hit_rec;
    }
    else {
        pd.accumL += pd.accumR * background_radiance(pd.r);
        pd.depth = 0;
    }
}

__global__ void scatter(pixel_data *pixel_buffer, int num_pixels, hittable_list **scene)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_pixels)
        return;

    pixel_data    &pd = pixel_buffer[idx];
    scatter_record srec;

    if (pd.depth <= 0)
        return;

    pd.accumL += pd.accumR * pd.rec.mat_ptr->emitted(pd.r, pd.rec, pd.rec.u, pd.rec.v, pd.rec.p);

    if (!pd.rec.mat_ptr->scatter(pd.r, pd.rec, srec, &pd.rand_state)) {
        pd.depth = 0;
    }
    else {
        pd.accumR = pd.accumR * srec.attenuation;

        if (srec.is_specular) {
            pd.r = srec.specular_ray;
        }
        else {
            hittable_pdf light(*scene[1], pd.rec.p);
            mixture_pdf  p(light, *srec.pdf_ptr);
            ray          scattered = ray(pd.rec.p, p.generate(&pd.rand_state));
            auto         pdf_val   = p.value(scattered.direction());

            pd.accumR *=
                pd.rec.mat_ptr->scattering_pdf(pd.r, pd.rec, scattered, &pd.rand_state) / pdf_val;
            pd.r = scattered;
        }

        pd.depth--;
    }
}

void render(pixel_data     *pixel_buffer,
            sample         *frame_buffer,
            int             width,
            int             height,
            int             samples_per_pixel,
            int             max_depth,
            camera         *cam,
            hittable_list **scene)
{
    const int num_pixels = width * height;
    const int threads    = 512;
    const int blocks     = (num_pixels + threads - 1) / threads;

    for (int i = 0; i < samples_per_pixel; i++) {
        generate_rays<<<blocks, threads>>>(pixel_buffer,
                                           frame_buffer,
                                           num_pixels,
                                           width,
                                           height,
                                           max_depth,
                                           cam);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        hit_world<<<blocks, threads>>>(pixel_buffer, num_pixels, scene);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        scatter<<<blocks, threads>>>(pixel_buffer, num_pixels, scene);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    // accumulate colors back to frame buffer for completed rays

    generate_rays<<<blocks, threads>>>(pixel_buffer,
                                       frame_buffer,
                                       num_pixels,
                                       width,
                                       height,
                                       max_depth,
                                       cam);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

int main()
{
    // Image

    const int  image_width       = 800;
    const int  image_height      = 800;
    const int  thread_width      = 32;
    const int  thread_height     = 16;
    const int  samples_per_pixel = 200;
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

    pixel_data *pixel_buffer;
    sample     *frame_buffer;
    checkCudaErrors(
        cudaMalloc((void **)&pixel_buffer, image_width * image_height * sizeof(pixel_data)));
    checkCudaErrors(
        cudaMallocManaged((void **)&frame_buffer, image_width * image_height * sizeof(sample)));

    dim3 blocks((image_width + thread_width - 1) / thread_width,
                (image_height + thread_height - 1) / thread_height);
    dim3 threads(thread_width, thread_height);

    init_pixel_data<<<blocks, threads>>>(pixel_buffer, frame_buffer, image_width, image_height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Render

    auto start_time = std::chrono::high_resolution_clock::now();

    render(pixel_buffer,
           frame_buffer,
           image_width,
           image_height,
           samples_per_pixel,
           max_depth,
           cam,
           scene_ptr);

    auto  end_time   = std::chrono::high_resolution_clock::now();
    auto  elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    float elapsed_seconds = elapsed_ms.count() * 0.001f;
    int64_t total_rays    = (int64_t)samples_per_pixel * image_height * image_width;
    std::cout << std::fixed << std::setprecision(3) << "\nDone after " << elapsed_seconds
              << " seconds, " << total_rays / elapsed_seconds / 1000000.f << " Mrays per second.\n";

    // Copy frame buffer to window
    for (int j = image_height - 1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            const sample &s                  = frame_buffer[j * image_width + i];
            color         c                  = radiance_to_color(s.radiance, s.num_samples);
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
    checkCudaErrors(cudaFree(pixel_buffer));
}
