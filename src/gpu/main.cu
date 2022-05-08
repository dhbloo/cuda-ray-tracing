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

__global__ void setup_scene(hittable_list **scene_ptr)
{
    scene_ptr[0] = new hittable_list;
    scene_ptr[1] = new hittable_list;
    cornell_box(*scene_ptr[0], *scene_ptr[1]);
}

// 构造场景

__global__ void cleanup_scene(hittable_list **scene_ptr)
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

struct render_data
{
    int         *x;
    int         *y;
    int         *depth;
    curandState *rand_state;
    ray         *r;
    color       *accumL;
    color       *accumR;
    hit_record  *rec;

    // managed memory
    color *m_radiance;
    int   *m_num_samples;
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

__global__ void init_pixel_data(const render_data rd, int width, int height)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= width || j >= height)
        return;

    int idx       = j * width + i;
    rd.x[idx]     = i;
    rd.y[idx]     = j;
    rd.depth[idx] = 0;
    curand_init(42 + idx, 0, 0, &rd.rand_state[idx]);

    rd.m_radiance[idx]    = color();
    rd.m_num_samples[idx] = 0;
}

__global__ void generate_rays(const render_data rd,
                              int               offset,
                              int               num_pixels,
                              int               width,
                              int               height,
                              int               max_depth,
                              const camera     *cam)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x + offset;
    if (idx >= num_pixels)
        return;

    if (rd.depth[idx] <= 0) {
        float u = (rd.x[idx] + random_float(&rd.rand_state[idx])) / (width - 1);
        float v = (rd.y[idx] + random_float(&rd.rand_state[idx])) / (height - 1);

        int frame_idx = rd.y[idx] * width + rd.x[idx];
        rd.m_radiance[frame_idx] += rd.accumL[idx];
        rd.m_num_samples[frame_idx]++;

        rd.r[idx]      = cam->get_ray(&rd.rand_state[idx], u, v);
        rd.accumL[idx] = color();
        rd.accumR[idx] = color(1.0f, 1.0f, 1.0f);
        rd.depth[idx]  = max_depth;
    }
}

__global__ void hit_world(const render_data rd, int offset, int num_pixels, hittable_list **scene)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x + offset;
    if (idx >= num_pixels)
        return;

    hit_record hit_rec;
    ray        r = rd.r[idx];

    if (scene[0]->hit(r, 0.001f, infinity, hit_rec)) {
        rd.rec[idx] = hit_rec;
    }
    else {
        rd.accumL[idx] += rd.accumR[idx] * background_radiance(r);
        rd.depth[idx] = 0;
    }
}

__global__ void scatter(const render_data rd, int offset, int num_pixels, hittable_list **scene)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x + offset;
    if (idx >= num_pixels)
        return;

    if (rd.depth[idx] <= 0)
        return;

    scatter_record srec;
    ray            r   = rd.r[idx];
    hit_record     rec = rd.rec[idx];

    rd.accumL[idx] += rd.accumR[idx] * rec.mat_ptr->emitted(r, rec, rec.u, rec.v, rec.p);

    if (!rec.mat_ptr->scatter(r, rec, srec, &rd.rand_state[idx])) {
        rd.depth[idx] = 0;
    }
    else {
        rd.accumR[idx] = rd.accumR[idx] * srec.attenuation;

        if (srec.is_specular) {
            rd.r[idx] = srec.specular_ray;
        }
        else {
            hittable_pdf light(*scene[1], rec.p);
            mixture_pdf  p(light, *srec.pdf_ptr);
            ray          scattered = ray(rec.p, p.generate(&rd.rand_state[idx]));
            auto         pdf_val   = p.value(scattered.direction());

            rd.accumR[idx] *=
                rec.mat_ptr->scattering_pdf(r, rec, scattered, &rd.rand_state[idx]) / pdf_val;
            rd.r[idx] = scattered;
        }

        rd.depth[idx]--;
    }
}

void setup_render_data(render_data &rd, int width, int height)
{
    int n = width * height;
    checkCudaErrors(cudaMalloc((void **)&rd.x, n * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&rd.y, n * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&rd.depth, n * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **)&rd.rand_state, n * sizeof(curandState)));
    checkCudaErrors(cudaMalloc((void **)&rd.r, n * sizeof(ray)));
    checkCudaErrors(cudaMalloc((void **)&rd.accumL, n * sizeof(color)));
    checkCudaErrors(cudaMalloc((void **)&rd.accumR, n * sizeof(color)));
    checkCudaErrors(cudaMalloc((void **)&rd.rec, n * sizeof(hit_record)));

    checkCudaErrors(cudaMallocManaged((void **)&rd.m_radiance, n * sizeof(color)));
    checkCudaErrors(cudaMallocManaged((void **)&rd.m_num_samples, n * sizeof(int)));
}

void cleanup_render_data(render_data &rd)
{
    checkCudaErrors(cudaFree(rd.x));
    checkCudaErrors(cudaFree(rd.y));
    checkCudaErrors(cudaFree(rd.depth));
    checkCudaErrors(cudaFree(rd.rand_state));
    checkCudaErrors(cudaFree(rd.r));
    checkCudaErrors(cudaFree(rd.accumL));
    checkCudaErrors(cudaFree(rd.accumR));
    checkCudaErrors(cudaFree(rd.rec));

    checkCudaErrors(cudaFree(rd.m_radiance));
    checkCudaErrors(cudaFree(rd.m_num_samples));
}

void render(const render_data &rd,
            int                width,
            int                height,
            int                samples_per_pixel,
            int                max_depth,
            camera            *cam,
            hittable_list    **scene)
{
    const int num_pixels = width * height;
    const int threads    = 512;
    const int blocks     = (num_pixels + threads - 1) / threads;
    const int offset     = 0;

    for (int i = 0; i < samples_per_pixel; i++) {
        generate_rays<<<blocks, threads>>>(rd, offset, num_pixels, width, height, max_depth, cam);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        hit_world<<<blocks, threads>>>(rd, offset, num_pixels, scene);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());

        scatter<<<blocks, threads>>>(rd, offset, num_pixels, scene);
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    // accumulate colors back to frame buffer for completed rays

    generate_rays<<<blocks, threads>>>(rd, offset, num_pixels, width, height, max_depth, cam);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

int main()
{
    // Image

    const int  image_width       = 800;
    const int  image_height      = 800;
    const int  samples_per_pixel = 500;
    const int  max_depth         = 10;
    const auto aspect_ratio      = static_cast<float>(image_width) / image_height;

    // World

    hittable_list **scene_ptr;
    checkCudaErrors(cudaMalloc((void **)&scene_ptr, 2 * sizeof(hittable_list *)));

    setup_scene<<<1, 1>>>(scene_ptr);

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

    render_data rd;
    setup_render_data(rd, image_width, image_height);

    const int thread_width  = 32;
    const int thread_height = 16;
    dim3      blocks((image_width + thread_width - 1) / thread_width,
                (image_height + thread_height - 1) / thread_height);
    dim3      threads(thread_width, thread_height);

    init_pixel_data<<<blocks, threads>>>(rd, image_width, image_height);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Render

    auto start_time = std::chrono::high_resolution_clock::now();

    render(rd, image_width, image_height, samples_per_pixel, max_depth, cam, scene_ptr);

    auto  end_time   = std::chrono::high_resolution_clock::now();
    auto  elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    float elapsed_seconds = elapsed_ms.count() * 0.001f;
    int64_t total_rays    = (int64_t)samples_per_pixel * image_height * image_width;
    std::cout << std::fixed << std::setprecision(3) << "\nDone after " << elapsed_seconds
              << " seconds, " << total_rays / elapsed_seconds / 1000000.f << " Mrays per second.\n";

    // Copy frame buffer to window
    for (int j = image_height - 1; j >= 0; --j) {
        for (int i = 0; i < image_width; ++i) {
            int   idx = j * image_width + i;
            color c   = radiance_to_color(rd.m_radiance[idx], rd.m_num_samples[idx]);
            *window(i, image_height - 1 - j) = color_to_rgb_integer(c);
        }
    }

    window.update();
    while (window.is_run()) {
        window.dispatch();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Cleanup

    cleanup_render_data(rd);
    cleanup_scene<<<1, 1>>>(scene_ptr);
    checkCudaErrors(cudaFree(scene_ptr));
    checkCudaErrors(cudaFree(cam));
}
