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

// headers
#include "aarect.h"
#include "box.h"
#include "camera.h"
#include "color.h"
#include "external/window.h"
#include "hittable_list.h"
#include "material.h"
#include "rtweekend.h"
#include "sphere.h"

#include <atomic>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
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

// ======================================================

__device__ color
ray_radiance(ray r, color background, const hittable &world, const hittable &lights, int depth)
{
    hit_record rec;
    color      accumL, accumR(1.0f, 1.0f, 1.0f);

    // If we've exceeded the ray bounce limit, no more light is gathered.
    for (; depth > 0; depth--) {
        // If the ray hits nothing, return the background color.
        if (!world.hit(r, 0.001f, infinity, rec)) {
            accumL += accumR * background;
            break;
        }

        scatter_record srec;
        accumL += accumR * rec.mat_ptr->emitted(r, rec, rec.u, rec.v, rec.p);

        if (!rec.mat_ptr->scatter(r, rec, srec))
            break;
        else
            accumR = accumR * srec.attenuation;

        if (srec.is_specular) {
            r = srec.specular_ray;
        }
        else {
            hittable_pdf light(lights, rec.p);
            mixture_pdf  p(light, *srec.pdf_ptr);
            ray          scattered = ray(rec.p, p.generate(), r.time());
            auto         pdf_val   = p.value(scattered.direction());

            accumR = accumR * rec.mat_ptr->scattering_pdf(r, rec, scattered) / pdf_val;
            r      = scattered;
        }
    }

    return accumL;
}

__device__ void cornell_box(hittable_list &objects, hittable_list &lights)
{
    auto red   = make_shared<lambertian>(color(.65f, .05f, .05f));
    auto white = make_shared<lambertian>(color(.73f, .73f, .73f));
    auto green = make_shared<lambertian>(color(.12f, .45f, .15f));
    auto light = make_shared<diffuse_light>(color(15, 15, 15));

    objects.add(make_shared<yz_rect>(0, 555, 0, 555, 555, green));
    objects.add(make_shared<yz_rect>(0, 555, 0, 555, 0, red));
    objects.add(make_shared<flip_face>(make_shared<xz_rect>(213, 343, 227, 332, 554, light)));
    objects.add(make_shared<xz_rect>(0, 555, 0, 555, 555, white));
    objects.add(make_shared<xz_rect>(0, 555, 0, 555, 0, white));
    objects.add(make_shared<xy_rect>(0, 555, 0, 555, 555, white));

    shared_ptr<material> aluminum = make_shared<metal>(color(0.8f, 0.85f, 0.88f), 0.0f);
    shared_ptr<hittable> box1 = make_shared<box>(point3(0, 0, 0), point3(165, 330, 165), aluminum);
    box1                      = make_shared<rotate_y>(box1, 15);
    box1                      = make_shared<translate>(box1, vec3(265, 0, 295));
    objects.add(box1);

    auto glass = make_shared<dielectric>(1.5);
    objects.add(make_shared<sphere>(point3(190, 90, 190), 90, glass));

    lights.add(make_shared<xz_rect>(213, 343, 227, 332, 554, make_shared<material>()));
    lights.add(make_shared<sphere>(point3(190, 90, 190), 90, make_shared<material>()));
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

// 初始化随机数状态

__global__ void rand_init(int width, int height, curandState *rand_state)
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

__global__ void render(color          *fb,
                       int             width,
                       int             height,
                       int             samples_per_pixel,
                       int             max_depth,
                       color           background,
                       camera         *cam,
                       hittable_list **scene_ptr)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if (i >= width || j >= height)
        return;

    color radiance(0, 0, 0);
    for (int s = 0; s < samples_per_pixel; ++s) {
        auto u = (i + random_float()) / (width - 1);
        auto v = (j + random_float()) / (height - 1);
        ray  r = cam->get_ray(u, v);
        radiance += ray_radiance(r, background, *scene_ptr[0], *scene_ptr[1], max_depth);
    }

    fb[j * width + i] = radiance_to_color(radiance, samples_per_pixel);
}

int main()
{
    // Image

    const int  image_width       = 600;
    const int  image_height      = 600;
    const int  thread_width      = 8;
    const int  thread_height     = 8;
    const int  samples_per_pixel = 100;
    const int  max_depth         = 2;
    const auto aspect_ratio      = static_cast<float>(image_width) / image_height;

    color background(0, 0, 0);

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

    curandState *rand_state;
    checkCudaErrors(cudaMalloc((void **)&rand_state, num_pixels * sizeof(curandState)));

    dim3 blocks((image_width + thread_width - 1) / thread_width,
                (image_height + thread_height - 1) / thread_height);
    dim3 threads(thread_width, thread_height);

    rand_init<<<blocks, threads>>>(image_width, image_height, rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Render

    auto start_time = std::chrono::high_resolution_clock::now();

    render<<<blocks, threads>>>(frame_buffer,
                                image_width,
                                image_height,
                                samples_per_pixel,
                                max_depth,
                                background,
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

    cleanup_secne<<<1, 1>>>(scene_ptr);

    checkCudaErrors(cudaFree(scene_ptr));
    checkCudaErrors(cudaFree(cam));
    checkCudaErrors(cudaFree(frame_buffer));
    checkCudaErrors(cudaFree(rand_state));
}
