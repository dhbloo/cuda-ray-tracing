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

#include "camera.h"
#include "color.h"
#include "hittable_list.h"
#include "material.h"
#include "rtweekend.h"
#include "scene.h"

// window
#include "external/window.h"

#include <atomic>
#include <iomanip>
#include <iostream>
#include <thread>
#ifdef WITH_OPENMP
    #include <omp.h>
#endif

color ray_radiance(ray r, const hittable &world, const hittable &lights, int depth)
{
    hit_record rec;
    color      accumL, accumR(1.0f, 1.0f, 1.0f);

    // If we've exceeded the ray bounce limit, no more light is gathered.
    for (; depth > 0; depth--) {
        // If the ray hits nothing, return the background color.
        if (!world.hit(r, 0.001f, infinity, rec)) {
            accumL += accumR * background_radiance(r);
            break;
        }

        scatter_record srec;
        accumL += accumR * rec.mat_ptr->emitted(r, rec, rec.u, rec.v, rec.p);

        if (!rec.mat_ptr->scatter(r, rec, srec, 0))
            break;
        else
            accumR = accumR * srec.attenuation;

        if (srec.is_specular) {
            r = srec.specular_ray;
        }
        else {
            hittable_pdf light(lights, rec.p);
            mixture_pdf  p(light, *srec.pdf_ptr);
            ray          scattered = ray(rec.p, p.generate(0));
            auto         pdf_val   = p.value(scattered.direction());

            accumR = accumR * rec.mat_ptr->scattering_pdf(r, rec, scattered, 0) / pdf_val;
            r      = scattered;
        }
    }

    return accumL;
}

int main()
{
    // Image

    const int  image_width       = 800;
    const int  image_height      = 800;
    const auto aspect_ratio      = static_cast<float>(image_width) / image_height;
    const int  samples_per_pixel = 50;
    const int  max_depth         = 5;

    // World

    hittable_list world;
    hittable_list lights;
    cornell_box(world, lights);

    // Camera

    point3 lookfrom(278, 278, -800);
    point3 lookat(278, 278, 0);
    vec3   vup(0, 1, 0);
    auto   dist_to_focus = 10.0f;
    auto   aperture      = 0.001f;
    auto   vfov          = 40.0f;
    auto   time0         = 0.0f;
    auto   time1         = 1.0f;

    camera cam(lookfrom, lookat, vup, vfov, aspect_ratio, aperture, dist_to_focus, time0, time1);

    // Render

    Window           window(image_width, image_height, "Ray tracing (CPU)");
    std::atomic<int> scanline_remained = image_height;
    auto             start_time        = std::chrono::high_resolution_clock::now();

#ifdef WITH_OPENMP
    omp_lock_t lock;
    omp_init_lock(&lock);
    omp_set_num_threads(8);
    #pragma omp parallel for
#endif
    for (int j = image_height - 1; j >= 0; --j) {
#ifdef WITH_OPENMP
        if (omp_test_lock(&lock))
#endif
        {
            std::cout << "\rScanlines remaining: "
                      << scanline_remained.load(std::memory_order_relaxed) << ' ' << std::flush;
            window.update();
            if (!window.is_run())
                std::exit(0);
#ifdef WITH_OPENMP
            omp_unset_lock(&lock);
#endif
        }

        for (int i = 0; i < image_width; ++i) {
            color radiance(0, 0, 0);
            for (int s = 0; s < samples_per_pixel; ++s) {
                auto u = (i + random_float(0)) / (image_width - 1);
                auto v = (j + random_float(0)) / (image_height - 1);
                ray  r = cam.get_ray(0, u, v);
                radiance += ray_radiance(r, world, lights, max_depth);
            }
            color pixel_color                = radiance_to_color(radiance, samples_per_pixel);
            *window(i, image_height - 1 - j) = color_to_rgb_integer(pixel_color);
        }

        scanline_remained.fetch_sub(1, std::memory_order_relaxed);
    }

    auto  end_time   = std::chrono::high_resolution_clock::now();
    auto  elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    float elapsed_seconds = elapsed_ms.count() * 0.001f;
    int64_t total_rays    = (int64_t)samples_per_pixel * image_height * image_width;
    std::cout << std::fixed << std::setprecision(3) << "\nDone after " << elapsed_seconds
              << " seconds, " << total_rays / elapsed_seconds / 1000000.f << " Mrays per second.\n";

    window.update();
    while (window.is_run()) {
        window.dispatch();
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}
