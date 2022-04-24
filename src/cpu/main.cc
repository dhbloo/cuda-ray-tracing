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
#include <iomanip>
#include <iostream>
#include <thread>
#ifdef WITH_OPENMP
    #include <omp.h>
#endif

color ray_radiance(ray             r,
                   color           background,
                   const hittable &world,
                   const hittable &lights,
                   int             depth)
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

void cornell_box(hittable_list &objects, hittable_list &lights)
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

int main()
{
    // Image

    const int  image_width       = 600;
    const int  image_height      = 600;
    const auto aspect_ratio      = static_cast<float>(image_width) / image_height;
    const int  samples_per_pixel = 10;
    const int  max_depth         = 2;

    // World

    hittable_list world;
    hittable_list lights;
    cornell_box(world, lights);

    color background(0, 0, 0);

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
                auto u = (i + random_float()) / (image_width - 1);
                auto v = (j + random_float()) / (image_height - 1);
                ray  r = cam.get_ray(u, v);
                radiance += ray_radiance(r, background, world, lights, max_depth);
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
