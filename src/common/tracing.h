#ifndef TRACING_H
#define TRACING_H
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
#include "constant_medium.h"
#include "hittable_list.h"
#include "material.h"
#include "rtweekend.h"
#include "sphere.h"

__device__ color background_radiance(ray r)
{
    return color(0, 0, 0);

    // auto t = 0.5f * (r.direction().y() + 1.0f);
    // auto c = (1.0f - t) * color(1, 1, 1) + t * color(0.5f, 0.7f, 1.0f);
    // return c;
}

__device__ color ray_radiance(ray r, const hittable &world, const hittable &lights, int depth)
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
    auto light = make_shared<diffuse_light>(color(20, 20, 20));

    auto top_light = make_shared<xz_rect>(213, 343, 227, 332, 554, light);

    objects.add(make_shared<yz_rect>(0, 555, 0, 555, 555, green));
    objects.add(make_shared<yz_rect>(0, 555, 0, 555, 0, red));
    objects.add(make_shared<flip_face>(top_light));
    objects.add(make_shared<xz_rect>(0, 555, 0, 555, 555, white));
    objects.add(make_shared<xz_rect>(0, 555, 0, 555, 0, white));
    objects.add(make_shared<xy_rect>(0, 555, 0, 555, 555, white));

    shared_ptr<material> aluminum = make_shared<metal>(color(0.8f, 0.85f, 0.88f), 0.6f);
    shared_ptr<hittable> box1 = make_shared<box>(point3(0, 0, 0), point3(165, 330, 165), aluminum);
    box1                      = make_shared<rotate_y>(box1, 15);
    box1                      = make_shared<translate>(box1, vec3(265, 0, 295));
    objects.add(box1);

    auto glass      = make_shared<dielectric>(1.5);
    auto glass_ball = make_shared<sphere>(point3(190, 90, 190), 90, glass);
    objects.add(glass_ball);

    auto ss_ball = make_shared<sphere>(point3(420, 60, 120), 60, glass);
    objects.add(ss_ball);
    objects.add(
        make_shared<constant_medium>(ss_ball, 0.015f, make_shared<solid_color>(0.2f, 0.4f, 0.9f)));

    //auto boundary =
    //    make_shared<box>(point3(0, 400, 0), point3(555, 555, 555), make_shared<material>());
    //auto fog = make_shared<constant_medium>(boundary, 0.0006f, make_shared<solid_color>(1, 1, 1));
    //objects.add(fog);

    lights.add(top_light);
    lights.add(glass_ball);
    lights.add(ss_ball);
}

#endif
