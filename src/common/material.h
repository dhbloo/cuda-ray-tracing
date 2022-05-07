#ifndef MATERIAL_H
#define MATERIAL_H
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

#include "hittable.h"
#include "pdf.h"
#include "rtweekend.h"
#include "texture.h"

struct scatter_record
{
    ray        specular_ray;
    bool       is_specular;
    color      attenuation;
    const pdf *pdf_ptr = nullptr;

    cosine_pdf pdf;  // get rid of dynamic malloc
};

class material
{
public:
    virtual color __device__
    emitted(const ray &r_in, const hit_record &rec, float u, float v, const point3 &p) const
    {
        return color(0, 0, 0);
    }

    __device__ virtual bool
    scatter(const ray &r_in, const hit_record &rec, scatter_record &srec) const
    {
        return false;
    }

    __device__ virtual float
    scattering_pdf(const ray &r_in, const hit_record &rec, const ray &scattered) const
    {
        return 0;
    }
};

class lambertian : public material
{
public:
    __device__ lambertian(const color &a) : albedo(make_shared<solid_color>(a)) {}
    __device__ lambertian(shared_ptr<tex> a) : albedo(a) {}

    __device__ virtual bool
    scatter(const ray &r_in, const hit_record &rec, scatter_record &srec) const override
    {
        srec.is_specular = false;
        srec.attenuation = albedo->value(rec.u, rec.v, rec.p);
        srec.pdf         = cosine_pdf(rec.normal);
        srec.pdf_ptr     = &srec.pdf;
        return true;
    }

    __device__ float
    scattering_pdf(const ray &r_in, const hit_record &rec, const ray &scattered) const override
    {
        auto cosine = dot(rec.normal, unit_vector(scattered.direction()));
        return cosine < 0 ? 0 : cosine / pi;
    }

public:
    shared_ptr<tex> albedo;
};

class metal : public material
{
public:
    __device__ metal(const color &a, float f) : albedo(a), fuzz(f < 1 ? f : 1) {}

    __device__ virtual bool
    scatter(const ray &r_in, const hit_record &rec, scatter_record &srec) const override
    {
        vec3 reflected    = reflect(unit_vector(r_in.direction()), rec.normal);
        srec.specular_ray = ray(rec.p, reflected + fuzz * random_in_unit_sphere());
        srec.attenuation  = albedo;
        srec.is_specular  = true;
        srec.pdf_ptr      = nullptr;
        return true;
    }

public:
    color albedo;
    float fuzz;
};

class dielectric : public material
{
public:
    __device__ dielectric(float index_of_refraction) : ir(index_of_refraction) {}

    __device__ virtual bool
    scatter(const ray &r_in, const hit_record &rec, scatter_record &srec) const override
    {
        srec.is_specular       = true;
        srec.pdf_ptr           = nullptr;
        srec.attenuation       = color(1.0f, 1.0f, 1.0f);
        float refraction_ratio = rec.front_face ? (1.0f / ir) : ir;

        vec3  unit_direction = unit_vector(r_in.direction());
        float cos_theta      = fmin(dot(-unit_direction, rec.normal), 1.0f);
        float sin_theta      = sqrt(1.0f - cos_theta * cos_theta);

        bool cannot_refract = refraction_ratio * sin_theta > 1.0f;
        vec3 direction;

        if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_float())
            direction = reflect(unit_direction, rec.normal);
        else
            direction = refract(unit_direction, rec.normal, refraction_ratio);

        srec.specular_ray = ray(rec.p, direction);
        return true;
    }

public:
    float ir;  // Index of Refraction

private:
    __device__ static float reflectance(float cosine, float ref_idx)
    {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - ref_idx) / (1 + ref_idx);
        r0      = r0 * r0;
        return r0 + (1 - r0) * powf(1 - cosine, 5);
    }
};

class diffuse_light : public material
{
public:
    __device__ diffuse_light(shared_ptr<tex> a) : emit(a) {}
    __device__ diffuse_light(color c) : emit(make_shared<solid_color>(c)) {}

    __device__ virtual color emitted(const ray        &r_in,
                                     const hit_record &rec,
                                     float             u,
                                     float             v,
                                     const point3     &p) const override
    {
        if (!rec.front_face)
            return color(0, 0, 0);
        return emit->value(u, v, p);
    }

public:
    shared_ptr<tex> emit;
};

class isotropic : public material
{
public:
    __device__ isotropic(color c) : albedo(make_shared<solid_color>(c)) {}
    __device__ isotropic(shared_ptr<tex> a) : albedo(a) {}

    __device__ virtual bool
    scatter(const ray &r_in, const hit_record &rec, scatter_record &srec) const override
    {
        srec.specular_ray = ray(rec.p, random_in_unit_sphere());
        srec.attenuation  = albedo->value(rec.u, rec.v, rec.p);
        srec.is_specular  = true;
        srec.pdf_ptr      = nullptr;
        return true;
    }

public:
    shared_ptr<tex> albedo;
};

#endif
