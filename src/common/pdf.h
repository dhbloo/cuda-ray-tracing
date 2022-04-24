#ifndef PDF_H
#define PDF_H
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

#include "onb.h"
#include "rtweekend.h"

__device__ inline vec3 random_cosine_direction()
{
    auto r1 = random_float();
    auto r2 = random_float();
    auto z  = sqrt(1 - r2);

    auto phi = 2 * pi * r1;
    auto x   = cos(phi) * sqrt(r2);
    auto y   = sin(phi) * sqrt(r2);

    return vec3(x, y, z);
}

__device__ inline vec3 random_to_sphere(float radius, float distance_squared)
{
    auto r1 = random_float();
    auto r2 = random_float();
    auto z  = 1 + r2 * (sqrt(1 - radius * radius / distance_squared) - 1);

    auto phi = 2 * pi * r1;
    auto x   = cos(phi) * sqrt(1 - z * z);
    auto y   = sin(phi) * sqrt(1 - z * z);

    return vec3(x, y, z);
}

class pdf
{
public:
    __device__ virtual ~pdf() {}
    __device__ virtual float value(const vec3 &direction) const = 0;
    __device__ virtual vec3  generate() const                   = 0;
};

class cosine_pdf : public pdf
{
public:
    __device__ cosine_pdf() {}
    __device__ cosine_pdf(const vec3 &w) { uvw.build_from_w(w); }

    __device__ virtual float value(const vec3 &direction) const override
    {
        auto cosine = dot(unit_vector(direction), uvw.w());
        return (cosine <= 0) ? 0 : cosine / pi;
    }

    __device__ virtual vec3 generate() const override
    {
        return uvw.local(random_cosine_direction());
    }

public:
    onb uvw;
};

class hittable_pdf : public pdf
{
public:
    __device__ hittable_pdf(const hittable &h, const point3 &origin) : h(h), o(origin) {}
    __device__ virtual float value(const vec3 &direction) const override
    {
        return h.pdf_value(o, direction);
    }
    __device__ virtual vec3 generate() const override { return h.random(o); }

public:
    point3          o;
    const hittable &h;
};

class mixture_pdf : public pdf
{
public:
    __device__ mixture_pdf(const pdf &p0, const pdf &p1) : p0(p0), p1(p1) {}

    __device__ virtual float value(const vec3 &direction) const override
    {
        return 0.5f * p0.value(direction) + 0.5f * p1.value(direction);
    }

    __device__ virtual vec3 generate() const override
    {
        if (random_float() < 0.5)
            return p0.generate();
        else
            return p1.generate();
    }

public:
    const pdf &p0, &p1;
};

#endif
