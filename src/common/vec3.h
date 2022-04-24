#ifndef VEC3_H
#define VEC3_H
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

#include "rtweekend.h"

#include <cmath>
#include <iostream>

class vec3
{
public:
    __dual__ vec3() : e {0, 0, 0} {}
    __dual__ vec3(float e0, float e1, float e2) : e {e0, e1, e2} {}

    __dual__ float x() const { return e[0]; }
    __dual__ float y() const { return e[1]; }
    __dual__ float z() const { return e[2]; }

    __dual__ vec3   operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __dual__ float  operator[](int i) const { return e[i]; }
    __dual__ float &operator[](int i) { return e[i]; }

    __dual__ vec3 &operator+=(const vec3 &v)
    {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __dual__ vec3 &operator*=(const float t)
    {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __dual__ vec3 &operator/=(const float t) { return *this *= 1 / t; }

    __dual__ float length() const { return sqrt(length_squared()); }

    __dual__ float length_squared() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }

    __dual__ bool near_zero() const
    {
        // Return true if the vector is close to zero in all dimensions.
        const float s = 1e-7f;
        return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
    }

    __device__ inline static vec3 random()
    {
        return vec3(random_float(), random_float(), random_float());
    }

    __device__ inline static vec3 random(float min, float max)
    {
        return vec3(random_float(min, max), random_float(min, max), random_float(min, max));
    }

public:
    float e[3];
};

// Type aliases for vec3
using point3 = vec3;  // 3D point
using color  = vec3;  // RGB color

// vec3 Utility Functions

__host__ inline std::ostream &operator<<(std::ostream &out, const vec3 &v)
{
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__dual__ inline vec3 operator+(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__dual__ inline vec3 operator-(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__dual__ inline vec3 operator*(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__dual__ inline vec3 operator*(float t, const vec3 &v)
{
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__dual__ inline vec3 operator*(const vec3 &v, float t)
{
    return t * v;
}

__dual__ inline vec3 operator/(vec3 v, float t)
{
    return (1 / t) * v;
}

__dual__ inline float dot(const vec3 &u, const vec3 &v)
{
    return u.e[0] * v.e[0] + u.e[1] * v.e[1] + u.e[2] * v.e[2];
}

__dual__ inline vec3 cross(const vec3 &u, const vec3 &v)
{
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
                u.e[2] * v.e[0] - u.e[0] * v.e[2],
                u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__dual__ inline vec3 unit_vector(vec3 v)
{
    return v / v.length();
}

__device__ inline vec3 random_in_unit_disk()
{
    while (true) {
        auto p = vec3(random_float(-1, 1), random_float(-1, 1), 0);
        if (p.length_squared() >= 1)
            continue;
        return p;
    }
}

__device__ inline vec3 random_in_unit_sphere()
{
    while (true) {
        auto p = vec3::random(-1, 1);
        if (p.length_squared() >= 1)
            continue;
        return p;
    }
}

__device__ inline vec3 random_unit_vector()
{
    return unit_vector(random_in_unit_sphere());
}

__device__ inline vec3 random_in_hemisphere(const vec3 &normal)
{
    vec3 in_unit_sphere = random_in_unit_sphere();
    if (dot(in_unit_sphere, normal) > 0.0)  // In the same hemisphere as the normal
        return in_unit_sphere;
    else
        return -in_unit_sphere;
}

__dual__ inline vec3 reflect(const vec3 &v, const vec3 &n)
{
    return v - 2 * dot(v, n) * n;
}

__dual__ inline vec3 refract(const vec3 &uv, const vec3 &n, float etai_over_etat)
{
    auto cos_theta      = fmin(dot(-uv, n), 1.0f);
    vec3 r_out_perp     = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -sqrt(fabs(1.0f - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}

#endif
