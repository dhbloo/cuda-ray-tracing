#ifndef SPHERE_H
#define SPHERE_H
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
#include "onb.h"
#include "rtweekend.h"

class sphere : public hittable
{
public:
    __host__ sphere(point3 cen, float r, shared_ptr<material> m)
        : center(cen)
        , radius(r)
        , mat_ptr(m) {};

    __device__ virtual bool
    hit(const ray &r, float t_min, float t_max, hit_record &rec) const override;
    __dual__ virtual bool bounding_box(float time0, float time1, aabb &output_box) const override;
    __device__ virtual float pdf_value(const point3 &o, const vec3 &v) const override;
    __device__ virtual vec3  random(const point3 &o) const override;

public:
    point3               center;
    float                radius;
    shared_ptr<material> mat_ptr;

private:
    __device__ static void get_sphere_uv(const point3 &p, float &u, float &v)
    {
        // p: a given point on the sphere of radius one, centered at the origin.
        // u: returned value [0,1] of angle around the Y axis from X=-1.
        // v: returned value [0,1] of angle from Y=-1 to Y=+1.
        //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
        //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
        //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

        auto theta = acos(-p.y());
        auto phi   = atan2(-p.z(), p.x()) + pi;

        u = phi / (2 * pi);
        v = theta / pi;
    }
};

__device__ float sphere::pdf_value(const point3 &o, const vec3 &v) const
{
    hit_record rec;
    if (!this->hit(ray(o, v), 0.001f, infinity, rec))
        return 0;

    auto cos_theta_max = sqrt(1 - radius * radius / (center - o).length_squared());
    auto solid_angle   = 2 * pi * (1 - cos_theta_max);

    return 1 / solid_angle;
}

__device__ vec3 sphere::random(const point3 &o) const
{
    vec3 direction        = center - o;
    auto distance_squared = direction.length_squared();
    onb  uvw;
    uvw.build_from_w(direction);
    return uvw.local(random_to_sphere(radius, distance_squared));
}

__dual__ bool sphere::bounding_box(float time0, float time1, aabb &output_box) const
{
    output_box = aabb(center - vec3(radius, radius, radius), center + vec3(radius, radius, radius));
    return true;
}

__device__ bool sphere::hit(const ray &r, float t_min, float t_max, hit_record &rec) const
{
    vec3 oc     = r.origin() - center;
    auto a      = r.direction().length_squared();
    auto half_b = dot(oc, r.direction());
    auto c      = oc.length_squared() - radius * radius;

    auto discriminant = half_b * half_b - a * c;
    if (discriminant < 0)
        return false;
    auto sqrtd = sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    auto root = (-half_b - sqrtd) / a;
    if (root < t_min || t_max < root) {
        root = (-half_b + sqrtd) / a;
        if (root < t_min || t_max < root)
            return false;
    }

    rec.t               = root;
    rec.p               = r.at(rec.t);
    vec3 outward_normal = (rec.p - center) / radius;
    rec.set_face_normal(r, outward_normal);
    get_sphere_uv(outward_normal, rec.u, rec.v);
    rec.mat_ptr = mat_ptr.get();

    return true;
}

#endif
