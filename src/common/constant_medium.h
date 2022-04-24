#ifndef CONSTANT_MEDIUM_H
#define CONSTANT_MEDIUM_H
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
#include "material.h"
#include "texture.h"

class constant_medium : public hittable
{
public:
    __device__ constant_medium(shared_ptr<hittable> b, float d, shared_ptr<tex> a)
        : boundary(b)
        , density(d)
    {
        phase_function = make_shared<isotropic>(a);
    }
    __device__ virtual bool hit(const ray &r, float t_min, float t_max, hit_record &rec) const;
    __device__ virtual bool bounding_box(float t0, float t1, aabb &box) const
    {
        return boundary->bounding_box(t0, t1, box);
    }

public:
    shared_ptr<hittable> boundary;
    float                density;
    shared_ptr<material> phase_function;
};

__device__ bool constant_medium::hit(const ray &r, float t_min, float t_max, hit_record &rec) const
{
    hit_record rec1, rec2;
    if (boundary->hit(r, -infinity, infinity, rec1)) {
        if (boundary->hit(r, rec1.t + 0.0001, infinity, rec2)) {
            if (rec1.t < t_min)
                rec1.t = t_min;
            if (rec2.t > t_max)
                rec2.t = t_max;
            if (rec1.t >= rec2.t)
                return false;
            if (rec1.t < 0)
                rec1.t = 0;
            float distance_inside_boundary = (rec2.t - rec1.t) * r.direction().length();
            float hit_distance             = -(1 / density) * log(random_float());
            if (hit_distance < distance_inside_boundary) {
                rec.t       = rec1.t + hit_distance / r.direction().length();
                rec.p       = r.at(rec.t);
                rec.normal  = vec3(1, 0, 0);  // arbitrary
                rec.mat_ptr = phase_function.get();
                return true;
            }
        }
    }
    return false;
}

#endif
