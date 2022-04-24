#ifndef HITTABLE_LIST_H
#define HITTABLE_LIST_H
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
#include "rtweekend.h"

#include <memory>
#include <vector>

class hittable_list : public hittable
{
public:
    __host__ hittable_list() { clear(); };
    __host__ hittable_list(shared_ptr<hittable> object) { add(object); }

    __host__ void clear()
    {
        objects.clear();
        begin_ = end_ = objects.data();
    }
    __host__ void add(shared_ptr<hittable> object)
    {
        objects.push_back(object);
        begin_ = objects.data();
        end_   = objects.data() + objects.size();
    }
    __dual__ size_t size() const { return end_ - begin_; }

    __device__ virtual bool
    hit(const ray &r, float t_min, float t_max, hit_record &rec) const override;
    __dual__ virtual bool bounding_box(float time0, float time1, aabb &output_box) const override;
    __device__ virtual float pdf_value(const vec3 &o, const vec3 &v) const override;
    __device__ virtual vec3  random(const vec3 &o) const override;

public:
    std::vector<shared_ptr<hittable>> objects;
    shared_ptr<hittable>             *begin_;
    shared_ptr<hittable>             *end_;
};

__device__ bool hittable_list::hit(const ray &r, float t_min, float t_max, hit_record &rec) const
{
    hit_record temp_rec;
    auto       hit_anything   = false;
    auto       closest_so_far = t_max;

    for (auto object_ptr = begin_; object_ptr != end_; object_ptr++) {
        const auto &object = *object_ptr;
        if (object->hit(r, t_min, closest_so_far, temp_rec)) {
            hit_anything   = true;
            closest_so_far = temp_rec.t;
            rec            = temp_rec;
        }
    }

    return hit_anything;
}

__dual__ bool hittable_list::bounding_box(float time0, float time1, aabb &output_box) const
{
    if (size() == 0)
        return false;

    aabb temp_box;
    bool first_box = true;

    for (auto object_ptr = begin_; object_ptr != end_; object_ptr++) {
        const auto &object = *object_ptr;
        if (!object->bounding_box(time0, time1, temp_box))
            return false;
        output_box = first_box ? temp_box : surrounding_box(output_box, temp_box);
        first_box  = false;
    }

    return true;
}

__device__ float hittable_list::pdf_value(const point3 &o, const vec3 &v) const
{
    auto weight = 1.0f / size();
    auto sum    = 0.0f;

    for (auto object_ptr = begin_; object_ptr != end_; object_ptr++) {
        const auto &object = *object_ptr;
        sum += weight * object->pdf_value(o, v);
    }

    return sum;
}

__device__ vec3 hittable_list::random(const vec3 &o) const
{
    return begin_[random_int(0, static_cast<int>(size()) - 1)]->random(o);
}

#endif
