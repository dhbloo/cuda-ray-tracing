#ifndef RAY_H
#define RAY_H
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

#include "vec3.h"

class ray
{
public:
    __dual__ ray() {}
    __dual__ ray(const point3 &origin, const vec3 &direction)
        : orig(origin)
        , dir(direction)
    {}

    __dual__ point3 origin() const { return orig; }
    __dual__ vec3   direction() const { return dir; }

    __dual__ point3 at(float t) const { return orig + t * dir; }

public:
    point3 orig;
    vec3   dir;
};

#endif
