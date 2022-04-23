#ifndef RTWEEKEND_H
#define RTWEEKEND_H
//==============================================================================================
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include <cmath>
#include <cstdlib>
#include <limits>
#include <memory>

#ifdef __NVCC__

    #define __dual__ __host__ __device__

__device__ float random_float();

#else

    #define __host__
    #define __device__
    #define __dual__

inline float random_float()
{
    // Returns a random real in [0,1).
    return rand() / (RAND_MAX + 1.0f);
}

using std::fabs;
using std::sqrt;

#endif

// Usings

using std::make_shared;
using std::shared_ptr;

// Constants

constexpr float infinity = std::numeric_limits<float>::infinity();
constexpr float pi       = 3.1415926535897932385f;

// Utility Functions

__dual__ constexpr float degrees_to_radians(float degrees)
{
    return degrees * pi / 180.0f;
}

__dual__ constexpr float clamp(float x, float min, float max)
{
    if (x < min)
        return min;
    if (x > max)
        return max;
    return x;
}

__device__ inline float random_float(float min, float max)
{
    // Returns a random real in [min,max).
    return min + (max - min) * random_float();
}

__device__ inline int random_int(int min, int max)
{
    // Returns a random integer in [min,max].
    return static_cast<int>(random_float(min, max + 1));
}

// Common Headers

#include "ray.h"
#include "vec3.h"

#endif
