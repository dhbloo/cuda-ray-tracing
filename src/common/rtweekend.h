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
#include <type_traits>

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

template <typename T>
class shared_ptr
{
public:
    __device__ shared_ptr(T *ptr = nullptr) : p(ptr), ref_count(new size_t(1)) {}
    __device__ shared_ptr(const shared_ptr &sp) : p(sp.p), ref_count(sp.ref_count)
    {
        (*ref_count)++;
    }
    template <typename U,
              typename = typename std::enable_if_t<std::is_convertible<U *, T *>::value>>
    __device__ shared_ptr(const shared_ptr<U> &sp) : p(sp.p)
                                                   , ref_count(sp.ref_count)
    {
        (*ref_count)++;
    }
    __device__ ~shared_ptr()
    {
        if (ref_count)
            release();
    }

    __device__ shared_ptr &operator=(const shared_ptr &sp)
    {
        if (ref_count)
            release();

        p         = sp.p;
        ref_count = sp.ref_count;
        (*ref_count)++;
        return *this;
    }

    __device__ T *operator->() const noexcept { return p; }
    __device__ T &operator*() const noexcept { return *p; }
    __device__    operator bool() const noexcept { return p != nullptr; }
    __device__ T *get() const noexcept { return p; }

private:
    template <typename U>
    friend class shared_ptr;

    T      *p;
    size_t *ref_count;

    __device__ void release()
    {
        (*ref_count)--;
        if (*ref_count == 0) {
            if (p)
                delete p;
            delete ref_count;
            ref_count = nullptr;
        }
    }
};

template <class T, typename... Args>
__device__ shared_ptr<T> make_shared(Args &&...args)
{
    T *ptr = new T(::std::forward<Args>(args)...);
    return shared_ptr<T>(ptr);
}

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
    int x = (int)random_float(float(min), float(max + 1));
    return x < min ? min : x > max ? max : x;
}

// Common Headers

#include "ray.h"
#include "vec3.h"

#endif
