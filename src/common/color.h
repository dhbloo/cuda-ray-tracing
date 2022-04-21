#ifndef COLOR_H
#define COLOR_H
//==============================================================================================
// Originally written in 2020 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include "vec3.h"

#include <iostream>

inline color radiance_to_color(color radiance, int samples_per_pixel)
{
    auto r = radiance.x();
    auto g = radiance.y();
    auto b = radiance.z();

    // Replace NaN components with zero. See explanation in Ray Tracing: The Rest of Your Life.
    if (r != r)
        r = 0.0;
    if (g != g)
        g = 0.0;
    if (b != b)
        b = 0.0;

    // Divide the color by the number of samples and gamma-correct for gamma=2.0.
    auto scale = 1.0f / samples_per_pixel;
    r          = sqrt(scale * r);
    g          = sqrt(scale * g);
    b          = sqrt(scale * b);

    return color(r, g, b);
}

inline void write_color(std::ostream &out, color pixel_color)
{
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    // Write the translated [0,255] value of each color component.
    out << static_cast<int>(256 * clamp(r, 0.0f, 0.999f)) << ' '
        << static_cast<int>(256 * clamp(g, 0.0f, 0.999f)) << ' '
        << static_cast<int>(256 * clamp(b, 0.0f, 0.999f)) << '\n';
}

inline int color_to_rgb_integer(color pixel_color)
{
    auto r = pixel_color.x();
    auto g = pixel_color.y();
    auto b = pixel_color.z();

    int rb = (int)clamp(r * 255, 0.0f, 255.0f);
    int gb = (int)clamp(g * 255, 0.0f, 255.0f);
    int bb = (int)clamp(b * 255, 0.0f, 255.0f);

    return bb | (gb << 8) | (rb << 16);
}

#endif
