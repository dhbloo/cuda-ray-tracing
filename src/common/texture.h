#ifndef TEXTURE_H
#define TEXTURE_H
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

#include "external/stb_image.h"
#include "rtweekend.h"

#include <iostream>

class tex
{
public:
    __device__ virtual color value(float u, float v, const vec3 &p) const = 0;
};

class solid_color : public tex
{
public:
    __device__ solid_color(color c) : color_value(c) {}
    __device__ solid_color(float red, float green, float blue)
        : solid_color(color(red, green, blue))
    {}
    __device__ virtual color value(float u, float v, const vec3 &p) const override
    {
        return color_value;
    }

private:
    color color_value;
};

class checker_texture : public tex
{
public:
    __device__ checker_texture(shared_ptr<tex> _even, shared_ptr<tex> _odd) : even(_even), odd(_odd)
    {}
    __device__ checker_texture(color c1, color c2)
        : even(make_shared<solid_color>(c1))
        , odd(make_shared<solid_color>(c2))
    {}

    __device__ virtual color value(float u, float v, const vec3 &p) const override
    {
        auto sines = sin(10 * p.x()) * sin(10 * p.y()) * sin(10 * p.z());
        if (sines < 0)
            return odd->value(u, v, p);
        else
            return even->value(u, v, p);
    }

public:
    shared_ptr<tex> odd;
    shared_ptr<tex> even;
};

/*
class image_texture : public tex
{
public:
    const static int bytes_per_pixel = 3;

    __host__ image_texture(const char *filename)
    {
        auto components_per_pixel = bytes_per_pixel;

        unsigned char *buf =
            stbi_load(filename, &width, &height, &components_per_pixel, components_per_pixel);

        if (!buf) {
            std::cerr << "ERROR: Could not load texture image file '" << filename << "'.\n";
            width = height = 0;
        }

        bytes_per_scanline = bytes_per_pixel * width;
        int total_bytes    = bytes_per_scanline * height;

        data = new unsigned char[total_bytes];
        memcpy(data, buf, total_bytes);

        free(buf);
    }

    __host__ ~image_texture() { delete[] data; }

    __device__ virtual color value(float u, float v, const vec3 &p) const override
    {
        // If we have no texture data, then return solid cyan as a debugging aid.
        if (data == nullptr)
            return color(0, 1, 1);

        // Clamp input texture coordinates to [0,1] x [1,0]
        u = clamp(u, 0.0f, 1.0f);
        v = 1.0f - clamp(v, 0.0f, 1.0f);  // Flip V to image coordinates

        auto i = static_cast<int>(u * width);
        auto j = static_cast<int>(v * height);

        // Clamp integer mapping, since actual coordinates should be less than 1.0
        if (i >= width)
            i = width - 1;
        if (j >= height)
            j = height - 1;

        const auto color_scale = 1.0f / 255.0f;
        auto       pixel       = data + j * bytes_per_scanline + i * bytes_per_pixel;

        return color(color_scale * pixel[0], color_scale * pixel[1], color_scale * pixel[2]);
    }

private:
    unsigned char *data;
    int            width, height;
    int            bytes_per_scanline;
};
*/

#endif
