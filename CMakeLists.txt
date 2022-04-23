cmake_minimum_required(VERSION 3.12.0)

project(CudaRayTracing)
set(CMAKE_CXX_STANDARD 14)

# Source

set(COMMON_ALL
	src/common/rtweekend.h
	src/common/camera.h
	src/common/ray.h
	src/common/vec3.h
	src/common/aabb.h
	src/common/texture.h
	src/common/perlin.h
	src/common/onb.h
	src/common/rtw_stb_image.cpp
)

set(SOURCE_CPU
	${COMMON_ALL}
	src/cpu/aarect.h
	src/cpu/box.h
	src/cpu/bvh.h
	src/cpu/hittable.h
	src/cpu/hittable_list.h
	src/cpu/material.h
	src/cpu/pdf.h
	src/cpu/sphere.h
	src/cpu/main.cc
	src/common/external/window.h
	src/common/external/window.cpp
)

include_directories(src/common)

add_executable(ray-tracing-cpu ${SOURCE_CPU})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	add_executable(ray-tracing-cpu-openmp ${SOURCE_CPU})
    target_link_libraries(ray-tracing-cpu-openmp PUBLIC OpenMP::OpenMP_CXX)
	target_compile_definitions(ray-tracing-cpu-openmp PUBLIC WITH_OPENMP)
endif()