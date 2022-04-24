cmake_minimum_required(VERSION 3.17.0 FATAL_ERROR)

project(CudaRayTracing LANGUAGES CUDA CXX)
set(CMAKE_CXX_STANDARD 14)

# Source

set(COMMON_ALL
	src/common/rtweekend.h
	src/common/camera.h
	src/common/ray.h
	src/common/vec3.h
	src/common/aabb.h
	src/common/texture.h
	src/common/onb.h
    src/common/aarect.h
	src/common/box.h
	src/common/bvh.h
	src/common/hittable.h
	src/common/hittable_list.h
	src/common/material.h
	src/common/pdf.h
	src/common/sphere.h
	src/common/rtw_stb_image.cpp
	src/common/external/window.h
	src/common/external/window.cpp
)

set(SOURCE_CPU
	${COMMON_ALL}
	
	src/cpu/main.cc
)

set(SOURCE_GPU
	${COMMON_ALL}
	src/gpu/main.cu
)

include_directories(src/common)

add_executable(ray-tracing-cpu ${SOURCE_CPU})

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
	add_executable(ray-tracing-cpu-openmp ${SOURCE_CPU})
    target_link_libraries(ray-tracing-cpu-openmp PUBLIC OpenMP::OpenMP_CXX)
	target_compile_definitions(ray-tracing-cpu-openmp PUBLIC WITH_OPENMP)
endif()

find_package(CUDAToolkit)
add_executable(ray-tracing-gpu ${SOURCE_GPU})
target_link_libraries(ray-tracing-gpu PRIVATE CUDA::cudart)
set_target_properties(ray-tracing-gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
