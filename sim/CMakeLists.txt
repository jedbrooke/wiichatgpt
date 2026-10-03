cmake_minimum_required(VERSION 3.16)
project(wiichatgpt-sim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

option(WII_MATH_BACKEND "Math backend: CPU or GX" "CPU")
option(GX_SIM_TIMING "Enable simulated GPU latency (sleeps in DrawDone/CopyTex)" OFF)

include_directories(include)

# GX simulator library (only needed for GX backend and TEV tests)
add_library(gx_sim STATIC src/gx_sim.cpp)
if(GX_SIM_TIMING)
    target_compile_definitions(gx_sim PUBLIC GX_SIM_TIMING=1)
endif()

# TEV tests (always uses the GX simulator directly)
add_executable(test_tev tests/test_tev.cpp)
target_link_libraries(test_tev PRIVATE gx_sim)

# Math backend selection
if(WII_MATH_BACKEND STREQUAL "GX")
    add_library(wii_math STATIC src/math_gx.cpp)
    target_link_libraries(wii_math PRIVATE gx_sim)
    target_compile_definitions(wii_math PUBLIC WII_MATH_USE_GX=1)
else()
    add_library(wii_math STATIC src/math_cpu.cpp)
    target_compile_definitions(wii_math PUBLIC WII_MATH_USE_CPU=1)
endif()

# Matmul tests (uses whichever math backend is selected)
add_executable(test_matmul tests/test_matmul.cpp)
target_link_libraries(test_matmul PRIVATE wii_math)

# LLM inference executable
add_executable(llm_inference src/llm.cpp)
target_link_libraries(llm_inference PRIVATE wii_math)

# CTest integration
enable_testing()
add_test(NAME tev COMMAND test_tev)
add_test(NAME matmul COMMAND test_matmul)
