cmake_minimum_required (VERSION 2.8)

# projectname is the same as the main-executable
project(kNN)

SET (CMAKE_C_COMPILER /usr/bin/gcc)
SET (CMAKE_C_FLAGS          "-std=c99 -march=x86-64 -m64")
SET (CMAKE_CXX_COMPILER /usr/bin/g++)
SET (CMAKE_CXX_FLAGS        "-std=c++1y -march=x86-64 -m64 -fno-omit-frame-pointer -fPIC -I/usr/local/include -Wall -Wextra -Woverloaded-virtual -Wno-unused-local-typedefs")
SET (CMAKE_INCLUDE_SYSTEM_FLAG_CXX "isystem")

set (CMAKE_C_FLAGS          "${CMAKE_C_FLAGS}" CACHE STRING "c flags")
set (CMAKE_CXX_FLAGS        "${CMAKE_CXX_FLAGS}" CACHE STRING "c++ flags")
SET (CMAKE_EXPORT_COMPILE_COMMANDS 1)
set (CMAKE_EXPORT_COMPILE_COMMANDS "${CMAKE_EXPORT_COMPILE_COMMANDS}" CACHE STRING "export compile_commands.json")

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)
set (CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}" CACHE STRING "build type")

SET (CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  external/catch/single_include
  external/cub
  external/moderngpu/src
  external/vexcl
)

find_package(CUDA REQUIRED)
include_directories( ${CUDA_INCLUDE_DIRS} )
set(BACKEND_LIBS ${CUDA_LIBRARIES})

#----------------------------------------------------------------------------
# Find OpenMP
#----------------------------------------------------------------------------
find_package(OpenMP)

if (OpenMP_CXX_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

#----------------------------------------------------------------------------
# Enable C++11 support, set compilation flags
#----------------------------------------------------------------------------
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wclobbered -Wempty-body -Wmissing-field-initializers -Wtype-limits -Wuninitialized -Wunused-but-set-parameter -Wno-comment -Wno-type-limits -Wno-strict-aliasing -Wno-missing-braces -Wno-deprecated-declarations -Wno-unused-function")
endif ()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wempty-body -Wmissing-field-initializers -Wtype-limits -Wuninitialized -Wno-comment -Wno-tautological-compare -Wno-missing-braces -Wno-deprecated-declarations -Wno-unused-function")

    option(USE_LIBCPP "Use libc++ with Clang" OFF)
    if (USE_LIBCPP)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
    endif ()
endif ()

list(APPEND CUDA_NVCC_FLAGS "-arch=sm_30;-std=c++11;-O3;--use_fast_math;--expt-extended-lambda")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)

cuda_add_library(knn_impl kNN/kNN_impl.cu kNN/kNN_impl.h)
cuda_add_library(distance distance.cu distance.h)
cuda_add_library(cub_knn cub_impl.cu cub_impl.h)
cuda_add_library(mgpu_knn mgpu_impl.cu mgpu_impl.h)
target_link_libraries(cub_knn
  mgpu_knn
  distance
  knn_impl
  glog
)

add_library(kNN kNN.cpp)

install(DIRECTORY kNN DESTINATION include)
install(TARGETS kNN DESTINATION lib)

set(TEST unit_tests)
file(GLOB TEST_FILES tests/unit_tests.cpp tests/*_test.cpp)
add_executable(${TEST} ${TEST_FILES})
target_link_libraries(${TEST}
  knn_impl
  kNN
  mgpu_knn
  cub_knn
  distance
  ${BACKEND_LIBS}
)
add_custom_target(check COMMAND unit_tests)
add_dependencies(check unit_tests)
