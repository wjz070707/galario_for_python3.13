###############################################################################
# This file is part of GALARIO:                                               #
# Gpu Accelerated Library for Analysing Radio Interferometer Observations     #
#                                                                             #
# Copyright (C) 2017-2020, Marco Tazzari, Frederik Beaujean, Leonardo Testi.  #
#                                                                             #
# This program is free software: you can redistribute it and/or modify        #
# it under the terms of the Lesser GNU General Public License as published by #
# the Free Software Foundation, either version 3 of the License, or           #
# (at your option) any later version.                                         #
#                                                                             #
# This program is distributed in the hope that it will be useful,             #
# but WITHOUT ANY WARRANTY; without even the implied warranty of              #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                        #
#                                                                             #
# For more details see the LICENSE file.                                      #
# For documentation see https://mtazzari.github.io/galario/                   #
###############################################################################

cmake_minimum_required(VERSION 3.15)
project(galario VERSION 1.3.0)

# Keep third-party CMake helper noise under control on modern CMake versions.
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "Suppress developer warnings from bundled CMake modules" FORCE)
set(CMAKE_WARN_DEPRECATED FALSE CACHE BOOL "Suppress deprecated CMake warnings from bundled modules" FORCE)
set(PACKAGE_VERSION "'${PROJECT_VERSION}'")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif(NOT CMAKE_BUILD_TYPE)

# if inside a conda env, install everything in there
if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND DEFINED ENV{CONDA_PREFIX} AND NOT DEFINED SKBUILD_PROJECT_NAME)
  # https://stackoverflow.com/questions/21437514/
  set(CMAKE_INSTALL_PREFIX "$ENV{CONDA_PREFIX}" CACHE PATH "Set the prefix based on conda environment" FORCE)
  message(STATUS "Installing inside conda env at ${CMAKE_INSTALL_PREFIX}")
endif()

# ship extra findXXX.cmake in `cmake/`
# APPEND doesn't propagate variable to subdirectory
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

# many more useful cmake module and scripts
# are downloaded into the build directory
include(LookUp-GreatCMakeCookOff)

###
# the optional packages
###
if(NOT DEFINED GALARIO_CHECK_CUDA)
  if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    message(STATUS "Ignoring cuda on the mac by default. Force searching for cuda with `cmake -DGALARIO_CHECK_CUDA=1`")
    set(GALARIO_CHECK_CUDA 0)
  else()
    set(GALARIO_CHECK_CUDA 1)
  endif()
endif()

if(GALARIO_CHECK_CUDA AND UNIX AND NOT APPLE)
  set(_galario_cuda_roots
    "$ENV{CUDA_HOME}"
    "$ENV{CUDA_PATH}"
    "/usr/local/cuda"
    "/usr/local/cuda-12.6"
    "/usr/local/cuda-12"
  )
  foreach(_cuda_root IN LISTS _galario_cuda_roots)
    if(NOT _cuda_root)
      continue()
    endif()
    if(EXISTS "${_cuda_root}/bin/nvcc")
      if(NOT DEFINED CMAKE_CUDA_COMPILER OR CMAKE_CUDA_COMPILER STREQUAL "NOTFOUND")
        set(CMAKE_CUDA_COMPILER "${_cuda_root}/bin/nvcc" CACHE FILEPATH "CUDA compiler" FORCE)
      endif()
      if(NOT DEFINED CUDAToolkit_ROOT OR NOT CUDAToolkit_ROOT)
        set(CUDAToolkit_ROOT "${_cuda_root}" CACHE PATH "CUDA toolkit root" FORCE)
      endif()
      break()
    endif()
  endforeach()
endif()

set(CUDA_FOUND FALSE)
if(GALARIO_CHECK_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CUDA_FOUND TRUE)
  else()
    message(STATUS "CUDA compiler not found, building CPU-only targets")
  endif()
endif()

# has to come before subdirectories are added
enable_testing()

# we can't use static libs with python
set(BUILD_SHARED_LIBS ON)
add_subdirectory(src)

###
# python not required but optional
###
set(BUILD_WRAPPER FALSE)
find_package(Python COMPONENTS Interpreter Development.Module QUIET)
if(Python_FOUND)
  set(BUILD_WRAPPER TRUE)
else()
  message(STATUS "Skipping the ${PROJECT_NAME} python wrapper because Python development headers were not found")
endif()

if(BUILD_WRAPPER)
  message(STATUS "Building the ${PROJECT_NAME} python wrapper")
  add_subdirectory(python)
else()
  message(STATUS "Skipping the ${PROJECT_NAME} python wrapper")
endif()

find_package(Sphinx)
if(SPHINX_FOUND)
  add_subdirectory(docs)
else()
  message(STATUS "Cannot build the documentation without sphinx")
endif()

###
# uninstall target
###
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
