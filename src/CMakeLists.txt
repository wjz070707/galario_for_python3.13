#####################################################################################
# This file is part of GALARIO.                                                     #
#                                                                                   #
# GALARIO - Gpu Accelerated Library for Analysing Radio Interferometer Observations #
# Copyright (C) 2017-2020, Marco Tazzari, Frederik Beaujean, Leonardo Testi.        #
#                                                                                   #
# This program is free software: you can redistribute it and/or modify              #
# it under the terms of the Lesser GNU General Public License as published by       #
# the Free Software Foundation, either version 3 of the License, or                 #
# (at your option) any later version.                                               #
#                                                                                   #
# This program is distributed in the hope that it will be useful,                   #
# but WITHOUT ANY WARRANTY; without even the implied warranty of                    #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                              #
#                                                                                   #
# For more details see the LICENSE file.                                            #
# For documentation see https://mtazzari.github.io/galario/                         #
#####################################################################################

# C++ 11 support http://stackoverflow.com/questions/37621342/ for cmake >= 2.8
if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
  set(min_gcc 4.8.1)
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS ${min_gcc})
    message(FATAL_ERROR "${PROJECT_NAME} requires g++ >= ${min_gcc} for C++11 support.")
  endif()
elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
  set(min_clang 3.3)
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS ${min_clang})
    message(FATAL_ERROR "${PROJECT_NAME} requires clang++ >= ${min_clang} for C++11 support.")
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
  endif()
else()
  message(FATAL_ERROR "Your compiler is not supported for lack of complete C++11 support")
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Keep host compiler warnings/aliasing flags on the C++ compilation path only.
add_compile_options(
  $<$<COMPILE_LANGUAGE:CXX>:-Wall>
  $<$<COMPILE_LANGUAGE:CXX>:-pedantic>
  $<$<COMPILE_LANGUAGE:CXX>:-fstrict-aliasing>
  $<$<COMPILE_LANGUAGE:CXX>:-pthread>
)

# explicit CPU and CUDA translation units keep the build layout honest and
# avoid the old "copy .cpp to .cu" workaround.
set(common_sources
  "${CMAKE_CURRENT_SOURCE_DIR}/galario_common.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/galario_api.cpp"
)
set(cpu_sources
  ${common_sources}
  "${CMAKE_CURRENT_SOURCE_DIR}/galario_cpu.cpp"
)
set(cuda_sources
  ${common_sources}
  "${CMAKE_CURRENT_SOURCE_DIR}/galario_gpu.cu"
)

add_library(galario ${cpu_sources})

set(install_libs galario)

find_package(OpenMP)
find_package(Threads)
if(OpenMP_FOUND AND Threads_FOUND)
  set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
  set(CMAKE_SHARED_LINKER_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_SHARED_LINKER_FLAGS}")
  set(FFTW_PARALLEL "THREADS")

  # macOS/AppleClang needs explicit libomp; Linux/GCC typically does not.
  if(APPLE)
    find_library(OMP_LIB omp HINTS "$ENV{CONDA_PREFIX}/lib" "${CMAKE_PREFIX_PATH}/lib")
    if(OMP_LIB)
      target_link_libraries(galario ${OMP_LIB})
    else()
      message(FATAL_ERROR "OpenMP found but libomp (omp) not found. Install llvm-openmp in conda.")
    endif()
  endif()
endif()

# GALARIO is double precision only. The local finder exposes just the serial
# and optional pthread FFTW libraries used by the CPU backend.
find_package(FFTW3 REQUIRED COMPONENTS DOUBLE ${FFTW_PARALLEL})

OPTION(GALARIO_TIMING "Output timing of selected functions. For testing only. Default: OFF" OFF)

foreach(t IN ITEMS galario)
  target_link_libraries(${t} ${FFTW3_LIBRARIES})
  target_include_directories(${t} PUBLIC ${FFTW3_INCLUDE_DIRS} ${FFTW3_INCLUDE_DIR_PARALLEL})
  if(GALARIO_TIMING)
    target_compile_definitions(${t} PRIVATE GALARIO_TIMING=1)
  endif()
endforeach()

if (CUDA_FOUND)
  add_library(galario_cuda ${cuda_sources})

  foreach(t IN ITEMS galario_cuda)
    target_include_directories(${t} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(${t} PUBLIC CUDA::cudart CUDA::cublas CUDA::cufft)
    target_compile_options(${t} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:--default-stream=per-thread>
    )
    target_compile_definitions(${t} PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:_FORCE_INLINES>
    )
    if(GALARIO_TIMING)
      target_compile_definitions(${t} PRIVATE GALARIO_TIMING=1)
    endif()
    set_target_properties(${t} PROPERTIES
      CUDA_ARCHITECTURES 80
      CUDA_STANDARD 11
      CUDA_STANDARD_REQUIRED ON
      POSITION_INDEPENDENT_CODE ON
    )
    list(APPEND install_libs ${t})
  endforeach()
endif() # cuda

###
# install
###
# it would be nice if directory would let me access all libraries but there is no such property
# https://cmake.org/cmake/help/v3.0/manual/cmake-properties.7.html
install (TARGETS ${install_libs}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  )
install(FILES galario.h galario_py.h galario_defs.h DESTINATION include)

###
# testing
###
add_executable(galario_test_cpp galario_test.cpp)
add_test(cpp_compile_test galario_test_cpp)

foreach(t IN ITEMS galario_test_cpp)
  target_link_libraries(${t} galario ${FFTW3_LIBRARIES})
endforeach()
