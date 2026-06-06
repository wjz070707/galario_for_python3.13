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

set(PYGALARIO_DIR "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}")
file(REMOVE_RECURSE "${PYGALARIO_DIR}")
file(MAKE_DIRECTORY "${PYGALARIO_DIR}")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/__init__.py.in"
  "${PYGALARIO_DIR}/__init__.py"
  )
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/nanobind_api_v2.py"
  "${PYGALARIO_DIR}/nanobind_api_v2.py"
  COPYONLY
  )
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/api_constants.py"
  "${PYGALARIO_DIR}/api_constants.py"
  COPYONLY
  )
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/api_utils.py"
  "${PYGALARIO_DIR}/api_utils.py"
  COPYONLY
  )
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/api_builders.py"
  "${PYGALARIO_DIR}/api_builders.py"
  COPYONLY
  )
file(MAKE_DIRECTORY "${PYGALARIO_DIR}/double")
set(GALARIO_NANOBIND_PACKAGE "nanobind_double")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/__init_module__.py.in"
  "${PYGALARIO_DIR}/double/__init__.py"
  @ONLY
  )
file(MAKE_DIRECTORY "${PYGALARIO_DIR}/nanobind_double")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/nanobind_double/__init__.py"
  "${PYGALARIO_DIR}/nanobind_double/__init__.py"
  COPYONLY
  )
if (CUDA_FOUND)
  file(MAKE_DIRECTORY "${PYGALARIO_DIR}/double_cuda")
  set(GALARIO_NANOBIND_PACKAGE "nanobind_double_cuda")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/__init_module__.py.in"
    "${PYGALARIO_DIR}/double_cuda/__init__.py"
    @ONLY
    )
  file(MAKE_DIRECTORY "${PYGALARIO_DIR}/nanobind_double_cuda")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/nanobind_double_cuda/__init__.py"
    "${PYGALARIO_DIR}/nanobind_double_cuda/__init__.py"
    COPYONLY
    )
endif()

set(NANOBIND_FOUND FALSE)
set(_nanobind_python "${PYTHON_EXECUTABLE}")
if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}/bin/python")
  set(_nanobind_python "$ENV{CONDA_PREFIX}/bin/python")
  set(Python_EXECUTABLE "${_nanobind_python}" CACHE FILEPATH "Python executable for nanobind bindings" FORCE)
endif()
if(_nanobind_python)
  execute_process(
    COMMAND ${_nanobind_python} -m nanobind --cmake_dir
    RESULT_VARIABLE _nanobind_status
    OUTPUT_VARIABLE NANOBIND_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(_nanobind_status EQUAL 0 AND EXISTS "${NANOBIND_CMAKE_DIR}")
    list(APPEND CMAKE_PREFIX_PATH "${NANOBIND_CMAKE_DIR}")
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(nanobind CONFIG QUIET)
  endif()
endif()

if(nanobind_FOUND)
  message(STATUS "Building experimental nanobind wrapper")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PYGALARIO_DIR}")
  nanobind_add_module(_nanobind_double bindings_nanobind.cpp)
  target_include_directories(_nanobind_double PUBLIC "${CMAKE_SOURCE_DIR}/src")
  target_compile_definitions(_nanobind_double PUBLIC GALARIO_NB_MODULE_NAME=_nanobind_double GALARIO_NB_WITH_CONTEXT=1 GALARIO_NB_CONTEXT_TAG=0)
  target_link_libraries(_nanobind_double PRIVATE galario)
  set_target_properties(_nanobind_double PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PYGALARIO_DIR}"
    OUTPUT_NAME "_nanobind_double"
  )
endif()

if(nanobind_FOUND AND CUDA_FOUND)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PYGALARIO_DIR}")
  nanobind_add_module(_nanobind_double_cuda bindings_nanobind.cpp)
  target_include_directories(_nanobind_double_cuda PUBLIC "${CMAKE_SOURCE_DIR}/src")
  target_compile_definitions(_nanobind_double_cuda PUBLIC GALARIO_NB_MODULE_NAME=_nanobind_double_cuda GALARIO_NB_WITH_CONTEXT=1 GALARIO_NB_CONTEXT_TAG=1)
  target_link_libraries(_nanobind_double_cuda PRIVATE galario_cuda)
  set_target_properties(_nanobind_double_cuda PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${PYGALARIO_DIR}"
    OUTPUT_NAME "_nanobind_double_cuda"
  )
endif()

###
# installation
###
# scikit-build-core installs directly into the wheel platlib root, so keep the
# generated `galario/` package directory relative in that case. For regular
# CMake installs, continue to resolve the active interpreter site-packages path.
if(DEFINED SKBUILD_PROJECT_NAME)
  set(GALARIO_PYTHON_PKG_DIR "." CACHE PATH "Python install path for wheel builds" FORCE)
else()
  # to set ${PYTHON_PKG_DIR}. It is prepended by ${CMAKE_INSTALL_PREFIX} if set
  include(PythonInstall)

  # Python 3.12+ removed distutils, so keep a sysconfig fallback even if the
  # helper script fails to populate PYTHON_PKG_DIR.
  if(NOT PYTHON_PKG_DIR)
    execute_process(
      COMMAND ${PYTHON_EXECUTABLE} -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
      OUTPUT_VARIABLE PYTHON_PKG_DIR
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  endif()
  if(NOT PYTHON_PKG_DIR)
    message(FATAL_ERROR "Unable to determine PYTHON_PKG_DIR for installation")
  endif()

  # I guess variable has to be in cache to survive reinvocations of cmake that happen during building
  set(GALARIO_PYTHON_PKG_DIR "${PYTHON_PKG_DIR}" CACHE PATH "Current python install path" FORCE)
endif()

# Refuse stale regular CMake installs into the wrong conda environment.
# Wheel builds intentionally stage files under a temporary scikit-build path.
if(NOT DEFINED SKBUILD_PROJECT_NAME)
  install(CODE "
    if(DEFINED ENV{CONDA_PREFIX} AND NOT \"$ENV{CONDA_PREFIX}\" STREQUAL \"\")
      get_filename_component(_active_conda_prefix \"$ENV{CONDA_PREFIX}\" REALPATH)
      get_filename_component(_configured_python \"${Python_EXECUTABLE}\" REALPATH)
      set(_expected_python \"\${_active_conda_prefix}/bin/python\")
      get_filename_component(_expected_python \"\${_expected_python}\" REALPATH)

      if(NOT _configured_python STREQUAL _expected_python)
        message(FATAL_ERROR
          \"GALARIO install aborted: build directory is configured for Python '${Python_EXECUTABLE}', \"
          \"but the active conda environment is '\${_active_conda_prefix}'. \"
          \"Please remove build/ and re-run cmake from the desired environment.\")
      endif()

      if(NOT \"${CMAKE_INSTALL_PREFIX}\" MATCHES \"^\${_active_conda_prefix}(/|$)\")
        message(FATAL_ERROR
          \"GALARIO install aborted: CMAKE_INSTALL_PREFIX='${CMAKE_INSTALL_PREFIX}' does not belong to \"
          \"the active conda environment '\${_active_conda_prefix}'. \"
          \"Please remove build/ and re-run cmake from the desired environment.\")
      endif()
    endif()
  ")
endif()

# on unix: install .py and .so relative to ${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT}
install(DIRECTORY ${PYGALARIO_DIR}
  DESTINATION "${GALARIO_PYTHON_PKG_DIR}"
  FILES_MATCHING REGEX ".*py$|.*so$"
  )

###
# tests
###
set(GALARIO_PYTHON_TEST_DIR "${CMAKE_SOURCE_DIR}/tests/python")
configure_file(
  "${GALARIO_PYTHON_TEST_DIR}/utils.py"
  "${CMAKE_CURRENT_BINARY_DIR}/utils.py"
  COPYONLY
  )
configure_file(
  "${GALARIO_PYTHON_TEST_DIR}/conftest.py"
  "${CMAKE_CURRENT_BINARY_DIR}/conftest.py"
  COPYONLY
  )

# TODO avoid installation of test scripts
# http://stackoverflow.com/questions/13381742/cmake-exclude-executable-during-install
set(galario_test test_galario.py)
set(galario_nanobind_test test_nanobind.py)

# use configure_file so it's copied on every change of the source
# http://stackoverflow.com/a/34800667/987623
configure_file(
  "${GALARIO_PYTHON_TEST_DIR}/${galario_test}"
  "${CMAKE_CURRENT_BINARY_DIR}/${galario_test}"
  COPYONLY
  )
configure_file(
  "${GALARIO_PYTHON_TEST_DIR}/${galario_nanobind_test}"
  "${CMAKE_CURRENT_BINARY_DIR}/${galario_nanobind_test}"
  COPYONLY
  )
add_test(NAME CanImport COMMAND ${PYTHON_EXECUTABLE} -c "import ${PROJECT_NAME}")

# https://github.com/UCL/GreatCMakeCookOff/wiki/Adding-py.test-to-ctest
include(AddPyTest)

# I don't understand the first arg but it seems to work if py.test already installed
setup_pytest("${CMAKE_CURRENT_BINARY_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/py.test.sh")

# the source name has to match test_*.py
# https://github.com/UCL/GreatCMakeCookOff/issues/56
#
# need to run in binary_dir so import of pygalario works
add_pytest("${CMAKE_CURRENT_BINARY_DIR}/${galario_test}" NOINSTALL CMDLINE --cov=./)
