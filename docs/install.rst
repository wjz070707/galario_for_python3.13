Build and installation
======================

Supported environment
---------------------

The maintained target is Linux x86-64 with Python 3.10 or newer. CPU builds
require FFTW double-precision libraries. CUDA builds require an NVIDIA CUDA
toolkit and compatible driver.

Windows users should build inside WSL2. Native Windows is not currently
supported. macOS CPU builds may work with Clang, FFTW, and OpenMP but are not
part of the current validation matrix.

Python environment
------------------

A minimal conda environment is:

.. code-block:: bash

    conda create -n galario13 python=3.13 numpy scipy pytest \
        cmake ninja nanobind scikit-build-core build
    conda activate galario13

The project is built from source. Historical conda-forge releases do not
represent the 1.3 API or CUDA build described in this documentation.

CPU build
---------

From the repository root:

.. code-block:: bash

    cmake -S . -B build \
        -DGALARIO_CHECK_CUDA=0 \
        -DCMAKE_BUILD_TYPE=Release \
        -DPython_EXECUTABLE="$(which python)" \
        -DPYTHON_EXECUTABLE="$(which python)"
    cmake --build build -j

Use the build-tree Python package without installing:

.. code-block:: bash

    PYTHONPATH=build/python python -c "import galario; print(galario.__version__)"

CUDA build
----------

Set ``CUDA_HOME`` when the toolkit is not available under a standard
``/usr/local/cuda`` path:

.. code-block:: bash

    export CUDA_HOME=/usr/local/cuda
    cmake -S . -B build_gpu \
        -DGALARIO_CHECK_CUDA=1 \
        -DCMAKE_BUILD_TYPE=Release \
        -DPython_EXECUTABLE="$(which python)" \
        -DPYTHON_EXECUTABLE="$(which python)"
    cmake --build build_gpu -j

Verify the CUDA package:

.. code-block:: bash

    PYTHONPATH=build_gpu/python python -c \
        "import galario; print(galario.HAVE_CUDA)"

Python wheel
------------

``pyproject.toml`` uses scikit-build-core and nanobind:

.. code-block:: bash

    python -m pip install build
    python -m build --wheel

Wheel portability depends on the linked FFTW and CUDA libraries. Treat locally
built CUDA wheels as machine/toolkit-specific unless they are produced in a
controlled wheel-building environment.

Dependencies
------------

Required for the Python package:

* Python 3.10+
* NumPy 2.0+
* CMake 3.15+
* scikit-build-core
* nanobind
* a GCC or Clang C++ compiler
* FFTW double and, when OpenMP is enabled, FFTW threads

Optional:

* CUDA Toolkit, cuFFT, and cuBLAS
* SciPy and pytest for tests
* Sphinx for documentation
* emcee, corner, and matplotlib for the tutorial

Testing
-------

CPU:

.. code-block:: bash

    PYTHONPATH=build/python python -m pytest -o addopts="" \
        build/python/test_galario.py

CUDA smoke tests:

.. code-block:: bash

    PYTHONPATH=build_gpu/python GALARIO_TEST_GPU=1 \
        python -m pytest -o addopts="" build_gpu/python/test_nanobind.py

Common issues
-------------

Wrong Python environment
~~~~~~~~~~~~~~~~~~~~~~~~

Use an empty build directory when changing conda environments. Configure both
``Python_EXECUTABLE`` and, for older helper modules,
``PYTHON_EXECUTABLE`` when CMake finds the wrong interpreter.

CUDA not detected
~~~~~~~~~~~~~~~~~

Check ``nvcc --version``, ``CUDA_HOME``, and ``nvidia-smi``. Pass
``-DGALARIO_CHECK_CUDA=0`` when only the CPU build is required.

Missing FFTW
~~~~~~~~~~~~

Install the double-precision FFTW and FFTW threads development libraries.
Single-precision FFTW is not required because GALARIO 1.3 is double-only.
