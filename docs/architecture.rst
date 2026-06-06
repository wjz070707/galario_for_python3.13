Architecture and maintenance
============================

This page is a map for maintainers. It describes where code belongs and which
boundaries should remain stable when the implementation changes.

Repository boundaries
---------------------

The public source distribution contains ``src``, ``python``, ``tests``,
``examples``, ``benchmarks``, and ``docs``. Local Measurement Sets, exported
science data, MCMC chains, and run outputs do not belong in the source
repository. This workspace keeps those under ignored ``data`` and
``galario_fit`` directories.

Runtime layers
--------------

The Python package is organized as a short stack::

    galario Python API
        -> nanobind extension
        -> public C++ API
        -> CPU or CUDA backend
        -> FFTW or CUDA/cuFFT/cuBLAS

``src/galario_api.cpp`` contains backend-neutral orchestration and pointer-erased
bridges used by the binding. ``src/galario_common.cpp`` contains shared model
rasterization and backend-selection heuristics. Numerical backend code belongs
in ``src/galario_cpu.cpp`` or ``src/galario_gpu.cu``.

Both backend libraries implement the private interface declared in
``src/galario_internal.h``. Keeping this seam small prevents CUDA types from
leaking into the CPU library and prevents backend conditionals from spreading
through public API code.

Python package
--------------

``python/bindings_nanobind.cpp`` owns the C++/Python boundary and the Python
lifetime wrapper for ``Chi2ImageContext``. ``python/api_builders.py`` is the
Python policy layer: it validates and normalizes NumPy arrays, provides the
snake_case API, and preserves selected legacy camelCase functions.

Do not put expensive numerical loops in the Python policy layer. Add them to
the C++ API and expose them through nanobind instead.

Python tests and their numerical reference helpers live in ``tests/python``.
Benchmarks and executable tutorials live in ``benchmarks`` and ``examples``;
none of those directories is part of the installed ``galario`` package.

Contexts and repeated evaluation
--------------------------------

Context reuse is the defining performance optimization of the maintained 1.3
line, not merely a convenience wrapper. Public examples and performance
claims should use the Context path unless they explicitly measure one-shot
sampling.

``Chi2ImageContext`` is intended for optimizers and MCMC workloads where
``u``, ``v``, observations, and weights remain fixed while model parameters
change. A context owns reusable buffers and backend plans. Context objects are
mutable workspaces and must not be used concurrently from multiple threads.

The CPU context retains aligned FFTW buffers and a plan. The plan is rebuilt if
the configured OpenMP thread count changes. The CUDA context additionally owns
device buffers, batch workspaces, and reusable cuFFT batch plans.

Backend selection
-----------------

``BACKEND_AUTO`` uses empirical cost heuristics in
``src/galario_common.cpp``. Changes to those thresholds must be supported by
benchmarks across representative image sizes, visibility counts, and batch
sizes. Explicit ``fft``, ``dft``, and ``nufft`` requests bypass the heuristic.

Why the CUDA file is larger
---------------------------

The CPU implementation delegates memory management and transforms largely to
the operating system, standard library, OpenMP, and FFTW. CUDA must explicitly
manage host/device memory, transfers, kernel launch geometry, synchronization,
error handling, cuFFT plans, and batch chunking. It also contains separate
kernels for operations that are ordinary loops on the CPU. The extra code is
mostly execution machinery rather than additional public functionality.

Maintenance checklist
---------------------

When changing a public numerical operation:

* Keep CPU and CUDA behavior and error handling aligned.
* Update the nanobind wrapper only when the C++ signature changes.
* Add or update Python tests for the public behavior.
* Build both CPU-only and CUDA configurations.
* Run the CPU suite and the CUDA nanobind suite.
* Benchmark before changing ``BACKEND_AUTO`` thresholds or workspace policy.

Lightweight benchmark
---------------------

``benchmarks/quickstart_benchmark.py`` uses the real uv distribution from
``docs/uvtable.txt`` and a small Gaussian model image. Its defaults are intended
for quick local checks rather than publication-quality performance claims::

    PYTHONPATH=build/python python benchmarks/quickstart_benchmark.py

Set ``GALARIO_BENCHMARK_GPU=1`` when running
``benchmarks/run_quickstart.sh`` to
also benchmark a CUDA build. Increase ``--points``, ``--size``, and
``--repeats`` only for deliberate performance studies. ``--loops`` controls
how many calls are grouped into each timing sample to reduce sub-millisecond
timer noise.
