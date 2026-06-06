# GALARIO 1.3

GPU Accelerated Library for Analysing Radio Interferometer Observations.

GALARIO computes synthetic interferometric visibilities from model images or
axisymmetric radial profiles and evaluates their weighted chi-squared against
observations. Version 1.3 is double precision only and provides:

- **Reusable Contexts for high-throughput fitting.** Create one Context before
  an optimizer or MCMC run, then reuse it for every likelihood evaluation.
  This is the principal performance improvement in the maintained 1.3 line.
- A C++ CPU backend using FFTW and optional OpenMP.
- An optional NVIDIA CUDA backend using cuFFT and cuBLAS.
- Python 3.10+ bindings built with nanobind.
- FFT, direct Fourier transform, NUFFT-style oversampling, and automatic
  backend selection.
- Reusable image contexts for repeated optimizer and MCMC evaluations.

> **Performance rule:** never create a Context inside the likelihood function.
> Reuse it while the observations, weights, image dimensions, and backend
> settings remain fixed. This avoids repeated transfers, allocations, and
> transform-plan setup, especially on CUDA.

## GALARIO 1.3 optimizations

- Context-cached observations, CPU/GPU workspaces, and FFTW/cuFFT plans.
- Batched image-component and radial-profile likelihoods for vectorized emcee
  walkers, including GPU-side profile rasterization.
- Automatic or explicit FFT, direct DFT, and oversampled NUFFT-style backends.
- Reusable FFT batch plans and bounded CUDA workspace chunking.
- Double-precision-only core, removing unused single-precision build variants.
- Modern Python 3.10-3.13 and NumPy 2 bindings through nanobind.
- Self-contained CMake FFTW discovery without GreatCMakeCookOff.
- Split CPU, CUDA, common, and public API translation units for maintenance.

## Repository layout

```text
src/             C++ and CUDA numerical core
python/          installable Python API and nanobind binding
tests/python/    Python integration and numerical reference tests
examples/        directly executable tutorials
benchmarks/      lightweight performance tools
docs/            Sphinx documentation and the public uvtable example
```

Large local observing data and research-run outputs belong outside the source
distribution. This workspace ignores `data/` and `galario_fit/`.

## Build

CPU-only:

```bash
python -m pip install numpy nanobind scikit-build-core
cmake -S . -B build -DGALARIO_CHECK_CUDA=0
cmake --build build -j
```

CUDA:

```bash
cmake -S . -B build_gpu -DGALARIO_CHECK_CUDA=1
cmake --build build_gpu -j
```

The currently maintained and tested target is Linux x86-64. CUDA builds require
an NVIDIA toolkit and compatible driver.

## Context-first Python API

```python
import galario

ctx = galario.create_image_context(
    image.shape[0], image.shape[1],
    u, v, vis_obs.real, vis_obs.imag, weights,
)

for image in model_images:
    chi2 = galario.chi2_image(ctx=ctx, image=image, dxy=dxy)
```

For repeated likelihood evaluations, always reuse one context while `u`, `v`,
observed visibilities, and weights remain fixed. Otherwise every call may
repeat observation transfers, allocations, and transform setup. The context
is a mutable workspace and must not be used concurrently.

Use `from galario import double as g` to explicitly select CPU. CUDA builds also
provide `from galario import double_cuda as g`.

## Tutorial and benchmark

Open `examples/chi2_profile_gaussian.py`, edit the `USER CONFIGURATION` block,
and run it directly in VSCode with the Python environment where GALARIO is
installed. It uses `docs/uvtable.txt`, emcee 3, and writes a corner plot. The
maintained default uses all uv points, 24 walkers, and 1000 steps.
Set `USE_GPU` and `GPU_DEVICE` at the top of the file;
set `USE_GPU = False` to run the same example on CPU.

```bash
PYTHONPATH=build/python python benchmarks/quickstart_benchmark.py
```

## Testing

```bash
PYTHONPATH=build/python python -m pytest -o addopts="" build/python/test_galario.py
```

## Citation

If you use GALARIO in research, cite Tazzari, Beaujean and Testi (2018),
MNRAS 476, 4527, DOI: 10.1093/mnras/sty409.

For reproducibility, cite an immutable archived release rather than only the
moving branch. Use a Git tag, GitHub Release, and Zenodo DOI, and record the
commit, hardware, dependencies, backend, Context use, and benchmark settings.
See `docs/reproducibility.rst`.

The 1.3 line is maintained by
[`wjz070707`](https://github.com/wjz070707).

GALARIO is licensed under LGPLv3.
