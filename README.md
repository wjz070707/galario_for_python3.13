# GALARIO 1.3

GPU Accelerated Library for Analysing Radio Interferometer Observations.

GALARIO computes synthetic interferometric visibilities from model images or
axisymmetric radial profiles and evaluates their weighted chi-squared against
observations. Version 1.3 is double precision only and provides:

- A C++ CPU backend using FFTW and optional OpenMP.
- An optional NVIDIA CUDA backend using cuFFT and cuBLAS.
- Python 3.10+ bindings built with nanobind.
- FFT, direct Fourier transform, NUFFT-style oversampling, and automatic
  backend selection.
- Reusable image contexts for repeated optimizer and MCMC evaluations.

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

## Python API

```python
import galario

vis = galario.sample_image(
    image=image,
    dxy=dxy,
    u=u,
    v=v,
    backend=galario.BACKEND_FFT,
)

ctx = galario.create_image_context(
    image.shape[0], image.shape[1],
    u, v, vis_obs.real, vis_obs.imag, weights,
)
chi2 = galario.chi2_image(ctx=ctx, image=image, dxy=dxy)
```

Use `from galario import double as g` to explicitly select CPU. CUDA builds also
provide `from galario import double_cuda as g`.

## Tutorial and benchmark

Open `examples/emcee_gaussian_profile.py`, edit the `USER CONFIGURATION` block,
and run it directly in VSCode with the Python environment where GALARIO is
installed. It uses `docs/uvtable.txt`, emcee 3, and writes a corner plot. The
maintained default uses 128 uv points, 24 walkers, and 1000 steps.
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

GALARIO is licensed under LGPLv3.
