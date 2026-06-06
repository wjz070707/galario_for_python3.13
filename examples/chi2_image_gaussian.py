#!/usr/bin/env python3
"""Fit the documented uv table with GALARIO's image API and emcee 3.

Performance-critical design: one Context is created before sampling and reused
for every vectorized walker batch. Do not move Context creation into the
likelihood function; that repeats observation transfers and setup work.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from galario import arcsec, deg
from galario import double as galario_cpu


# USER CONFIGURATION
# Edit these values in VSCode, then run this file directly.
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
UVTABLE = REPOSITORY_ROOT / "docs" / "uvtable.txt"
OUTPUT = REPOSITORY_ROOT / "triangle_image_example.png"
UV_SAMPLES = None  # None uses every valid row in the uv table.
MIN_IMAGE_SIZE = 128
WALKERS = 24
STEPS = 1000
BURN_IN = 500
CPU_THREADS = 2
RANDOM_SEED = 12345
SHOW_PROGRESS = True
USE_GPU = True
GPU_DEVICE = 0


PARAMETER_RANGES = np.array(
    [
        [1.0, 20.0],   # log10 central brightness [Jy/sr]
        [0.03, 0.8],   # Gaussian sigma [arcsec]
        [0.0, 90.0],   # inclination [deg]
        [0.0, 180.0],  # position angle [deg]
        [-2.0, 2.0],   # RA offset [arcsec]
        [-2.0, 2.0],   # Dec offset [arcsec]
    ],
    dtype=np.float64,
)
INITIAL_POSITION = np.array([10.7, 0.30, 73.0, 60.0, -0.35, -0.08])
INITIAL_SCALE = np.array([0.02, 0.01, 0.2, 0.2, 0.005, 0.005])
LABELS = [
    r"$\log_{10} f_0$",
    r"$\sigma$",
    r"$i$",
    r"$PA$",
    r"$\Delta RA$",
    r"$\Delta Dec$",
]


def gaussian_components(
    log10_f0,
    sigma_arcsec,
    dxy: float,
) -> np.ndarray:
    """Return one Gaussian component row per parameter vector."""
    log10_f0 = np.asarray(log10_f0)
    sigma_arcsec = np.asarray(sigma_arcsec)
    central_brightness = 10.0**log10_f0  # Jy/sr
    central_pixel_flux = central_brightness * dxy**2  # Jy/pixel
    return np.column_stack(
        (central_pixel_flux, sigma_arcsec * arcsec)
    ).astype(
        np.float64, copy=False
    )


def log_prior(parameters: np.ndarray) -> np.ndarray:
    parameters = np.atleast_2d(parameters)
    inside = (
        (parameters >= PARAMETER_RANGES[:, 0])
        & (parameters <= PARAMETER_RANGES[:, 1])
    )
    return np.where(np.all(inside, axis=1), 0.0, -np.inf)


def make_log_probability(backend, image_context, dxy: float):
    def log_probability(parameters: np.ndarray) -> float:
        parameters = np.atleast_2d(parameters)
        prior = log_prior(parameters)
        result = np.full(len(parameters), -np.inf)
        valid = np.isfinite(prior)
        if np.any(valid):
            selected = parameters[valid]
            log10_f0, sigma, inc, pa, dra, ddec = selected.T
            chi2 = backend.chi2_image(
                ctx=image_context,
                dxy=dxy,
                gauss_params_batch=gaussian_components(
                    log10_f0, sigma, dxy
                ),
                inc_batch=inc * deg,
                PA_batch=pa * deg,
                dRA_batch=dra * arcsec,
                dDec_batch=ddec * arcsec,
                origin="lower",
            )
            result[valid] = prior[valid] - 0.5 * chi2
        return result

    return log_probability


def load_uv_table():
    full_table = np.loadtxt(UVTABLE)
    available_samples = len(full_table)
    requested_samples = (
        available_samples if UV_SAMPLES is None else int(UV_SAMPLES)
    )
    used_samples = min(max(requested_samples, 1), available_samples)
    if used_samples == available_samples:
        table = full_table
    else:
        indices = np.linspace(
            0, available_samples - 1, used_samples, dtype=np.int64
        )
        table = full_table[indices]

    print(
        f"uv table contains {available_samples} rows; "
        f"using {used_samples}"
    )
    u_m, v_m, vis_re, vis_im, weights = table.T
    wavelength_m = 1.0e-3
    return tuple(
        np.ascontiguousarray(values)
        for values in (
            u_m / wavelength_m,
            v_m / wavelength_m,
            vis_re,
            vis_im,
            weights,
        )
    )


def main() -> None:
    if WALKERS < 2 * len(INITIAL_POSITION) or WALKERS % 2:
        raise ValueError("walkers must be even and at least twice ndim (12)")
    if BURN_IN >= STEPS:
        raise ValueError("burn-in must be smaller than steps")

    from emcee import EnsembleSampler

    u, v, vis_re, vis_im, weights = load_uv_table()

    if USE_GPU:
        from galario import double_cuda as backend

        backend.use_gpu(GPU_DEVICE)
        print(f"using CUDA device {GPU_DEVICE}")
    else:
        backend = galario_cpu
        backend.threads(CPU_THREADS)
        print(f"using CPU with {CPU_THREADS} OpenMP threads")

    # Derive a safe image geometry from the observed uv coverage.
    suggested_nxy, dxy = backend.get_image_size(u, v, verbose=True)
    nxy = max(MIN_IMAGE_SIZE, suggested_nxy)
    print(
        f"using a {nxy}x{nxy} image with "
        f"{dxy / arcsec:.6f} arcsec pixels"
    )

    # Reuse fixed observations, FFT plans, and backend work buffers in MCMC.
    image_context = backend.create_image_context(
        nxy,
        nxy,
        u,
        v,
        vis_re,
        vis_im,
        weights,
        backend=backend.BACKEND_FFT,
    )
    print(
        f"image context backend: requested={image_context.requested_backend}, "
        f"resolved={image_context.resolved_backend}"
    )
    print(
        "reusing one context avoids repeated observation transfers "
        "and FFT workspace setup"
    )

    # Demonstrate sample_image and verify it agrees with contextual chi2_image.
    initial_gaussian = gaussian_components(
        INITIAL_POSITION[0], INITIAL_POSITION[1], dxy
    )
    initial_vis = backend.sample_image(
        nx=nxy,
        ny=nxy,
        dxy=dxy,
        u=u,
        v=v,
        gauss_params=initial_gaussian,
        inc=INITIAL_POSITION[2] * deg,
        PA=INITIAL_POSITION[3] * deg,
        dRA=INITIAL_POSITION[4] * arcsec,
        dDec=INITIAL_POSITION[5] * arcsec,
        origin="lower",
        backend=backend.BACKEND_FFT,
    )
    sampled_chi2 = backend.reduce_chi2(
        vis_re, vis_im, weights, initial_vis
    )
    contextual_chi2 = backend.chi2_image(
        ctx=image_context,
        dxy=dxy,
        gauss_params=initial_gaussian,
        inc=INITIAL_POSITION[2] * deg,
        PA=INITIAL_POSITION[3] * deg,
        dRA=INITIAL_POSITION[4] * arcsec,
        dDec=INITIAL_POSITION[5] * arcsec,
        origin="lower",
    )
    np.testing.assert_allclose(sampled_chi2, contextual_chi2, rtol=1e-12)
    print(
        "sample_image + reduce_chi2 agrees with contextual chi2_image: "
        f"{contextual_chi2:.6e}"
    )

    log_probability = make_log_probability(
        backend, image_context, dxy
    )
    rng = np.random.default_rng(RANDOM_SEED)
    positions = INITIAL_POSITION + INITIAL_SCALE * rng.standard_normal(
        (WALKERS, len(INITIAL_POSITION))
    )
    sampler = EnsembleSampler(
        WALKERS,
        len(INITIAL_POSITION),
        log_probability,
        vectorize=True,
    )
    sampler.run_mcmc(positions, STEPS, progress=SHOW_PROGRESS)

    samples = sampler.get_chain(discard=BURN_IN, flat=True)
    mean_acceptance = np.mean(sampler.acceptance_fraction)
    percentiles = np.percentile(samples, [16, 50, 84], axis=0)
    print(
        f"completed {STEPS} steps with {WALKERS} walkers; "
        f"retained {len(samples)} posterior samples; "
        f"mean acceptance fraction {mean_acceptance:.3f}"
    )
    for label, lower, median, upper in zip(
        LABELS, percentiles[0], percentiles[1], percentiles[2]
    ):
        print(
            f"{label}: {median:.6g} "
            f"(+{upper - median:.3g}/-{median - lower:.3g})"
        )

    import matplotlib

    matplotlib.use("Agg")
    import corner

    figure = corner.corner(
        samples,
        labels=LABELS,
        show_titles=True,
        quantiles=[0.16, 0.50, 0.84],
    )
    figure.savefig(OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
