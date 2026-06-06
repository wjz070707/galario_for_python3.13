#!/usr/bin/env python3
"""Fit the documented uv table with GALARIO's profile API and emcee 3."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from galario import arcsec, deg
from galario import double as galario_cpu


# USER CONFIGURATION
# Edit these values in VSCode, then run this file directly.
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
UVTABLE = REPOSITORY_ROOT / "docs" / "uvtable.txt"
OUTPUT = REPOSITORY_ROOT / "triangle_example.png"
# Set UV_SAMPLES to None to use the complete uv table. A value larger than the
# table is also treated as "all data" rather than as an error.
UV_SAMPLES = None
MIN_IMAGE_SIZE = 128
RADIAL_CELLS = 1280
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
INITIAL_POSITION = np.array([10.0, 0.5, 70.0, 60.0, 0.0, 0.0])
INITIAL_SCALE = np.array([0.02, 0.01, 0.2, 0.2, 0.01, 0.01])
LABELS = [
    r"$\log_{10} f_0$",
    r"$\sigma$",
    r"$i$",
    r"$PA$",
    r"$\Delta RA$",
    r"$\Delta Dec$",
]


def gaussian_profile(
    log10_f0: float,
    sigma_arcsec: float,
    r_min: float,
    dr: float,
    radial_cells: int,
) -> np.ndarray:
    radius = r_min + dr * np.arange(radial_cells)
    sigma = sigma_arcsec * arcsec
    return 10.0**log10_f0 * np.exp(-0.5 * (radius / sigma) ** 2)


def log_prior(parameters: np.ndarray) -> float:
    if np.all(
        (parameters >= PARAMETER_RANGES[:, 0])
        & (parameters <= PARAMETER_RANGES[:, 1])
    ):
        # Uniform in the sampled coordinates, including log10(f0).
        return 0.0
    return -np.inf


def make_log_probability(
    backend,
    profile_context,
    r_min: float,
    dr: float,
    radial_cells: int,
    nxy: int,
    dxy: float,
):
    def log_probability(parameters: np.ndarray) -> float:
        prior = log_prior(parameters)
        if not np.isfinite(prior):
            return -np.inf

        log10_f0, sigma, inc, pa, dra, ddec = parameters
        intensity = gaussian_profile(
            log10_f0, sigma, r_min, dr, radial_cells
        )
        chi2 = backend.chi2_profile(
            intensity,
            r_min,
            dr,
            nxy,
            dxy,
            ctx=profile_context,
            inc=inc * deg,
            PA=pa * deg,
            dRA=dra * arcsec,
            dDec=ddec * arcsec,
        )
        return prior - 0.5 * chi2

    return log_probability


def main() -> None:
    if WALKERS < 2 * len(INITIAL_POSITION) or WALKERS % 2:
        raise ValueError("walkers must be even and at least twice ndim (12)")
    if BURN_IN >= STEPS:
        raise ValueError("burn-in must be smaller than steps")

    from emcee import EnsembleSampler

    full_table = np.loadtxt(UVTABLE)
    available_samples = len(full_table)
    requested_samples = (
        available_samples if UV_SAMPLES is None else int(UV_SAMPLES)
    )
    used_samples = min(max(requested_samples, 1), available_samples)
    if used_samples == available_samples:
        table = full_table
    else:
        sample_indices = np.linspace(
            0, available_samples - 1, used_samples, dtype=np.int64
        )
        table = full_table[sample_indices]
    print(
        f"uv table contains {available_samples} rows; "
        f"using {used_samples}"
    )
    u_m, v_m, vis_re, vis_im, weights = table.T
    wavelength_m = 1.0e-3
    u = np.ascontiguousarray(u_m / wavelength_m)
    v = np.ascontiguousarray(v_m / wavelength_m)
    vis_re = np.ascontiguousarray(vis_re)
    vis_im = np.ascontiguousarray(vis_im)
    weights = np.ascontiguousarray(weights)

    if USE_GPU:
        from galario import double_cuda as backend

        backend.use_gpu(GPU_DEVICE)
        print(f"using CUDA device {GPU_DEVICE}")
    else:
        backend = galario_cpu
        backend.threads(CPU_THREADS)
        print(f"using CPU with {CPU_THREADS} OpenMP threads")

    # Infer a safe FFT image geometry from the observed uv coverage.
    suggested_nxy, dxy = backend.get_image_size(u, v, verbose=True)
    nxy = max(MIN_IMAGE_SIZE, suggested_nxy)

    r_min = 0.0
    dr = 0.004 * arcsec
    radial_extent = (RADIAL_CELLS - 1) * dr
    max_sigma = PARAMETER_RANGES[1, 1] * arcsec
    if radial_extent < 6.0 * max_sigma:
        raise ValueError(
            "radial grid must extend to at least six times max sigma"
        )
    print(
        f"using {len(table)} uv samples, {nxy}x{nxy} pixels, "
        f"and a {radial_extent / arcsec:.3f} arcsec radial grid"
    )

    # The profile FFT path now reuses the image context's observations,
    # transform plan, and work buffers across all likelihood evaluations.
    profile_context = backend.create_image_context(
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
        f"profile context backend: "
        f"requested={profile_context.requested_backend}, "
        f"resolved={profile_context.resolved_backend}"
    )

    # Cross-check the cached chi2_profile path against sample_profile followed
    # by the public chi-squared reduction.
    initial_intensity = gaussian_profile(
        INITIAL_POSITION[0],
        INITIAL_POSITION[1],
        r_min,
        dr,
        RADIAL_CELLS,
    )
    initial_vis = backend.sample_profile(
        initial_intensity,
        r_min,
        dr,
        nxy,
        dxy,
        u=u,
        v=v,
        inc=INITIAL_POSITION[2] * deg,
        PA=INITIAL_POSITION[3] * deg,
        dRA=INITIAL_POSITION[4] * arcsec,
        dDec=INITIAL_POSITION[5] * arcsec,
        backend=backend.BACKEND_FFT,
    )
    sampled_chi2 = backend.reduce_chi2(
        vis_re, vis_im, weights, initial_vis
    )
    profile_chi2 = backend.chi2_profile(
        initial_intensity,
        r_min,
        dr,
        nxy,
        dxy,
        ctx=profile_context,
        inc=INITIAL_POSITION[2] * deg,
        PA=INITIAL_POSITION[3] * deg,
        dRA=INITIAL_POSITION[4] * arcsec,
        dDec=INITIAL_POSITION[5] * arcsec,
    )
    np.testing.assert_allclose(sampled_chi2, profile_chi2, rtol=1e-12)
    print(
        "sample_profile + reduce_chi2 agrees with contextual chi2_profile: "
        f"{profile_chi2:.6e}"
    )

    log_probability = make_log_probability(
        backend,
        profile_context,
        r_min,
        dr,
        RADIAL_CELLS,
        nxy,
        dxy,
    )

    rng = np.random.default_rng(RANDOM_SEED)
    positions = INITIAL_POSITION + INITIAL_SCALE * rng.standard_normal(
        (WALKERS, len(INITIAL_POSITION))
    )
    sampler = EnsembleSampler(
        WALKERS, len(INITIAL_POSITION), log_probability
    )
    sampler.run_mcmc(positions, STEPS, progress=SHOW_PROGRESS)

    samples = sampler.get_chain(discard=BURN_IN, flat=True)
    mean_acceptance = np.mean(sampler.acceptance_fraction)
    print(
        f"completed {STEPS} steps with {WALKERS} walkers; "
        f"retained {len(samples)} posterior samples; "
        f"mean acceptance fraction {mean_acceptance:.3f}"
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
