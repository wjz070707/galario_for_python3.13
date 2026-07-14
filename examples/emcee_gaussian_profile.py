#!/usr/bin/env python3
"""Fit the documented uv table with GALARIO's profile API and emcee 3.

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
OUTPUT = REPOSITORY_ROOT / "triangle_example.png"
# Set UV_SAMPLES to None to use the complete uv table. A value larger than the
# table is also treated as "all data" rather than as an error.
UV_SAMPLES = None
SOURCE_RADIUS_ARCSEC = 4.0
FOV_PADDING = 4.0 / 3.0
MIN_IMAGE_SIZE = 128
RADIAL_CELLS = 1280
WALKERS = 24
STEPS = 1000
BURN_IN = 500
CPU_THREADS = 2
RANDOM_SEED = 12345
SHOW_PROGRESS = True
USE_GPU = False
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
    log10_f0,
    sigma_arcsec,
    r_min: float,
    dr: float,
    radial_cells: int,
) -> np.ndarray:
    radius = r_min + dr * np.arange(radial_cells)
    amplitude = np.atleast_1d(10.0**np.asarray(log10_f0))[:, None]
    sigma = np.atleast_1d(np.asarray(sigma_arcsec) * arcsec)[:, None]
    profiles = amplitude * np.exp(
        -0.5 * (radius[None, :] / sigma) ** 2
    )
    return profiles[0] if np.ndim(log10_f0) == 0 else profiles


def log_prior(parameters: np.ndarray) -> np.ndarray:
    parameters = np.atleast_2d(parameters)
    inside = np.all(
        (parameters >= PARAMETER_RANGES[:, 0])
        & (parameters <= PARAMETER_RANGES[:, 1]),
        axis=1,
    )
    # Uniform in the sampled coordinates, including log10(f0).
    return np.where(inside, 0.0, -np.inf)


def make_log_probability(
    backend,
    profile_context,
    r_min: float,
    dr: float,
    radial_cells: int,
    nxy: int,
    dxy: float,
):
    def log_probability(parameters: np.ndarray) -> np.ndarray:
        parameters = np.atleast_2d(parameters)
        prior = log_prior(parameters)
        result = np.full(len(parameters), -np.inf)
        valid = np.isfinite(prior)
        if np.any(valid):
            selected = parameters[valid]
            log10_f0, sigma, inc, pa, dra, ddec = selected.T
            intensity_batch = gaussian_profile(
                log10_f0, sigma, r_min, dr, radial_cells
            )
            chi2 = backend.chi2_profile(
                intensity_batch,
                r_min,
                dr,
                nxy,
                dxy,
                ctx=profile_context,
                inc_batch=inc * deg,
                PA_batch=pa * deg,
                dRA_batch=dra * arcsec,
                dDec_batch=ddec * arcsec,
            )
            result[valid] = prior[valid] - 0.5 * chi2
        return result

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

    # The model FOV is a source/offset choice; uv coverage sets the pixel size.
    image_fov = backend.estimate_fov_from_source(
        SOURCE_RADIUS_ARCSEC * arcsec,
        offset=(
            max(abs(PARAMETER_RANGES[4, 0]), abs(PARAMETER_RANGES[4, 1]))
            * arcsec,
            max(abs(PARAMETER_RANGES[5, 0]), abs(PARAMETER_RANGES[5, 1]))
            * arcsec,
        ),
        padding=FOV_PADDING,
        verbose=True,
    )
    suggested_nxy, dxy = backend.get_image_size_from_fov(
        u, v, image_fov, verbose=True
    )
    nxy = max(MIN_IMAGE_SIZE, suggested_nxy)
    dxy = image_fov / nxy

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
        f"{image_fov / arcsec:.3f} arcsec FOV, and a "
        f"{radial_extent / arcsec:.3f} arcsec radial grid"
    )

    # Always reuse a context when observations stay fixed. It uploads u/v,
    # visibilities, and weights once and retains FFT plans and work buffers.
    # Calling chi2_profile without ctx inside MCMC repeats those transfers.
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
    print(
        "reusing one context avoids repeated observation transfers "
        "and FFT workspace setup"
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
        WALKERS,
        len(INITIAL_POSITION),
        log_probability,
        vectorize=True,
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
