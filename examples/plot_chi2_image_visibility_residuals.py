#!/usr/bin/env python3
"""Plot visibility-domain diagnostics for the image chi2 Gaussian example.

The image-plane residual is not the quantity minimized by ``chi2_image``.
GALARIO compares complex visibilities at the observed ``(u, v)`` samples, so a
fit diagnostic should compare model and data in the uv plane.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from galario import arcsec, deg
from galario import double as galario_cpu


# USER CONFIGURATION
# Parameters follow examples/chi2_image_gaussian.py:
# log10 central brightness [Jy/sr], sigma [arcsec], inclination [deg],
# position angle [deg], RA offset [arcsec], Dec offset [arcsec].
PARAMETERS = np.array([10.7, 0.30, 73.0, 60.0, -0.35, -0.08])
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PICTURES_DIR = REPOSITORY_ROOT / "examples" / "pictures"
UVTABLE = REPOSITORY_ROOT / "docs" / "uvtable.txt"
UV_DISTANCE_OUTPUT = PICTURES_DIR / "visibility_uv_distance_image_example.png"
RESIDUAL_DISTANCE_OUTPUT = PICTURES_DIR / "visibility_residual_distance_image_example.png"
PARITY_OUTPUT = PICTURES_DIR / "visibility_parity_image_example.png"
UV_RESIDUAL_OUTPUT = PICTURES_DIR / "visibility_uv_residual_image_example.png"
RESIDUAL_HIST_OUTPUT = PICTURES_DIR / "visibility_residual_hist_image_example.png"
SURFACE_OUTPUT = PICTURES_DIR / "visibility_surface_image_example.png"
SURFACE_GIF_OUTPUT = PICTURES_DIR / "visibility_surface_image_example.gif"
SOURCE_RADIUS_ARCSEC = 4.0
MAX_OFFSET_ARCSEC = 2.0
FOV_PADDING = 4.0 / 3.0
MIN_IMAGE_SIZE = 128
CPU_THREADS = 8
UV_SAMPLES = None  # None uses every valid row in the uv table.
SURFACE_GRID = 80
MAKE_SURFACE_PLOT = True
MAKE_SURFACE_GIF = True
SURFACE_GIF_FRAMES = 72


def gaussian_components(log10_f0: float, sigma_arcsec: float, dxy: float) -> np.ndarray:
    central_brightness = 10.0**log10_f0  # Jy/sr
    central_pixel_flux = central_brightness * dxy**2  # Jy/pixel
    return np.array(
        [[central_pixel_flux, sigma_arcsec * arcsec]], dtype=np.float64
    )


def load_uv_table():
    full_table = np.loadtxt(UVTABLE)
    available_samples = len(full_table)
    requested_samples = available_samples if UV_SAMPLES is None else int(UV_SAMPLES)
    used_samples = min(max(requested_samples, 1), available_samples)
    if used_samples == available_samples:
        table = full_table
    else:
        indices = np.linspace(0, available_samples - 1, used_samples, dtype=np.int64)
        table = full_table[indices]

    u_m, v_m, vis_re, vis_im, weights = table.T
    wavelength_m = 1.0e-3
    return tuple(
        np.ascontiguousarray(values)
        for values in (u_m / wavelength_m, v_m / wavelength_m, vis_re, vis_im, weights)
    )


def weighted_bin(x, y, weights, nbins=35):
    bins = np.linspace(np.nanmin(x), np.nanmax(x), nbins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    values = np.full(nbins, np.nan)
    errors = np.full(nbins, np.nan)

    for idx in range(nbins):
        selected = (x >= bins[idx]) & (x < bins[idx + 1])
        if not np.any(selected):
            continue
        w = weights[selected]
        values[idx] = np.average(y[selected], weights=w)
        errors[idx] = 1.0 / np.sqrt(np.sum(w))

    return centers, values, errors


def set_matched_limits(ax, x, y):
    finite = np.isfinite(x) & np.isfinite(y)
    lower = min(np.nanmin(x[finite]), np.nanmin(y[finite]))
    upper = max(np.nanmax(x[finite]), np.nanmax(y[finite]))
    padding = 0.05 * (upper - lower) if upper > lower else 1.0
    ax.set_xlim(lower - padding, upper + padding)
    ax.set_ylim(lower - padding, upper + padding)


def draw_uv_distance_visibility(u, v, data_re, data_im, weights, model_vis):
    """Plot Re/Im visibility against uv distance.

    This compresses the two-dimensional uv plane to ``sqrt(u^2 + v^2)``. It is
    most physically interpretable for axisymmetric or deprojected axisymmetric
    models; for a general 2D image model it is only a compact trend view.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    uv_distance = np.hypot(u, v) / 1.0e6
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    for ax, data, model, ylabel, title in (
        (axes[0], data_re, model_vis.real, "Real visibility [Jy]", "Real part vs uv distance"),
        (axes[1], data_im, model_vis.imag, "Imag visibility [Jy]", "Imag part vs uv distance"),
    ):
        ax.scatter(uv_distance, data, s=7, color="0.15", alpha=0.18, label="Data points")
        ax.scatter(
            uv_distance,
            model,
            s=7,
            color="#d1495b",
            alpha=0.16,
            label="Model at same uv points",
        )
        centers, data_values, data_errors = weighted_bin(uv_distance, data, weights)
        _, model_values, _ = weighted_bin(uv_distance, model, weights)
        ax.errorbar(
            centers,
            data_values,
            yerr=data_errors,
            fmt="o",
            ms=4.0,
            color="#2d6cdf",
            ecolor="#9bb7f0",
            label="Weighted data bins",
        )
        ax.plot(centers, model_values, color="#d1495b", lw=2.0, label="Weighted model bins")
        ax.set_xlabel("uv distance [Mlambda]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)

    fig.savefig(UV_DISTANCE_OUTPUT, dpi=180)
    print(f"wrote {UV_DISTANCE_OUTPUT}")


def draw_residual_vs_uv_distance(u, v, data_re, data_im, weights, model_vis):
    """Plot Re/Im residuals against uv distance.

    This shows whether residuals have a systematic radial trend. Like the
    uv-distance visibility plot, it is a one-dimensional compression and should
    not be used as the only diagnostic for a non-axisymmetric 2D image model.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    uv_distance = np.hypot(u, v) / 1.0e6
    residual = (data_re + 1j * data_im) - model_vis
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)

    for ax, values, title in (
        (axes[0], residual.real, "Real residual vs uv distance"),
        (axes[1], residual.imag, "Imag residual vs uv distance"),
    ):
        ax.axhline(0.0, color="0.25", lw=1.0)
        ax.scatter(uv_distance, values, s=8, color="#1b998b", alpha=0.3)
        centers, binned, errors = weighted_bin(uv_distance, values, weights)
        ax.errorbar(centers, binned, yerr=errors, fmt="o", ms=3.5, color="#0b6e69")
        ax.set_xlabel("uv distance [Mlambda]")
        ax.set_ylabel("Data - model [Jy]")
        ax.set_title(title)
        ax.grid(alpha=0.22)

    fig.savefig(RESIDUAL_DISTANCE_OUTPUT, dpi=180)
    print(f"wrote {RESIDUAL_DISTANCE_OUTPUT}")


def draw_parity_plot(u, v, data_re, data_im, model_vis):
    """Plot data against model at identical uv samples.

    This is the 1:1 diagnostic: every point compares one observed visibility
    component to the model visibility component sampled at the same ``(u, v)``.
    Points close to the diagonal indicate good point-by-point agreement.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    uv_distance = np.hypot(u, v) / 1.0e6
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    for ax, data, model, label in (
        (axes[0], data_re, model_vis.real, "Real"),
        (axes[1], data_im, model_vis.imag, "Imag"),
    ):
        scatter = ax.scatter(model, data, s=8, c=uv_distance, cmap="viridis", alpha=0.28)
        set_matched_limits(ax, model, data)
        xmin, xmax = ax.get_xlim()
        ax.plot([xmin, xmax], [xmin, xmax], color="0.2", lw=1.0, label="1:1")
        ax.set_xlabel(f"Model {label} visibility [Jy]")
        ax.set_ylabel(f"Data {label} visibility [Jy]")
        ax.set_title(f"{label} data vs model")
        ax.grid(alpha=0.22)
        ax.legend(frameon=False)
        fig.colorbar(scatter, ax=ax, label="uv distance [Mlambda]")

    fig.savefig(PARITY_OUTPUT, dpi=180)
    print(f"wrote {PARITY_OUTPUT}")


def draw_uv_residual_map(u, v, data_re, data_im, weights, model_vis):
    """Plot where the largest weighted residuals occur in the uv plane.

    This keeps the full two-dimensional uv coordinates and colors each point by
    ``sqrt(weight) * abs(data - model)``. It is usually more informative than a
    uv-distance plot for a general 2D image model.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    residual = (data_re + 1j * data_im) - model_vis
    whitened_residual = np.sqrt(weights) * np.abs(residual)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    scatter = ax.scatter(
        u / 1.0e6,
        v / 1.0e6,
        c=whitened_residual,
        s=12,
        cmap="magma",
        alpha=0.8,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("u [Mlambda]")
    ax.set_ylabel("v [Mlambda]")
    ax.set_title("Weighted complex residual amplitude in uv plane")
    fig.colorbar(scatter, ax=ax, label=r"$\sqrt{w}\,|V_\mathrm{data}-V_\mathrm{model}|$")

    fig.savefig(UV_RESIDUAL_OUTPUT, dpi=180)
    print(f"wrote {UV_RESIDUAL_OUTPUT}")


def draw_residual_histogram(data_re, data_im, weights, model_vis, chi2):
    """Plot the distribution of weighted complex residual amplitudes.

    This is a compact scalar summary of the point-by-point residuals. The title
    reports the same weighted complex chi-squared used by ``chi2_image``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    residual = (data_re + 1j * data_im) - model_vis
    whitened_residual = np.sqrt(weights) * np.abs(residual)
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.hist(whitened_residual, bins=45, color="#6c7a89", alpha=0.85)
    ax.set_xlabel(r"$\sqrt{w}\,|V_\mathrm{data}-V_\mathrm{model}|$")
    ax.set_ylabel("Count")
    ax.set_title(f"Weighted residual distribution; chi2 = {chi2:.6e}")
    ax.grid(alpha=0.22)

    fig.savefig(RESIDUAL_HIST_OUTPUT, dpi=180)
    print(f"wrote {RESIDUAL_HIST_OUTPUT}")


def draw_visibility_surface(backend, nxy, dxy, data_u, data_v, data_re, data_im, weights, params):
    """Plot Re/Im model visibility as uv-plane surfaces with data points.

    The complex visibility is two real-valued surfaces, ``Re V(u, v)`` and
    ``Im V(u, v)``. The optional GIF rotates these surfaces so it is easier to
    judge whether observed points sit close to the model surface.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    limit = np.nanpercentile(np.hypot(data_u, data_v), 98.0)
    axis = np.linspace(-limit, limit, SURFACE_GRID)
    uu, vv = np.meshgrid(axis, axis)
    model_grid = backend.sample_image(
        nx=nxy,
        ny=nxy,
        dxy=dxy,
        u=np.ascontiguousarray(uu.ravel()),
        v=np.ascontiguousarray(vv.ravel()),
        gauss_params=gaussian_components(params[0], params[1], dxy),
        inc=params[2] * deg,
        PA=params[3] * deg,
        dRA=params[4] * arcsec,
        dDec=params[5] * arcsec,
        origin="lower",
        backend=backend.BACKEND_FFT,
    ).reshape(uu.shape)

    fig = plt.figure(figsize=(13, 6), constrained_layout=True)
    axes = []
    for idx, (part, data, title) in enumerate(
        ((model_grid.real, data_re, "Real visibility surface"), (model_grid.imag, data_im, "Imag visibility surface")),
        start=1,
    ):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        axes.append(ax)
        ax.plot_surface(
            uu / 1.0e6,
            vv / 1.0e6,
            part,
            cmap="viridis",
            linewidth=0,
            antialiased=True,
            alpha=0.78,
        )
        stride = max(len(data_u) // 1200, 1)
        ax.scatter(
            data_u[::stride] / 1.0e6,
            data_v[::stride] / 1.0e6,
            data[::stride],
            c=np.sqrt(weights[::stride]),
            cmap="magma",
            s=8,
            alpha=0.8,
        )
        ax.set_xlabel("u [Mlambda]")
        ax.set_ylabel("v [Mlambda]")
        ax.set_zlabel("Visibility [Jy]")
        ax.set_title(title)

    fig.savefig(SURFACE_OUTPUT, dpi=180)
    print(f"wrote {SURFACE_OUTPUT}")

    if MAKE_SURFACE_GIF:
        from matplotlib.animation import FuncAnimation, PillowWriter

        def rotate(frame):
            angle = 360.0 * frame / SURFACE_GIF_FRAMES
            for ax in axes:
                ax.view_init(elev=25.0, azim=angle)
            return axes

        animation = FuncAnimation(
            fig,
            rotate,
            frames=SURFACE_GIF_FRAMES,
            interval=80,
            blit=False,
        )
        animation.save(SURFACE_GIF_OUTPUT, writer=PillowWriter(fps=12), dpi=130)
        print(f"wrote {SURFACE_GIF_OUTPUT}")


def main() -> None:
    PICTURES_DIR.mkdir(parents=True, exist_ok=True)

    backend = galario_cpu
    backend.threads(CPU_THREADS)

    u, v, data_re, data_im, weights = load_uv_table()
    image_fov = backend.estimate_fov_from_source(
        SOURCE_RADIUS_ARCSEC * arcsec,
        offset=(MAX_OFFSET_ARCSEC * arcsec, MAX_OFFSET_ARCSEC * arcsec),
        padding=FOV_PADDING,
        verbose=True,
    )
    suggested_nxy, dxy = backend.get_image_size_from_fov(
        u, v, image_fov, verbose=True
    )
    nxy = max(MIN_IMAGE_SIZE, suggested_nxy)
    dxy = image_fov / nxy

    model_vis = backend.sample_image(
        nx=nxy,
        ny=nxy,
        dxy=dxy,
        u=u,
        v=v,
        gauss_params=gaussian_components(PARAMETERS[0], PARAMETERS[1], dxy),
        inc=PARAMETERS[2] * deg,
        PA=PARAMETERS[3] * deg,
        dRA=PARAMETERS[4] * arcsec,
        dDec=PARAMETERS[5] * arcsec,
        origin="lower",
        backend=backend.BACKEND_FFT,
    )
    chi2 = backend.reduce_chi2(data_re, data_im, weights, model_vis)
    draw_uv_distance_visibility(u, v, data_re, data_im, weights, model_vis)
    draw_residual_vs_uv_distance(u, v, data_re, data_im, weights, model_vis)
    draw_parity_plot(u, v, data_re, data_im, model_vis)
    draw_uv_residual_map(u, v, data_re, data_im, weights, model_vis)
    draw_residual_histogram(data_re, data_im, weights, model_vis, chi2)

    if MAKE_SURFACE_PLOT:
        draw_visibility_surface(backend, nxy, dxy, u, v, data_re, data_im, weights, PARAMETERS)


if __name__ == "__main__":
    main()
