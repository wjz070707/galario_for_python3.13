#!/usr/bin/env python3
"""Lightweight GALARIO benchmark using the documented quickstart uv table."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np


def find_default_uvtable() -> Path:
    """Find docs/uvtable.txt from either the source or CMake build tree."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "docs" / "uvtable.txt"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Unable to find docs/uvtable.txt; pass --uvtable explicitly."
    )


def load_case(path: Path, points: int, size: int):
    # The documented table is below 1 MiB, so reading it once avoids NumPy's
    # version-dependent max_rows behavior around leading comment lines.
    table = np.loadtxt(path)[:points]
    if table.ndim != 2 or table.shape[1] != 5:
        raise ValueError("Expected uvtable columns: u, v, Re, Im, weights")

    u_m, v_m, vis_re, vis_im, weights = table.T
    wavelength_m = 1.0e-3
    u = np.ascontiguousarray(u_m / wavelength_m)
    v = np.ascontiguousarray(v_m / wavelength_m)

    max_uv = np.max(np.hypot(u, v))
    dxy = 1.0 / (2.5 * max_uv)
    axis = (np.arange(size, dtype=np.float64) - size / 2) * dxy
    x, y = np.meshgrid(axis, axis)
    sigma = 0.5 * np.pi / (180.0 * 3600.0)
    image = np.ascontiguousarray(
        np.exp(-0.5 * (x * x + y * y) / (sigma * sigma)) * dxy * dxy
    )

    return image, dxy, u, v, vis_re, vis_im, weights


def median_seconds(call, repeats: int, loops: int) -> float:
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(loops):
            call()
        durations.append((time.perf_counter() - start) / loops)
    return statistics.median(durations)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uvtable", type=Path)
    parser.add_argument("--points", type=int, default=2_000)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--loops", type=int, default=20)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.points < 1 or args.size < 2 or args.repeats < 1 or args.loops < 1:
        parser.error("points, repeats, and loops must be positive; size must be >= 2")
    if args.size % 2:
        parser.error("size must be even")

    uvtable = args.uvtable or find_default_uvtable()
    image, dxy, u, v, vis_re, vis_im, weights = load_case(
        uvtable, args.points, args.size
    )

    import galario

    if args.gpu:
        if not galario.HAVE_CUDA:
            parser.error("this build does not provide the CUDA backend")
        from galario import double_cuda as backend
        backend_name = "cuda"
    else:
        from galario import double as backend
        backend.threads(args.threads)
        backend_name = "cpu"

    context = backend.create_image_context(
        args.size,
        args.size,
        u,
        v,
        vis_re,
        vis_im,
        weights,
        backend=backend.BACKEND_FFT,
    )

    cached_call = lambda: backend.chi2_image(
        ctx=context, image=image, dxy=dxy
    )
    uncached_call = lambda: backend.chi2Image(
        image,
        dxy,
        u,
        v,
        vis_re,
        vis_im,
        weights,
        backend=backend.BACKEND_FFT,
    )

    # Warm up imports, allocator state, and the context-owned transform plan.
    cached_value = cached_call()
    uncached_value = uncached_call()
    if not np.isclose(cached_value, uncached_value, rtol=1e-12, atol=1e-12):
        raise RuntimeError("cached and uncached benchmark paths disagree")

    cached = median_seconds(cached_call, args.repeats, args.loops)
    uncached = median_seconds(uncached_call, args.repeats, args.loops)
    result = {
        "backend": backend_name,
        "points": len(u),
        "size": args.size,
        "repeats": args.repeats,
        "loops": args.loops,
        "cached_ms": cached * 1_000.0,
        "uncached_ms": uncached * 1_000.0,
        "speedup": uncached / cached,
        "uvtable": str(uvtable.resolve()),
    }

    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            f"{backend_name}: points={result['points']} size={args.size} "
            f"cached={result['cached_ms']:.3f} ms "
            f"uncached={result['uncached_ms']:.3f} ms "
            f"speedup={result['speedup']:.2f}x"
        )


if __name__ == "__main__":
    main()
