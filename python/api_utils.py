from __future__ import annotations

import numpy as np

from .api_constants import arcsec


def set_v_origin(origin: str) -> float:
    if origin == "upper":
        return 1.0
    if origin == "lower":
        return -1.0
    raise ValueError(f"Expect origin to be 'upper' or 'lower', got {origin!r}")


def check_obs(vis_obs_re, vis_obs_im, vis_obs_w, vis=None, u=None, v=None):
    nd = len(vis_obs_re)
    assert len(vis_obs_im) == nd, "Wrong array length: vis_obs_im."
    assert len(vis_obs_w) == nd, "Wrong array length: vis_obs_w."
    if vis is not None:
        assert len(vis) == nd, "Wrong array length: vis."
    if u is not None:
        assert len(u) == nd, "Wrong array length: u"
    if v is not None:
        assert len(v) == nd, "Wrong array length: v"
    return True


def check_image_size(u, v, nxy, dxy, duv, PB=0, verbose=False):
    assert len(u) == len(v), "Wrong array length: u, v must have same length."
    uvdist = np.hypot(u, v)
    mrs = 0.6 / np.min(uvdist)
    max_uv = np.max(uvdist) * 2.0
    fov = nxy * dxy
    fov_to_mrs = fov / mrs
    uvfov_to_maxuv = 1.0 / (max_uv * dxy)
    fov_to_mrs_str = f"Nxy * dxy / MRS = {fov_to_mrs} must be > 1 at the very least"
    uvfov_to_maxuv_str = f"Nxy * duv / (2*max(u,v)) = {uvfov_to_maxuv} must be > 2 for Nyquist sampling"
    if PB != 0:
        fov_to_pb = fov / PB
        fov_to_pb_str = f"Nxy * dxy / PB = {fov_to_pb} must be > 1"
    if verbose:
        print(fov_to_mrs_str)
        print(uvfov_to_maxuv_str)
        if PB != 0:
            print(fov_to_pb_str)
    assert fov_to_mrs > 1, fov_to_mrs_str
    assert uvfov_to_maxuv > 2, uvfov_to_maxuv_str
    if PB != 0:
        assert fov_to_pb > 1, fov_to_pb_str
    assert np.max(np.abs(u) / duv <= nxy // 2 + 1)
    assert np.max(np.abs(v) / duv <= nxy // 2)
    return True


def get_image_size(u, v, PB=0, f_min=5.0, f_max=2.5, verbose=False):
    uvdist = np.hypot(u, v)
    mrs = 0.6 / np.min(uvdist)
    duv = 1.0 / mrs / f_min
    max_uv = np.max(uvdist) * 2.0 * f_max
    nxy = int(2 ** np.ceil(np.log2(max_uv / duv)))
    nxy_mrs = nxy
    dxy = 1.0 / (nxy * duv)
    if PB != 0:
        while dxy * nxy / PB < 1.0:
            nxy *= 2
    if verbose:
        print(f"dxy:{dxy / arcsec:e}arcsec\tnxy_MRS:{nxy_mrs}")
        print(f"nxy_MRS: matrix size to have FOV > f_min * MRS, where f_min:{f_min} and MRS:{mrs / arcsec:e}arcsec")
        if PB != 0:
            print(f"nxy_FOV:{nxy}")
            print("nxy_FOV: matrix size to have FOV > PB")
    return nxy, dxy


def estimate_fov_from_source(
    source_radius,
    offset=0.0,
    padding=1.25,
    min_fov=None,
    max_fov=None,
    primary_beam=None,
    primary_beam_level=None,
    verbose=False,
):
    """Estimate a square model image FOV from source extent and offset.

    Parameters are angular sizes in radians. ``source_radius`` should describe
    the effective model extent that must fit inside the image, while ``offset``
    is either a scalar radial offset or a ``(dRA, dDec)`` pair. ``padding`` is
    applied to the required image half-width.
    """
    source_radius = float(source_radius)
    padding = float(padding)

    if not np.isfinite(source_radius):
        raise ValueError("source_radius must be finite")
    if source_radius < 0:
        raise ValueError("source_radius must be non-negative")
    if not np.isfinite(padding):
        raise ValueError("padding must be finite")
    if padding < 1:
        raise ValueError("padding must be >= 1")

    offset_array = np.asarray(offset, dtype=np.float64)
    if offset_array.ndim == 0:
        offset_extent = abs(float(offset_array))
        if not np.isfinite(offset_extent):
            raise ValueError("offset must be finite")
    elif offset_array.shape == (2,):
        if not np.all(np.isfinite(offset_array)):
            raise ValueError("offset must contain finite values")
        offset_extent = float(np.max(np.abs(offset_array)))
    else:
        raise ValueError("offset must be a scalar or a (dRA, dDec) pair")

    half_width = padding * (source_radius + offset_extent)
    fov = 2.0 * half_width

    if primary_beam is not None:
        primary_beam = float(primary_beam)
        if not np.isfinite(primary_beam):
            raise ValueError("primary_beam must be finite")
        if primary_beam <= 0:
            raise ValueError("primary_beam must be positive")
        if primary_beam_level is None:
            pb_fov = primary_beam
        else:
            primary_beam_level = float(primary_beam_level)
            if not np.isfinite(primary_beam_level):
                raise ValueError("primary_beam_level must be finite")
            if not 0 < primary_beam_level < 1:
                raise ValueError("primary_beam_level must be between 0 and 1")
            pb_fov = primary_beam * np.sqrt(
                np.log(primary_beam_level) / np.log(0.5)
            )
        fov = max(fov, pb_fov)

    if min_fov is not None:
        min_fov = float(min_fov)
        if not np.isfinite(min_fov):
            raise ValueError("min_fov must be finite")
        if min_fov <= 0:
            raise ValueError("min_fov must be positive")
        fov = max(fov, min_fov)

    if max_fov is not None:
        max_fov = float(max_fov)
        if not np.isfinite(max_fov):
            raise ValueError("max_fov must be finite")
        if max_fov <= 0:
            raise ValueError("max_fov must be positive")
        if fov > max_fov:
            raise RuntimeError(
                f"Estimated fov={fov} exceeds max_fov={max_fov}. "
                "Reduce source extent, offset, or padding only if the model "
                "can safely fit inside the smaller field."
            )

    if verbose:
        print(f"source radius: {source_radius / arcsec:.6f} arcsec")
        print(f"offset extent: {offset_extent / arcsec:.6f} arcsec")
        print(f"padding: {padding:.6f}")
        print(f"estimated FOV: {fov / arcsec:.6f} arcsec")

    return fov


def get_image_size_from_fov(
    u,
    v,
    fov,
    pixels_per_fringe=3.0,
    max_nxy=None,
    verbose=False,
):
    """Determine an FFT image size from an explicitly chosen field of view.

    The image FOV is a model choice, not an automatic consequence of the
    shortest observed baseline. This helper therefore uses the requested FOV
    for the FFT box size and the longest observed baseline for the pixel size.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)

    if u.shape != v.shape:
        raise ValueError("u and v must have identical shapes")
    if fov <= 0:
        raise ValueError("fov must be positive")
    if pixels_per_fringe < 2:
        raise ValueError(
            "pixels_per_fringe must be >= 2 for Nyquist sampling"
        )

    uvdist = np.hypot(u, v)
    valid = np.isfinite(uvdist) & (uvdist > 0)
    if not np.any(valid):
        raise ValueError("No finite, positive uv distances were found")

    valid_uvdist = uvdist[valid]
    umax = np.max(valid_uvdist)
    finest_fringe = 1.0 / umax
    dxy_target = finest_fringe / pixels_per_fringe
    n_required = int(np.ceil(fov / dxy_target))
    nxy = 1 << max(0, n_required - 1).bit_length()

    if max_nxy is not None and nxy > max_nxy:
        raise RuntimeError(
            f"Required nxy={nxy} exceeds max_nxy={max_nxy}. "
            "Reduce the FOV or pixels_per_fringe only after checking the "
            "numerical consequences."
        )

    dxy = fov / nxy
    uv_nyquist = 1.0 / (2.0 * dxy)
    duv = 1.0 / fov
    actual_pixels_per_fringe = finest_fringe / dxy

    if uv_nyquist < umax:
        raise RuntimeError(
            "The selected image does not Nyquist-sample the uv data"
        )

    if verbose:
        umin = np.min(valid_uvdist)
        umin_p1 = np.percentile(valid_uvdist, 1.0)
        print(f"FOV: {fov / arcsec:.6f} arcsec")
        print(f"nxy: {nxy}")
        print(f"dxy: {dxy / arcsec:.6e} arcsec")
        print(f"umax: {umax / 1e6:.6f} Mlambda")
        print(f"FFT duv: {duv / 1e3:.6f} klambda")
        print(f"uv Nyquist limit: {uv_nyquist / 1e6:.6f} Mlambda")
        print(
            "actual pixels per finest fringe: "
            f"{actual_pixels_per_fringe:.3f}"
        )
        print(f"MRS from minimum uv distance: {0.6 / umin / arcsec:.6f} arcsec")
        print(f"MRS from 1st percentile uv distance: {0.6 / umin_p1 / arcsec:.6f} arcsec")

    return nxy, dxy


def prepare_component_array(array, width, name, dtype):
    if array is None:
        return np.empty((0, width), dtype=dtype)
    arr = np.ascontiguousarray(array, dtype=dtype)
    if arr.ndim != 2:
        raise ValueError(f"Expect {name} to be a 2D array.")
    if arr.shape[1] != width:
        raise ValueError(f"Expect {name} to have shape (n, {width}).")
    return arr


def prepare_component_batch_array(array, batch_size, dtype):
    if array is None:
        return np.empty((batch_size, 0), dtype=dtype)
    arr = np.ascontiguousarray(array, dtype=dtype)
    if arr.ndim != 2:
        raise ValueError("Expect component batch arrays to be 2D.")
    if arr.shape[0] != batch_size:
        raise ValueError(f"Expect component batch arrays to have batch dimension {batch_size}.")
    return arr


def infer_batch_size(*arrays):
    for array in arrays:
        if array is not None:
            arr = np.asarray(array)
            if arr.ndim > 0:
                return arr.shape[0]
    raise ValueError("Unable to infer batch size from empty batch inputs.")


def central_pixel(intensity, Rmin, dR, dxy):
    iin = int(np.floor((dxy / 2.0 - Rmin) // dR))
    flux = 0.0
    for i in range(1, iin):
        flux += (Rmin + dR * i) * intensity[i]
    flux *= 2.0
    flux += Rmin * intensity[0] + (Rmin + iin * dR) * intensity[iin]
    flux *= dR
    interp = (intensity[iin + 1] - intensity[iin]) / dR * (dxy / 2.0 - (Rmin + dR * iin)) + intensity[iin]
    flux += ((Rmin + iin * dR) * intensity[iin] + dxy / 2.0 * interp) * (dxy / 2.0 - (Rmin + iin * dR))
    area = (dxy / 2.0) ** 2 - Rmin ** 2
    return flux / area


def uv_idx_r2c(udat, vdat, du, half_size):
    indu = np.abs(udat) / du
    indv = half_size + vdat / du
    uneg = udat < 0
    indv[uneg] = half_size - vdat[uneg] / du
    return indu, indv


def int_bilin(f, x, y):
    vis_int = np.zeros(len(x), dtype=f.dtype)
    for i in range(len(x)):
        t = y[i] - np.floor(y[i])
        u = x[i] - np.floor(x[i])
        yi = int(np.floor(y[i]))
        xi = int(np.floor(x[i]))
        y0 = f[yi, xi]
        y1 = f[yi + 1, xi]
        y2 = f[yi + 1, xi + 1]
        y3 = f[yi, xi + 1]
        vis_int[i] = t * u * (y0 - y1 + y2 - y3)
        vis_int[i] += t * (y1 - y0)
        vis_int[i] += u * (y3 - y0)
        vis_int[i] += y0
    return vis_int
