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
