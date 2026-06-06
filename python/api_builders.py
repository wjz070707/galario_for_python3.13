from __future__ import annotations

import numpy as np

from .api_constants import arcsec, au, cgs_to_Jy, deg, pc
from .api_utils import (
    central_pixel,
    check_image_size,
    check_obs,
    get_image_size,
    infer_batch_size,
    int_bilin,
    prepare_component_array,
    prepare_component_batch_array,
    set_v_origin,
    uv_idx_r2c,
)

# Python policy layer, not a numerical backend. It normalizes NumPy inputs,
# assembles the same API around CPU or CUDA bindings, and preserves selected
# legacy camelCase entry points. Expensive loops should stay in C++/CUDA.

def build_api(raw_module, module_name: str, real_dtype, complex_dtype):
    init_fn = getattr(raw_module, "_init", None)
    cleanup_fn = getattr(raw_module, "_cleanup", None)
    if callable(init_fn):
        init_fn()

    def _component_dtype():
        return real_dtype

    def create_image_context(nx, ny, u, v, vis_obs_re, vis_obs_im, weights, backend=raw_module.BACKEND_AUTO, nufft_oversample=2.0):
        return raw_module._create_image_context(
            int(nx), int(ny),
            np.ascontiguousarray(u, dtype=real_dtype),
            np.ascontiguousarray(v, dtype=real_dtype),
            np.ascontiguousarray(vis_obs_re, dtype=real_dtype),
            np.ascontiguousarray(vis_obs_im, dtype=real_dtype),
            np.ascontiguousarray(weights, dtype=real_dtype),
            backend=backend, nufft_oversample=nufft_oversample,
        )

    def sample_profile(intensity, Rmin, dR, nxy, dxy, u, v, dRA=0.0, dDec=0.0, PA=0.0, inc=0.0,
                       backend=raw_module.BACKEND_AUTO, nufft_oversample=2.0):
        return np.asarray(
            raw_module.sampleProfile(
                np.ascontiguousarray(intensity, dtype=real_dtype),
                Rmin, dR, int(nxy), dxy,
                np.ascontiguousarray(u, dtype=real_dtype),
                np.ascontiguousarray(v, dtype=real_dtype),
                dRA=dRA, dDec=dDec, PA=PA, inc=inc,
                backend=backend, nufft_oversample=nufft_oversample,
            ),
            dtype=complex_dtype,
        )

    def chi2_profile(intensity, Rmin, dR, nxy, dxy, u=None, v=None,
                     vis_obs_re=None, vis_obs_im=None, weights=None,
                     vis_obs_w=None, dRA=0.0, dDec=0.0, PA=0.0, inc=0.0,
                     backend=raw_module.BACKEND_AUTO, nufft_oversample=2.0,
                     ctx=None):
        if ctx is not None:
            return raw_module._chi2_profile_from_context(
                ctx,
                np.ascontiguousarray(intensity, dtype=real_dtype),
                Rmin, dR, int(nxy), dxy,
                dRA=dRA, dDec=dDec, PA=PA, inc=inc,
            )
        if weights is None:
            weights = vis_obs_w
        if weights is None:
            raise ValueError("weights must be provided.")
        return raw_module.chi2Profile(
            np.ascontiguousarray(intensity, dtype=real_dtype),
            Rmin, dR, int(nxy), dxy,
            np.ascontiguousarray(u, dtype=real_dtype),
            np.ascontiguousarray(v, dtype=real_dtype),
            np.ascontiguousarray(vis_obs_re, dtype=real_dtype),
            np.ascontiguousarray(vis_obs_im, dtype=real_dtype),
            np.ascontiguousarray(weights, dtype=real_dtype),
            dRA=dRA, dDec=dDec, PA=PA, inc=inc,
            backend=backend, nufft_oversample=nufft_oversample,
        )

    def sample_image(*, dxy, u, v, image=None, nx=None, ny=None, gauss_params=None, ring_params=None, arc_params=None,
                     inc=0.0, dRA=0.0, dDec=0.0, PA=0.0, origin="upper", backend=raw_module.BACKEND_AUTO, nufft_oversample=2.0):
        if image is not None:
            return np.asarray(
                raw_module.sampleImage(
                    np.ascontiguousarray(image, dtype=real_dtype),
                    dxy,
                    np.ascontiguousarray(u, dtype=real_dtype),
                    np.ascontiguousarray(v, dtype=real_dtype),
                    dRA=dRA, dDec=dDec, PA=PA, origin=origin,
                    backend=backend, nufft_oversample=nufft_oversample,
                ),
                dtype=complex_dtype,
            )
        if nx is None or ny is None:
            raise ValueError("nx and ny must be provided when sampling component models without an image.")
        return np.asarray(
            raw_module.sampleImageComponents(
                int(nx), int(ny), dxy,
                np.ascontiguousarray(u, dtype=real_dtype),
                np.ascontiguousarray(v, dtype=real_dtype),
                prepare_component_array(gauss_params, 2, "gauss_params", real_dtype),
                prepare_component_array(ring_params, 3, "ring_params", real_dtype),
                prepare_component_array(arc_params, 5, "arc_params", real_dtype),
                inc=inc, dRA=dRA, dDec=dDec, PA=PA, origin=origin,
                backend=backend, nufft_oversample=nufft_oversample,
            ),
            dtype=complex_dtype,
        )

    def chi2_image(*, dxy, image=None, ctx=None, nx=None, ny=None, u=None, v=None, vis_obs_re=None, vis_obs_im=None,
                   weights=None, vis_obs_w=None, gauss_params=None, ring_params=None, arc_params=None,
                   inc=0.0, dRA=0.0, dDec=0.0, PA=0.0,
                   gauss_params_batch=None, ring_params_batch=None, arc_params_batch=None,
                   inc_batch=None, dRA_batch=None, dDec_batch=None, PA_batch=None,
                   origin="upper", backend=raw_module.BACKEND_AUTO, nufft_oversample=2.0):
        if weights is None:
            weights = vis_obs_w
        if ctx is not None:
            # Preferred optimizer/MCMC path: the context reuses observations,
            # backend work buffers, and transform plans.
            if image is not None:
                return raw_module._chi2_image_from_context(
                    ctx, np.ascontiguousarray(image, dtype=real_dtype), dxy,
                    dRA=dRA, dDec=dDec, PA=PA, origin=origin,
                )
            batch_request = any(value is not None for value in (gauss_params_batch, ring_params_batch, arc_params_batch, inc_batch, dRA_batch, dDec_batch, PA_batch))
            if batch_request:
                batch_size = infer_batch_size(inc_batch, dRA_batch, dDec_batch, PA_batch, gauss_params_batch, ring_params_batch, arc_params_batch)
                inc_arr = np.zeros(batch_size, dtype=real_dtype) if inc_batch is None else np.ascontiguousarray(inc_batch, dtype=real_dtype)
                dra_arr = np.zeros(batch_size, dtype=real_dtype) if dRA_batch is None else np.ascontiguousarray(dRA_batch, dtype=real_dtype)
                ddec_arr = np.zeros(batch_size, dtype=real_dtype) if dDec_batch is None else np.ascontiguousarray(dDec_batch, dtype=real_dtype)
                pa_arr = np.zeros(batch_size, dtype=real_dtype) if PA_batch is None else np.ascontiguousarray(PA_batch, dtype=real_dtype)
                return np.asarray(
                    raw_module._chi2_image_from_context_components_batch(
                        ctx, dxy,
                        prepare_component_batch_array(gauss_params_batch, batch_size, real_dtype),
                        prepare_component_batch_array(ring_params_batch, batch_size, real_dtype),
                        prepare_component_batch_array(arc_params_batch, batch_size, real_dtype),
                        inc_arr, dra_arr, ddec_arr, pa_arr, origin=origin,
                    ),
                    dtype=real_dtype,
                )
            return raw_module._chi2_image_from_context_components(
                ctx, dxy,
                prepare_component_array(gauss_params, 2, "gauss_params", real_dtype),
                prepare_component_array(ring_params, 3, "ring_params", real_dtype),
                prepare_component_array(arc_params, 5, "arc_params", real_dtype),
                inc=inc, dRA=dRA, dDec=dDec, PA=PA, origin=origin,
            )
        if image is not None:
            if weights is None:
                raise ValueError("weights must be provided for direct chi2_image calls.")
            return raw_module.chi2Image(
                np.ascontiguousarray(image, dtype=real_dtype),
                dxy,
                np.ascontiguousarray(u, dtype=real_dtype),
                np.ascontiguousarray(v, dtype=real_dtype),
                np.ascontiguousarray(vis_obs_re, dtype=real_dtype),
                np.ascontiguousarray(vis_obs_im, dtype=real_dtype),
                np.ascontiguousarray(weights, dtype=real_dtype),
                dRA=dRA, dDec=dDec, PA=PA, origin=origin,
                backend=backend, nufft_oversample=nufft_oversample,
            )
        if weights is None:
            raise ValueError("weights must be provided.")
        if nx is None or ny is None:
            raise ValueError("nx and ny must be provided when evaluating component models without a cached context.")
        return raw_module.chi2ImageComponents(
            int(nx), int(ny), dxy,
            np.ascontiguousarray(u, dtype=real_dtype),
            np.ascontiguousarray(v, dtype=real_dtype),
            np.ascontiguousarray(vis_obs_re, dtype=real_dtype),
            np.ascontiguousarray(vis_obs_im, dtype=real_dtype),
            np.ascontiguousarray(weights, dtype=real_dtype),
            prepare_component_array(gauss_params, 2, "gauss_params", real_dtype),
            prepare_component_array(ring_params, 3, "ring_params", real_dtype),
            prepare_component_array(arc_params, 5, "arc_params", real_dtype),
            inc=inc, dRA=dRA, dDec=dDec, PA=PA, origin=origin,
            backend=backend, nufft_oversample=nufft_oversample,
        )

    def get_coords_meshgrid(nrow, ncol, dxy=1.0, inc=0.0, Dx=0.0, Dy=0.0, origin="upper"):
        v_origin = set_v_origin(origin)
        x = (np.linspace(0.5, -0.5 + 1.0 / float(ncol), ncol, dtype=real_dtype)) * ncol * dxy
        y = (np.linspace(0.5, -0.5 + 1.0 / float(nrow), nrow, dtype=real_dtype)) * nrow * dxy * v_origin
        x_m, y_m = np.meshgrid((x - Dx) / np.cos(inc), (y - Dy))
        R_m = np.hypot(x_m, y_m)
        return x, y, x_m.astype(real_dtype), y_m.astype(real_dtype), R_m.astype(real_dtype)

    def sweep(intensity, Rmin, dR, nxy, dxy, inc=0.0):
        I = np.ascontiguousarray(intensity, dtype=real_dtype)
        image = np.zeros((int(nxy), int(nxy)), dtype=real_dtype)
        nrad = len(I)
        irow_center = int(nxy) // 2
        icol_center = int(nxy) // 2
        inc_cos = np.cos(inc)
        rmax = min(int(np.ceil((Rmin + nrad * dR) / dxy)), irow_center)
        row_offset = irow_center - rmax
        col_offset = icol_center - rmax
        for irow in range(rmax * 2):
            for jcol in range(rmax * 2):
                x = (rmax - jcol) * dxy
                y = (rmax - irow) * dxy
                rr = np.sqrt((x / inc_cos) ** 2.0 + y ** 2.0)
                iR = int(np.floor((rr - Rmin) / dR))
                if iR >= nrad - 1:
                    image[irow + row_offset, jcol + col_offset] = 0.0
                else:
                    image[irow + row_offset, jcol + col_offset] = I[iR] + (rr - iR * dR - Rmin) * (I[iR + 1] - I[iR]) / dR
        image[irow_center, icol_center] = central_pixel(I, Rmin, dR, dxy)
        image *= dxy ** 2
        return np.ascontiguousarray(image)

    def uv_rotate(PA, dRA, dDec, u, v):
        u_arr = np.ascontiguousarray(u, dtype=real_dtype)
        v_arr = np.ascontiguousarray(v, dtype=real_dtype)
        cos_pa = np.cos(PA)
        sin_pa = np.sin(PA)
        urot = u_arr * cos_pa - v_arr * sin_pa
        vrot = u_arr * sin_pa + v_arr * cos_pa
        dRArot = dRA * cos_pa - dDec * sin_pa
        dDecrot = dRA * sin_pa + dDec * cos_pa
        return real_dtype(dRArot), real_dtype(dDecrot), urot.astype(real_dtype), vrot.astype(real_dtype)

    def interpolate(r2cFT, duv, u, v, origin="upper"):
        ft = np.ascontiguousarray(r2cFT, dtype=complex_dtype)
        u_arr = np.ascontiguousarray(u, dtype=real_dtype)
        v_arr = np.ascontiguousarray(v, dtype=real_dtype)
        uroti, vroti = uv_idx_r2c(u_arr.astype(np.float64), v_arr.astype(np.float64), float(duv), ft.shape[0] / 2.0)
        re_int = int_bilin(ft.real.astype(np.float64), uroti, vroti)
        im_int = int_bilin(ft.imag.astype(np.float64), uroti, vroti)
        amp_int = int_bilin(np.abs(ft).astype(np.float64), uroti, vroti)
        uneg = u_arr < 0
        im_int[uneg] *= -1
        phase_int = np.angle(re_int + 1j * im_int)
        vis = amp_int * (np.cos(phase_int) + 1j * np.sin(phase_int))
        return np.asarray(vis, dtype=complex_dtype)

    def apply_phase_vis(dRA, dDec, u, v, vis):
        u_arr = np.ascontiguousarray(u, dtype=real_dtype)
        v_arr = np.ascontiguousarray(v, dtype=real_dtype)
        vis_arr = np.ascontiguousarray(vis, dtype=complex_dtype)
        theta = u_arr * (2.0 * np.pi * dRA) + v_arr * (2.0 * np.pi * dDec)
        return np.asarray(vis_arr * (np.cos(theta) + 1j * np.sin(theta)), dtype=complex_dtype)

    def reduce_chi2(vis_obs_re, vis_obs_im, vis_obs_w, vis):
        vis_arr = np.ascontiguousarray(vis, dtype=complex_dtype)
        re_arr = np.ascontiguousarray(vis_obs_re, dtype=real_dtype)
        im_arr = np.ascontiguousarray(vis_obs_im, dtype=real_dtype)
        w_arr = np.ascontiguousarray(vis_obs_w, dtype=real_dtype)
        check_obs(re_arr, im_arr, w_arr, vis=vis_arr)
        return np.sum(((vis_arr.real - re_arr) ** 2.0 + (vis_arr.imag - im_arr) ** 2.0) * w_arr, dtype=np.float64).item()

    def _fft2d(image):
        image_arr = np.ascontiguousarray(image, dtype=real_dtype)
        if image_arr.ndim != 2:
            raise ValueError("Expect a 2D image")
        if image_arr.shape[0] != image_arr.shape[1]:
            raise ValueError(f"Expect a square image but got shape {image_arr.shape}")
        if image_arr.shape[0] < 2:
            raise ValueError(f"x dimension = {image_arr.shape[0]} is less than 2")
        if image_arr.shape[0] % 2 != 0:
            raise ValueError(f"x dimension = {image_arr.shape[0]} is odd")
        return np.ascontiguousarray(np.fft.rfft2(image_arr), dtype=complex_dtype)

    def _fftshift(matrix):
        matrix_arr = np.ascontiguousarray(matrix, dtype=real_dtype)
        nx, ny = matrix_arr.shape
        if nx % 2 != 0 or ny % 2 != 0:
            raise ValueError(f"Expect even matrix size but got {matrix_arr.shape}")
        shifted = np.fft.fftshift(matrix_arr)
        out = np.zeros((nx, ny // 2 + 1), dtype=complex_dtype)
        out.view(dtype=real_dtype)[:, :ny] = shifted
        return out

    def _fftshift_axis0(matrix):
        matrix[...] = np.fft.fftshift(np.asarray(matrix), axes=0)

    def sampleImage(image, dxy, u, v, dRA=0.0, dDec=0.0, PA=0.0, check=False, origin="upper", backend=raw_module.BACKEND_AUTO, nufft_oversample=2.0):
        if check:
            duv = 1.0 / (image.shape[0] * dxy)
            check_image_size(u, v, image.shape[0], dxy, duv)
        return sample_image(image=image, dxy=dxy, u=u, v=v, dRA=dRA, dDec=dDec, PA=PA, origin=origin, backend=backend, nufft_oversample=nufft_oversample)

    def sampleImageComponents(nx, ny, dxy, u, v, gauss_params=None, ring_params=None, arc_params=None,
                              inc=0.0, dRA=0.0, dDec=0.0, PA=0.0, origin="upper", backend=raw_module.BACKEND_AUTO, nufft_oversample=2.0):
        return sample_image(nx=nx, ny=ny, dxy=dxy, u=u, v=v, gauss_params=gauss_params, ring_params=ring_params, arc_params=arc_params, inc=inc, dRA=dRA, dDec=dDec, PA=PA, origin=origin, backend=backend, nufft_oversample=nufft_oversample)

    def sampleProfile(intensity, Rmin, dR, nxy, dxy, u, v, dRA=0.0, dDec=0.0, PA=0.0, inc=0.0, check=False, backend=raw_module.BACKEND_AUTO, nufft_oversample=2.0):
        return sample_profile(intensity, Rmin, dR, nxy, dxy, u, v, dRA=dRA, dDec=dDec, PA=PA, inc=inc, backend=backend, nufft_oversample=nufft_oversample)

    exports = {
        "arcsec": arcsec,
        "deg": deg,
        "cgs_to_Jy": cgs_to_Jy,
        "pc": pc,
        "au": au,
        "BACKEND_AUTO": raw_module.BACKEND_AUTO,
        "BACKEND_DFT": raw_module.BACKEND_DFT,
        "BACKEND_FFT": raw_module.BACKEND_FFT,
        "BACKEND_NUFFT": raw_module.BACKEND_NUFFT,
        "_init": init_fn,
        "_cleanup": cleanup_fn,
        "threads": getattr(raw_module, "threads"),
        "ngpus": getattr(raw_module, "ngpus"),
        "use_gpu": getattr(raw_module, "use_gpu"),
        "check_obs": check_obs,
        "check_image_size": check_image_size,
        "get_image_size": get_image_size,
        "set_v_origin": set_v_origin,
        "_fft2d": _fft2d,
        "_fftshift": _fftshift,
        "_fftshift_axis0": _fftshift_axis0,
        "_component_dtype": _component_dtype,
        "create_image_context": create_image_context,
        "sample_image": sample_image,
        "chi2_image": chi2_image,
        "sample_profile": sample_profile,
        "chi2_profile": chi2_profile,
        "get_coords_meshgrid": get_coords_meshgrid,
        "sweep": sweep,
        "uv_rotate": uv_rotate,
        "interpolate": interpolate,
        "apply_phase_vis": apply_phase_vis,
        "reduce_chi2": reduce_chi2,
        "sampleImage": sampleImage,
        "sampleImageComponents": sampleImageComponents,
        "sampleProfile": sampleProfile,
        "chi2Image": raw_module.chi2Image,
        "chi2ImageComponents": raw_module.chi2ImageComponents,
        "chi2Profile": raw_module.chi2Profile,
        "Chi2ImageContext": getattr(raw_module, "Chi2ImageContext", None),
    }

    exports["__all__"] = [name for name, value in exports.items() if not name.startswith("_") and value is not None]
    for name, value in list(exports.items()):
        if value is None:
            del exports[name]
            continue
        if callable(value):
            try:
                value.__module__ = module_name
            except (AttributeError, TypeError):
                pass
    return exports
