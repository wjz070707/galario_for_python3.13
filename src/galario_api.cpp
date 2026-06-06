#include "galario.h"
#include "galario_internal.h"
#include "galario_profile_common.h"
#include "galario_py.h"

#include <cstddef>
#include <vector>

/*
 * Backend-neutral orchestration layer.
 *
 * Public operations live here when they can be expressed by composing common
 * model code with the private backend ABI from galario_internal.h. Keep CPU
 * details in galario_cpu.cpp and CUDA details in galario_gpu.cu.
 *
 * Functions prefixed with '_' are pointer-erased bridges for the Python
 * binding. They deliberately contain no numerical implementation.
 */
namespace galario {

dreal chi2_image(
    int nx,
    int ny,
    const dreal* realdata,
    const dreal v_origin,
    dreal dRA,
    dreal dDec,
    dreal duv,
    dreal PA,
    int nd,
    const dreal* u,
    const dreal* v,
    const dreal* vis_obs_re,
    const dreal* vis_obs_im,
    const dreal* weights,
    int backend,
    dreal nufft_oversample
) {
    std::vector<dcomplex> vis_int(nd);
    _sample_image(
        nx,
        ny,
        const_cast<dreal*>(realdata),
        v_origin,
        dRA,
        dDec,
        duv,
        PA,
        nd,
        const_cast<dreal*>(u),
        const_cast<dreal*>(v),
        static_cast<void*>(vis_int.data()),
        backend,
        nufft_oversample
    );
    return _reduce_chi2(
        nd,
        const_cast<dreal*>(vis_obs_re),
        const_cast<dreal*>(vis_obs_im),
        static_cast<void*>(vis_int.data()),
        const_cast<dreal*>(weights)
    );
}

dreal _chi2_image(
    int nx,
    int ny,
    void* realdata,
    const dreal v_origin,
    dreal dRA,
    dreal dDec,
    dreal duv,
    dreal PA,
    int nd,
    void* u,
    void* v,
    void* vis_obs_re,
    void* vis_obs_im,
    void* weights,
    int backend,
    dreal nufft_oversample
) {
    return chi2_image(
        nx,
        ny,
        static_cast<dreal*>(realdata),
        v_origin,
        dRA,
        dDec,
        duv,
        PA,
        nd,
        static_cast<dreal*>(u),
        static_cast<dreal*>(v),
        static_cast<dreal*>(vis_obs_re),
        static_cast<dreal*>(vis_obs_im),
        static_cast<dreal*>(weights),
        backend,
        nufft_oversample
    );
}

void sample_image_components(
    int nx,
    int ny,
    dreal dxy,
    int ngauss,
    const dreal* gauss_params,
    int nrings,
    const dreal* ring_params,
    int narcs,
    const dreal* arc_params,
    dreal inc,
    const dreal v_origin,
    dreal dRA,
    dreal dDec,
    dreal duv,
    dreal PA,
    int nd,
    const dreal* u,
    const dreal* v,
    dcomplex* vis_int,
    int backend,
    dreal nufft_oversample
) {
    std::vector<dreal> image(static_cast<size_t>(nx) * ny);
    galario_internal::rasterize_component_image(
        nx,
        ny,
        dxy,
        ngauss,
        gauss_params,
        nrings,
        ring_params,
        narcs,
        arc_params,
        inc,
        nx,
        ny,
        image.data()
    );
    _sample_image(
        nx,
        ny,
        static_cast<void*>(image.data()),
        v_origin,
        dRA,
        dDec,
        duv,
        PA,
        nd,
        const_cast<dreal*>(u),
        const_cast<dreal*>(v),
        static_cast<void*>(vis_int),
        backend,
        nufft_oversample
    );
}

void _sample_image_components(
    int nx,
    int ny,
    dreal dxy,
    int ngauss,
    void* gauss_params,
    int nrings,
    void* ring_params,
    int narcs,
    void* arc_params,
    dreal inc,
    dreal v_origin,
    dreal dRA,
    dreal dDec,
    dreal duv,
    dreal PA,
    int nd,
    void* u,
    void* v,
    void* vis_int,
    int backend,
    dreal nufft_oversample
) {
    sample_image_components(
        nx,
        ny,
        dxy,
        ngauss,
        static_cast<dreal*>(gauss_params),
        nrings,
        static_cast<dreal*>(ring_params),
        narcs,
        static_cast<dreal*>(arc_params),
        inc,
        v_origin,
        dRA,
        dDec,
        duv,
        PA,
        nd,
        static_cast<dreal*>(u),
        static_cast<dreal*>(v),
        static_cast<dcomplex*>(vis_int),
        backend,
        nufft_oversample
    );
}

dreal chi2_image_components(
    int nx,
    int ny,
    dreal dxy,
    int ngauss,
    const dreal* gauss_params,
    int nrings,
    const dreal* ring_params,
    int narcs,
    const dreal* arc_params,
    dreal inc,
    const dreal v_origin,
    dreal dRA,
    dreal dDec,
    dreal duv,
    dreal PA,
    int nd,
    const dreal* u,
    const dreal* v,
    const dreal* vis_obs_re,
    const dreal* vis_obs_im,
    const dreal* weights,
    int backend,
    dreal nufft_oversample
) {
    std::vector<dreal> image(static_cast<size_t>(nx) * ny);
    galario_internal::rasterize_component_image(
        nx,
        ny,
        dxy,
        ngauss,
        gauss_params,
        nrings,
        ring_params,
        narcs,
        arc_params,
        inc,
        nx,
        ny,
        image.data()
    );
    return chi2_image(
        nx,
        ny,
        image.data(),
        v_origin,
        dRA,
        dDec,
        duv,
        PA,
        nd,
        u,
        v,
        vis_obs_re,
        vis_obs_im,
        weights,
        backend,
        nufft_oversample
    );
}

dreal _chi2_image_components(
    int nx,
    int ny,
    dreal dxy,
    int ngauss,
    void* gauss_params,
    int nrings,
    void* ring_params,
    int narcs,
    void* arc_params,
    dreal inc,
    dreal v_origin,
    dreal dRA,
    dreal dDec,
    dreal duv,
    dreal PA,
    int nd,
    void* u,
    void* v,
    void* vis_obs_re,
    void* vis_obs_im,
    void* weights,
    int backend,
    dreal nufft_oversample
) {
    return chi2_image_components(
        nx,
        ny,
        dxy,
        ngauss,
        static_cast<dreal*>(gauss_params),
        nrings,
        static_cast<dreal*>(ring_params),
        narcs,
        static_cast<dreal*>(arc_params),
        inc,
        v_origin,
        dRA,
        dDec,
        duv,
        PA,
        nd,
        static_cast<dreal*>(u),
        static_cast<dreal*>(v),
        static_cast<dreal*>(vis_obs_re),
        static_cast<dreal*>(vis_obs_im),
        static_cast<dreal*>(weights),
        backend,
        nufft_oversample
    );
}

Chi2ImageContext* create_image_context(
    int nx,
    int ny,
    int nd,
    const dreal* u,
    const dreal* v,
    const dreal* vis_obs_re,
    const dreal* vis_obs_im,
    const dreal* weights,
    int backend,
    dreal nufft_oversample
) {
    // Contexts retain backend work buffers and plans. Reuse one when u/v,
    // observations, and weights stay fixed across likelihood evaluations.
    return galario_internal::create_image_context_impl(
        nx, ny, nd, u, v, vis_obs_re, vis_obs_im, weights,
        backend, nufft_oversample
    );
}

void destroy_image_context(Chi2ImageContext* context) {
    galario_internal::destroy_image_context_impl(context);
}

int image_context_requested_backend(const Chi2ImageContext* context) {
    return galario_internal::image_context_requested_backend_impl(context);
}

int image_context_backend(const Chi2ImageContext* context) {
    return galario_internal::image_context_backend_impl(context);
}

int image_context_batch_backend(
    const Chi2ImageContext* context,
    int batch_size
) {
    return galario_internal::image_context_batch_backend_impl(
        context, batch_size
    );
}

void* _create_image_context(
    int nx,
    int ny,
    int nd,
    void* u,
    void* v,
    void* vis_obs_re,
    void* vis_obs_im,
    void* weights,
    int backend,
    dreal nufft_oversample
) {
    return create_image_context(
        nx, ny, nd,
        static_cast<dreal*>(u),
        static_cast<dreal*>(v),
        static_cast<dreal*>(vis_obs_re),
        static_cast<dreal*>(vis_obs_im),
        static_cast<dreal*>(weights),
        backend, nufft_oversample
    );
}

void _destroy_image_context(void* context) {
    destroy_image_context(static_cast<Chi2ImageContext*>(context));
}

int _image_context_requested_backend(void* context) {
    return image_context_requested_backend(
        static_cast<Chi2ImageContext*>(context)
    );
}

int _image_context_backend(void* context) {
    return image_context_backend(static_cast<Chi2ImageContext*>(context));
}

int _image_context_batch_backend(void* context, int batch_size) {
    return image_context_batch_backend(
        static_cast<Chi2ImageContext*>(context), batch_size
    );
}

dreal _chi2_image_from_context(
    void* context,
    void* realdata,
    const dreal v_origin,
    dreal dRA,
    dreal dDec,
    dreal duv,
    dreal PA
) {
    return chi2_image_from_context(
        static_cast<Chi2ImageContext*>(context),
        static_cast<dreal*>(realdata),
        v_origin, dRA, dDec, duv, PA
    );
}

dreal _chi2_image_from_context_components(
    void* context,
    dreal dxy,
    int ngauss,
    void* gauss_params,
    int nrings,
    void* ring_params,
    int narcs,
    void* arc_params,
    dreal inc,
    const dreal v_origin,
    dreal dRA,
    dreal dDec,
    dreal duv,
    dreal PA
) {
    return chi2_image_from_context_components(
        static_cast<Chi2ImageContext*>(context),
        dxy,
        ngauss, static_cast<dreal*>(gauss_params),
        nrings, static_cast<dreal*>(ring_params),
        narcs, static_cast<dreal*>(arc_params),
        inc, v_origin, dRA, dDec, duv, PA
    );
}

void _chi2_image_from_context_components_batch(
    void* context,
    dreal dxy,
    int batch_size,
    int ngauss,
    void* gauss_params_batch,
    int nrings,
    void* ring_params_batch,
    int narcs,
    void* arc_params_batch,
    void* inc_batch,
    dreal v_origin,
    void* dRA_batch,
    void* dDec_batch,
    dreal duv,
    void* PA_batch,
    void* chi2_out
) {
    chi2_image_from_context_components_batch(
        static_cast<Chi2ImageContext*>(context),
        dxy, batch_size,
        ngauss, static_cast<dreal*>(gauss_params_batch),
        nrings, static_cast<dreal*>(ring_params_batch),
        narcs, static_cast<dreal*>(arc_params_batch),
        static_cast<dreal*>(inc_batch),
        v_origin,
        static_cast<dreal*>(dRA_batch),
        static_cast<dreal*>(dDec_batch),
        duv,
        static_cast<dreal*>(PA_batch),
        static_cast<dreal*>(chi2_out)
    );
}

void fft2d(int nx, int ny, dcomplex* data) {
    galario_internal::fft2d_impl(nx, ny, data);
}

void _fft2d(int nx, int ny, void* data) {
    fft2d(nx, ny, static_cast<dcomplex*>(data));
}

void fftshift(int nx, int ny, dcomplex* data) {
    galario_internal::fftshift_impl(nx, ny, data);
}

void _fftshift(int nx, int ny, void* data) {
    fftshift(nx, ny, static_cast<dcomplex*>(data));
}

void fftshift_axis0(int nrow, int ncol, dcomplex* matrix) {
    galario_internal::fftshift_axis0_impl(nrow, ncol, matrix);
}

void _fftshift_axis0(int nrow, int ncol, void* matrix) {
    fftshift_axis0(nrow, ncol, static_cast<dcomplex*>(matrix));
}

void interpolate(
    int nrow,
    int ncol,
    const dcomplex* data,
    dreal v_origin,
    int nd,
    const dreal* u,
    const dreal* v,
    dreal duv,
    dcomplex* vis_int
) {
    galario_internal::interpolate_impl(
        nrow, ncol, data, v_origin, nd, u, v, duv, vis_int
    );
}

void _interpolate(
    int nrow,
    int ncol,
    void* data,
    dreal v_origin,
    int nd,
    void* u,
    void* v,
    dreal duv,
    void* vis_int
) {
    interpolate(
        nrow, ncol, static_cast<dcomplex*>(data), v_origin, nd,
        static_cast<dreal*>(u), static_cast<dreal*>(v), duv,
        static_cast<dcomplex*>(vis_int)
    );
}

void sweep(
    int nr,
    const dreal* intensity,
    dreal r_min,
    dreal dr,
    int nxy,
    dreal dxy,
    dreal inc,
    dcomplex* image
) {
    galario_profile_detail::sweep_image(
        nr, intensity, r_min, dr, nxy, dxy, inc, image
    );
}

void _sweep(
    int nr,
    void* intensity,
    dreal r_min,
    dreal dr,
    int nxy,
    dreal dxy,
    dreal inc,
    void* image
) {
    sweep(
        nr, static_cast<dreal*>(intensity), r_min, dr,
        nxy, dxy, inc, static_cast<dcomplex*>(image)
    );
}

void sample_profile(
    int nr,
    const dreal* intensity,
    dreal r_min,
    dreal dr,
    dreal dxy,
    int nxy,
    dreal inc,
    dreal dra,
    dreal ddec,
    dreal duv,
    dreal pa,
    int nd,
    const dreal* u,
    const dreal* v,
    dcomplex* vis,
    int backend,
    dreal nufft_oversample
) {
    // Profiles have an efficient direct path. FFT/NUFFT requests first
    // rasterize the profile and then use the regular image pipeline.
    int const resolved_backend =
        resolve_profile_backend(nr, nd, backend);
    if (resolved_backend == BACKEND_DFT) {
        galario_profile_detail::sample_direct(
            nr, intensity, r_min, dr, inc, dra, ddec, pa,
            nd, u, v, vis
        );
        return;
    }

    std::vector<dcomplex> image(
        static_cast<size_t>(nxy) * (nxy / 2 + 1)
    );
    sweep(nr, intensity, r_min, dr, nxy, dxy, inc, image.data());
    std::vector<dreal> real_image =
        galario_profile_detail::unpack_image(nxy, image.data());
    _sample_image(
        nxy, nxy, real_image.data(), 1.0,
        dra, ddec, duv, pa, nd,
        const_cast<dreal*>(u),
        const_cast<dreal*>(v),
        vis,
        resolved_backend, nufft_oversample
    );
}

void _sample_profile(
    int nr,
    void* intensity,
    dreal r_min,
    dreal dr,
    dreal dxy,
    int nxy,
    dreal inc,
    dreal dra,
    dreal ddec,
    dreal duv,
    dreal pa,
    int nd,
    void* u,
    void* v,
    void* vis,
    int backend,
    dreal nufft_oversample
) {
    sample_profile(
        nr, static_cast<dreal*>(intensity), r_min, dr, dxy, nxy,
        inc, dra, ddec, duv, pa, nd,
        static_cast<dreal*>(u), static_cast<dreal*>(v),
        static_cast<dcomplex*>(vis), backend, nufft_oversample
    );
}

dreal chi2_profile(
    int nr,
    const dreal* intensity,
    dreal r_min,
    dreal dr,
    dreal dxy,
    int nxy,
    dreal inc,
    dreal dra,
    dreal ddec,
    dreal duv,
    dreal pa,
    int nd,
    const dreal* u,
    const dreal* v,
    const dreal* vis_obs_re,
    const dreal* vis_obs_im,
    const dreal* weights,
    int backend,
    dreal nufft_oversample
) {
    std::vector<dcomplex> vis(nd);
    sample_profile(
        nr, intensity, r_min, dr, dxy, nxy, inc,
        dra, ddec, duv, pa, nd, u, v, vis.data(),
        backend, nufft_oversample
    );
    return _reduce_chi2(
        nd,
        const_cast<dreal*>(vis_obs_re),
        const_cast<dreal*>(vis_obs_im),
        vis.data(),
        const_cast<dreal*>(weights)
    );
}

dreal chi2_profile_from_context(
    Chi2ImageContext* context,
    int nr,
    const dreal* intensity,
    dreal r_min,
    dreal dr,
    int nxy,
    dreal dxy,
    dreal inc,
    dreal dra,
    dreal ddec,
    dreal duv,
    dreal pa
) {
    // A profile context uses the FFT image path: rasterize the radial model,
    // then reuse the context's observations, transform plan, and work buffers.
    std::vector<dcomplex> packed_image(
        static_cast<size_t>(nxy) * (nxy / 2 + 1)
    );
    sweep(
        nr, intensity, r_min, dr, nxy, dxy, inc,
        packed_image.data()
    );
    std::vector<dreal> image =
        galario_profile_detail::unpack_image(nxy, packed_image.data());
    return chi2_image_from_context(
        context, image.data(), 1.0, dra, ddec, duv, pa
    );
}

dreal _chi2_profile(
    int nr,
    void* intensity,
    dreal r_min,
    dreal dr,
    dreal dxy,
    int nxy,
    dreal inc,
    dreal dra,
    dreal ddec,
    dreal duv,
    dreal pa,
    int nd,
    void* u,
    void* v,
    void* vis_obs_re,
    void* vis_obs_im,
    void* weights,
    int backend,
    dreal nufft_oversample
) {
    return chi2_profile(
        nr, static_cast<dreal*>(intensity), r_min, dr, dxy, nxy,
        inc, dra, ddec, duv, pa, nd,
        static_cast<dreal*>(u), static_cast<dreal*>(v),
        static_cast<dreal*>(vis_obs_re),
        static_cast<dreal*>(vis_obs_im),
        static_cast<dreal*>(weights),
        backend, nufft_oversample
    );
}

}
