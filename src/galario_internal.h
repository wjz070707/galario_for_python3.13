#pragma once

#include "galario.h"

#include <vector>

/*
 * Private seam between backend-neutral code and a concrete backend.
 *
 * galario_cpu.cpp and galario_gpu.cu provide the same *_impl symbols in
 * separate libraries. The void* FFT boundary keeps std::complex and CUDA
 * complex types out of cross-translation-unit symbol mangling.
 */
dreal clamp_nufft_oversample(dreal oversample);
int next_power_of_two(int value);
int resolve_padded_size(int nxy, dreal oversample);
int resolve_image_backend(int nx, int ny, int nd, int backend);
int resolve_batched_image_backend(int nx, int ny, int nd, int batch_size, int backend);
int resolve_profile_backend(int nr, int nd, int backend);
std::vector<dreal> pad_real_image(
    int nx_in,
    int ny_in,
    const dreal* realdata,
    int nx_out,
    int ny_out
);

namespace galario_internal {

void rasterize_component_image(
    int nx_model,
    int ny_model,
    dreal dxy,
    int ngauss,
    const dreal* gauss_params,
    int nrings,
    const dreal* ring_params,
    int narcs,
    const dreal* arc_params,
    dreal inc,
    int nx_out,
    int ny_out,
    dreal* out
);

galario::Chi2ImageContext* create_image_context_impl(
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
);
void destroy_image_context_impl(galario::Chi2ImageContext* context);
int image_context_requested_backend_impl(
    const galario::Chi2ImageContext* context
);
int image_context_backend_impl(const galario::Chi2ImageContext* context);
int image_context_batch_backend_impl(
    const galario::Chi2ImageContext* context,
    int batch_size
);

void fft2d_impl(int nx, int ny, void* data);
void fftshift_impl(int nx, int ny, void* data);
void fftshift_axis0_impl(int nrow, int ncol, void* matrix);
void interpolate_impl(
    int nrow,
    int ncol,
    const void* data,
    dreal v_origin,
    int nd,
    const dreal* u,
    const dreal* v,
    dreal duv,
    void* vis_int
);

}
