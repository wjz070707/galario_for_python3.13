/******************************************************************************
* This file is part of GALARIO:                                               *
* Gpu Accelerated Library for Analysing Radio Interferometer Observations     *
*                                                                             *
* Copyright (C) 2017-2020, Marco Tazzari, Frederik Beaujean, Leonardo Testi.  *
*                                                                             *
* This program is free software: you can redistribute it and/or modify        *
* it under the terms of the Lesser GNU General Public License as published by *
* the Free Software Foundation, either version 3 of the License, or           *
* (at your option) any later version.                                         *
*                                                                             *
* This program is distributed in the hope that it will be useful,             *
* but WITHOUT ANY WARRANTY; without even the implied warranty of              *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                        *
*                                                                             *
* For more details see the LICENSE file.                                      *
* For documentation see https://mtazzari.github.io/galario/                   *
******************************************************************************/

#pragma once

#include "galario_defs.h"

namespace galario {

enum Backend {
    BACKEND_AUTO = 0,
    BACKEND_FFT = 1,
    BACKEND_DFT = 2,
    BACKEND_NUFFT = 3
};

/* Main user functions */
void sample_profile(int nr, const dreal* intensity, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal inc, dreal dRA,
                    dreal dDec, dreal duv, dreal PA, int nd, const dreal *u, const dreal *v, dcomplex *vis_int,
                    int backend = BACKEND_AUTO, dreal nufft_oversample = 2.0);
void sample_image(int nx, int ny, const dreal* image, const dreal v_origin, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, const dreal* u, const dreal* v, dcomplex* vis_int,
                  int backend = BACKEND_AUTO, dreal nufft_oversample = 2.0);
dreal chi2_profile(int nr, const dreal* intensity, dreal Rmin, dreal dR, dreal dxy, int nxy, dreal inc, dreal dRA,
                   dreal dDec, dreal duv, dreal PA, int nd, const dreal *u, const dreal *v, const dreal *vis_obs_re,
                   const dreal *vis_obs_im, const dreal *weights, int backend = BACKEND_AUTO, dreal nufft_oversample = 2.0);
dreal chi2_image(int nx, int ny, const dreal* image, const dreal v_origin, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, const dreal* u, const dreal* v, const dreal* vis_obs_re, const dreal* vis_obs_im, const dreal* weights,
                 int backend = BACKEND_AUTO, dreal nufft_oversample = 2.0);
struct Chi2ImageContext;
Chi2ImageContext* create_image_context(int nx, int ny, int nd, const dreal* u, const dreal* v,
                                            const dreal* vis_obs_re, const dreal* vis_obs_im,
                                            const dreal* weights, int backend = BACKEND_AUTO, dreal nufft_oversample = 2.0);
void destroy_image_context(Chi2ImageContext* context);
int image_context_requested_backend(const Chi2ImageContext* context);
int image_context_backend(const Chi2ImageContext* context);
int image_context_batch_backend(const Chi2ImageContext* context, int batch_size);
void sample_image_components(int nx, int ny, dreal dxy,
                             int ngauss, const dreal* gauss_params,
                             int nrings, const dreal* ring_params,
                             int narcs, const dreal* arc_params,
                             dreal inc, const dreal v_origin,
                             dreal dRA, dreal dDec, dreal duv, dreal PA,
                             int nd, const dreal* u, const dreal* v, dcomplex* vis_int,
                             int backend = BACKEND_AUTO, dreal nufft_oversample = 2.0);
dreal chi2_image_components(int nx, int ny, dreal dxy,
                            int ngauss, const dreal* gauss_params,
                            int nrings, const dreal* ring_params,
                            int narcs, const dreal* arc_params,
                            dreal inc, const dreal v_origin,
                            dreal dRA, dreal dDec, dreal duv, dreal PA,
                            int nd, const dreal* u, const dreal* v,
                            const dreal* vis_obs_re, const dreal* vis_obs_im, const dreal* weights,
                            int backend = BACKEND_AUTO, dreal nufft_oversample = 2.0);
dreal chi2_image_from_context_components(Chi2ImageContext* context, dreal dxy,
                                   int ngauss, const dreal* gauss_params,
                                   int nrings, const dreal* ring_params,
                                   int narcs, const dreal* arc_params,
                                   dreal inc, const dreal v_origin,
                                   dreal dRA, dreal dDec, dreal duv, dreal PA);
void chi2_image_from_context_components_batch(Chi2ImageContext* context, dreal dxy,
                                        int batch_size,
                                        int ngauss, const dreal* gauss_params_batch,
                                        int nrings, const dreal* ring_params_batch,
                                        int narcs, const dreal* arc_params_batch,
                                        const dreal* inc_batch, const dreal v_origin,
                                        const dreal* dRA_batch, const dreal* dDec_batch,
                                        dreal duv, const dreal* PA_batch, dreal* chi2_out);
dreal chi2_image_from_context(Chi2ImageContext* context, const dreal* image, const dreal v_origin,
                        dreal dRA, dreal dDec, dreal duv, dreal PA);
dreal chi2_profile_from_context(Chi2ImageContext* context,
                                int nr, const dreal* intensity,
                                dreal Rmin, dreal dR, int nxy, dreal dxy,
                                dreal inc, dreal dRA, dreal dDec,
                                dreal duv, dreal PA);

void sweep(int nr, const dreal* intensity, dreal Rmin, dreal dR, int nxy, dreal dxy, dreal inc, dcomplex *image);
void uv_rotate(dreal PA, dreal dRA, dreal dDec, dreal* dRArot, dreal* dDecrot, int nd, const dreal* u, const dreal* v, dreal* urot, dreal* vrot);

/* Interface for the experts */
dcomplex* copy_input(int nx, int ny, const dreal* image);
void galario_free(void*);
void fft2d(int nx, int ny, dcomplex* image);
void fftshift(int nx, int ny, dcomplex* image);
void fftshift_axis0(int nx, int ny, dcomplex* matrix);
void interpolate(int nrow, int ncol, const dcomplex *image, const dreal v_origin, int nd,
                 const dreal* u, const dreal* v, const dreal duv,
                 dcomplex* vis_int);
void apply_phase_sampled(dreal dRA, dreal dDec, int nd, const dreal* u,
                         const dreal* v, dcomplex* vis_int);
dreal reduce_chi2(int nd, const dreal* vis_obs_re, const dreal* vis_obs_im,
                 const dcomplex* vis_int, const dreal* weights);

/* Required for multithreading */
void init();
void cleanup();
int threads(int num = 0);

/* GPU related functions */
int ngpus();
void use_gpu(int device_id);
}
