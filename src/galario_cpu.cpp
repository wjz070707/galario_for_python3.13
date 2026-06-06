/******************************************************************************
* This file is part of GALARIO:                                               *
* Gpu Accelerated Library for Analysing Radio Interferometer Observations     *
*                                                                             *
* Copyright (C) 2017-2020, Marco Tazzari, Frederik Beaujean, Leonardo Testi.  *
* Copyright (C) 2026, wjz070707.                                             *
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
* Maintained at https://github.com/wjz070707/galario_for_python3.13           *
******************************************************************************/

#include "galario.h"
#include "galario_internal.h"
#include "galario_profile_common.h"
#include "galario_py.h"

// full function makes code hard to read
#define tpb galario::threads()

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cstring>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <vector>

using std::to_string;

// general min function already available in cuda
// math_functions.hpp. Need `using` so the right implementation of
// `min` is chosen for the kernels that are both on gpu and cpu
#include <algorithm>
using std::min;
using std::max;

#include <fftw3.h>

/*
 * CPU backend.
 *
 * FFTW implements transform paths and OpenMP parallelizes shifts,
 * interpolation, direct Fourier sums, rasterization, and reductions. Cached
 * contexts retain aligned buffers and plans for repeated likelihood calls.
 */

// Stuff needed for GPU and CPU but should not be visible any other translation unit so we can use very common names.
namespace {
    /**
     * Provide a string buffer to avoid overhead from calling std::cout repeatedly.
     */
    std::ostringstream& out(bool reset=false) {
        static std::ostringstream my_stream;

        // insert a newline only if my_stream is empty
        if (!my_stream.tellp())
            my_stream.put('\n');

        if (reset) {
            my_stream.str("\n");
            my_stream.clear();
        }
        return my_stream;
    }

    void flush_timing() {
        // if nothing but the initial newline in there, nothing to show
        if (out().tellp() > 1)
            std::cout << out().str() << std::flush;
        // empty the stream
        out(true);
    }

    template <class T = std::runtime_error>
    void throw_exception(const char *file, const int line, const char* source, const std::string& msg) {
        std::stringstream ss;
        ss << file << ":" << line << ":\n";
        ss << "Error in " << source << ": " << msg;

        throw T(ss.str());
    }

   /**
    * Macros to check input image lengths.
    */
    #define CHECK_INPUT(nx) \
    do { \
        if (nx < 2) { throw_exception<std::invalid_argument>(__FILE__, __LINE__, "check input image", "x dimension = " + to_string(nx) + " is less than 2"); } \
        if (nx % 2 != 0) { throw_exception<std::invalid_argument>(__FILE__, __LINE__, "check input image", "x dimension = " + to_string(nx) + " is odd"); } \
    } while (0)

    #define CHECK_INPUTXY(nx, ny) \
    do { \
        if (nx != ny) { throw_exception<std::invalid_argument>(__FILE__, __LINE__, "check input image", "Expect a square image but got shape (" + to_string(nx) + ", " + to_string(ny) + ")"); } \
        CHECK_INPUT(nx); \
    } while (0)

#if defined(_OPENMP) && defined(GALARIO_TIMING)
    struct CPUTimer {
        double start;

        CPUTimer() {
            Start();
        }

        void Start() {
            start = omp_get_wtime();
        }

        void Elapsed(const std::string& msg) {
            const double elapsed = 1000 * (omp_get_wtime() - start);
            ::out() << "[CPU] " << msg << ": " << elapsed << " ms\n";
            // reset the timer for the next use
            Start();
        }
    };

    #define OPENMPTIME(body, msg)                                     \
    do {                                                              \
        CPUTimer t;                                                   \
        body;                                                         \
        t.Elapsed(msg);                                               \
    } while (false)
#else
    #define OPENMPTIME(body, msg) body
    struct CPUTimer {
        void Elapsed(const std::string&) {}
    };
#endif // _OPENMP && TIMING


    #define CMPLXSUB(a, b) ((a) - (b))
    #define CMPLXADD(a, b) ((a) + (b))
    #define CMPLXMUL(a, b) ((a) * (b))
    #define CMPLXCONJ conj

    #define CMPLXABS abs
    #define CMPLXARG arg

inline dreal cmplx_real_part(dcomplex const z) {
    return real(z);
}

inline dreal cmplx_imag_part(dcomplex const z) {
    return imag(z);
}

#define SQRT sqrt
#define FFTW(name) fftw_ ## name

} // anonymous namespace

namespace galario {
int threads(int num) {
    #if defined(_OPENMP)
        /* fix the number of openmp threads. disabling dynamic to respect the user's
           wish as much as possible */
        static int mynthreads = omp_get_max_threads();
        if (num > 0) {
            mynthreads = num;
            omp_set_dynamic(0);
            omp_set_num_threads(num);
        }
    #else
        // no threads, `num` ignored
        static int mynthreads = 1;
    #endif
    return mynthreads;
}

void init() {
    #ifdef _OPENMP
    const int status = FFTW(init_threads)();
    if (status == 0) {
        throw_exception(__FILE__, __LINE__, "fftw", "fftw_init_threads() failed");
    }
    #endif
}

void cleanup() {
    #ifdef _OPENMP
    FFTW(cleanup_threads)();
    #endif
    FFTW(cleanup)();
}

void galario_free(void* data) {
    fftw_free(data);
}
}


namespace galario {
/**
 * Copy an (nx, ny) square image into a complex buffer for real-to-complex FFTW.
 *
 * Buffer ownership transferred to caller, use `galario_free(buffer)`.
 *
 * If turns out to be slow have a look here:
 *   https://stackoverflow.com/questions/19601696/what-is-the-fastest-do-array-padding-of-the-image-array
 */
dcomplex* copy_input(int nx, int ny, const dreal* realdata) {
    CHECK_INPUTXY(nx, ny);
    // in r2c, the last dimension only has ~half the size
    auto const ncol = ny/2 + 1;

    // fftw_alloc for aligned memory to use SIMD acceleration
    auto buffer = reinterpret_cast<dcomplex*>(FFTW(alloc_complex)(nx*ncol));

    // copy and respect padding in last dimension. Treating the complex output
    // buffer as a sequence of real entries, the last (nx odd) or last two
    // columns (nx even) have to be skipped when copying in the input
    auto real_buffer = reinterpret_cast<dreal*>(buffer);

    // #reals = 2*#complex
    auto const rowsize = 2*ncol;

    // copy over entire input rows to output array
    auto const nbytes = sizeof(dreal)*ny;
#pragma omp parallel for shared(real_buffer, realdata)
    for (int i = 0; i < nx; ++i) {
       std::memcpy(&real_buffer[i*rowsize], &realdata[i*ny], nbytes);
    }
    return buffer;
}

void* _copy_input(int nx, int ny, void* realdata) {
    return copy_input(nx, ny, static_cast<dreal*>(realdata));
}
}

namespace {

void copy_input_h_into(int nx, int ny, const dreal* realdata, dcomplex* buffer) {
    auto real_buffer = reinterpret_cast<dreal*>(buffer);
    auto const rowsize = 2*(ny/2 + 1);
    auto const nbytes = sizeof(dreal)*ny;
#pragma omp parallel for shared(real_buffer, realdata)
    for (int i = 0; i < nx; ++i) {
        std::memcpy(&real_buffer[i*rowsize], &realdata[i*ny], nbytes);
    }
}

inline dreal wrap_angle_pi(dreal angle) {
    dreal const two_pi = 2. * (dreal)M_PI;
    return angle - two_pi * floor((angle + (dreal)M_PI) / two_pi);
}

inline dreal gaussian_ring_value(dreal radius, dreal flux, dreal ring_radius, dreal sigma) {
    dreal const diff = radius - ring_radius;
    return flux * exp(-(diff * diff) / (2. * sigma * sigma));
}

inline dreal gaussian_arc_value(dreal radius, dreal phi, dreal flux, dreal ring_radius, dreal sigma_radius,
                                dreal phi_center, dreal sigma_phi) {
    dreal const phi_wrapped = wrap_angle_pi(phi - phi_center);
    dreal const ring_term = gaussian_ring_value(radius, flux, ring_radius, sigma_radius);
    return ring_term * exp(-(phi_wrapped * phi_wrapped) / (2. * sigma_phi * sigma_phi));
}


} // anonymous namespace

namespace galario {

struct Chi2ImageContext {
    const int nx;
    const int ny;
    const int nd;
    const int requested_backend;
    const int backend;
    const dreal nufft_oversample;
    const int work_nx;
    const int work_ny;
    const int work_ncol;

    std::vector<dreal> model_image_h;
    std::vector<dreal> batch_inc_h;
    std::vector<dreal> batch_pa_h;
    std::vector<dreal> batch_dRA_h;
    std::vector<dreal> batch_dDec_h;
    std::vector<dreal> batch_gauss_h;
    std::vector<dreal> batch_ring_h;
    std::vector<dreal> batch_arc_h;
    dreal* u;
    dreal* v;
    dreal* vis_obs_re;
    dreal* vis_obs_im;
    dreal* weights;
    dreal* urot;
    dreal* vrot;
    dcomplex* vis_int;
    dcomplex* data;
    fftw_plan fft_plan;
    int fft_plan_threads;

    Chi2ImageContext(int nx_, int ny_, int nd_, const dreal* u_in, const dreal* v_in,
                     const dreal* vis_obs_re_in, const dreal* vis_obs_im_in, const dreal* weights_in,
                     int backend_in, dreal nufft_oversample_in)
        : nx(nx_), ny(ny_), nd(nd_), requested_backend(backend_in),
          backend(resolve_image_backend(nx_, ny_, nd_, backend_in)),
          nufft_oversample(clamp_nufft_oversample(nufft_oversample_in)),
          work_nx(backend == galario::BACKEND_NUFFT ? resolve_padded_size(nx_, nufft_oversample) : nx_),
          work_ny(work_nx), work_ncol(work_ny / 2 + 1),
          model_image_h(static_cast<size_t>(work_nx) * work_ny),
          u(reinterpret_cast<dreal*>(FFTW(alloc_real)(nd_))),
          v(reinterpret_cast<dreal*>(FFTW(alloc_real)(nd_))),
          vis_obs_re(reinterpret_cast<dreal*>(FFTW(alloc_real)(nd_))),
          vis_obs_im(reinterpret_cast<dreal*>(FFTW(alloc_real)(nd_))),
          weights(reinterpret_cast<dreal*>(FFTW(alloc_real)(nd_))),
          urot(reinterpret_cast<dreal*>(FFTW(alloc_real)(nd_))),
          vrot(reinterpret_cast<dreal*>(FFTW(alloc_real)(nd_))),
          vis_int(reinterpret_cast<dcomplex*>(FFTW(alloc_complex)(nd_))),
          data(reinterpret_cast<dcomplex*>(FFTW(alloc_complex)(work_nx * work_ncol))),
          fft_plan(nullptr), fft_plan_threads(0) {
        if (!u || !v || !vis_obs_re || !vis_obs_im || !weights || !urot || !vrot || !vis_int || !data) {
            throw std::bad_alloc();
        }
        std::memcpy(u, u_in, sizeof(dreal) * nd_);
        std::memcpy(v, v_in, sizeof(dreal) * nd_);
        std::memcpy(vis_obs_re, vis_obs_re_in, sizeof(dreal) * nd_);
        std::memcpy(vis_obs_im, vis_obs_im_in, sizeof(dreal) * nd_);
        std::memcpy(weights, weights_in, sizeof(dreal) * nd_);
    }

    ~Chi2ImageContext() {
        if (fft_plan) {
            FFTW(destroy_plan)(fft_plan);
        }
        galario_free(u);
        galario_free(v);
        galario_free(vis_obs_re);
        galario_free(vis_obs_im);
        galario_free(weights);
        galario_free(urot);
        galario_free(vrot);
        galario_free(vis_int);
        galario_free(data);
    }
};

} // namespace galario

namespace galario_internal {

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
) {
    CHECK_INPUTXY(nx, ny);
    return new galario::Chi2ImageContext(
        nx,
        ny,
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

void destroy_image_context_impl(galario::Chi2ImageContext* context) {
    delete context;
}

int image_context_requested_backend_impl(
    const galario::Chi2ImageContext* context
) {
    return context->requested_backend;
}

int image_context_backend_impl(
    const galario::Chi2ImageContext* context
) {
    return context->backend;
}

int image_context_batch_backend_impl(
    const galario::Chi2ImageContext* context,
    int batch_size
) {
    return resolve_batched_image_backend(
        context->nx,
        context->ny,
        context->nd,
        batch_size,
        context->requested_backend
    );
}

}

/**
 * Requires `data` to be large enough to hold the complex output after an
 * in-place transform, and the real input has to in the right memory locations
 * respecting the padding in the last dimension; see
 * http://fftw.org/fftw3_doc/Multi_002dDimensional-DFTs-of-Real-Data.html
 */
void fft_h(int nx, int ny, dcomplex* data) {
    dreal* input = reinterpret_cast<dreal*>(data);
    FFTW(complex)* output = reinterpret_cast<FFTW(complex)*>(data);
#ifdef _OPENMP
    fftw_plan_with_nthreads(galario::threads());
#endif
    FFTW(plan) p = FFTW(plan_dft_r2c_2d)(nx, ny, input, output, FFTW_ESTIMATE);
    if (!p) {
        throw std::runtime_error("FFTW failed to create a 2D transform plan");
    }
    FFTW(execute)(p);
    FFTW(destroy_plan)(p);
}

void execute_context_fft(galario::Chi2ImageContext* context) {
    int const plan_threads = galario::threads();
    if (!context->fft_plan || context->fft_plan_threads != plan_threads) {
        if (context->fft_plan) {
            FFTW(destroy_plan)(context->fft_plan);
            context->fft_plan = nullptr;
        }
#ifdef _OPENMP
        fftw_plan_with_nthreads(plan_threads);
#endif
        dreal* input = reinterpret_cast<dreal*>(context->data);
        FFTW(complex)* output =
            reinterpret_cast<FFTW(complex)*>(context->data);
        context->fft_plan = FFTW(plan_dft_r2c_2d)(
            context->work_nx,
            context->work_ny,
            input,
            output,
            FFTW_ESTIMATE
        );
        if (!context->fft_plan) {
            throw std::runtime_error("FFTW failed to create a cached 2D plan");
        }
        context->fft_plan_threads = plan_threads;
    }
    FFTW(execute)(context->fft_plan);
}

/**
 * Shift quadrants of the square image. Swap the upper-left quadrant with the
 * lower-right quadrant and the upper-right with the lower-left quadrant.
 *
 * We work on an array of real numbers stored inside a complex array, FFTW in-place format.
 *
 * To avoid if statements, we do two swaps.
 *
 * For cache efficiency, may have to do loop tiling; i.e., the source and target
 * should fit into the cache. If the image is too large, only part of a row may
 * fit. This is a responsibility of the caller.
 **/
// `a` is a matrix (size: nx*ny)
inline void shift_core(int const idx_x, int const idx_y, int const nx, int const ny, dreal* const __restrict__ a) {
    /* row-wise access */

    // number of real elements in the complex row of length ny/2+1
    auto const rowsize = 2*(ny/2+1);

    /* from upper left to lower right and from upper right to lower left */
    auto const src_ul = idx_x * rowsize + idx_y;
    auto const src_ur = src_ul + ny/2;

    // half the rows down
    auto const tgt_ul = src_ur + nx/2 * rowsize;

    // half a column to the left
    auto const tgt_ur = tgt_ul - ny/2;

    /* swap the values */
    auto tmp = a[src_ul];
    a[src_ul] = a[tgt_ul];
    a[tgt_ul] = tmp;

    tmp = a[src_ur];
    a[src_ur] = a[tgt_ur];
    a[tgt_ur] = tmp;
}

inline void shift_real_core(int const idx_x, int const idx_y, int const nx, int const ny,
                            dreal* const __restrict__ a) {
    auto const src_ul = idx_x * ny + idx_y;
    auto const src_ur = src_ul + ny / 2;
    auto const tgt_ul = src_ur + nx / 2 * ny;
    auto const tgt_ur = tgt_ul - ny / 2;

    auto tmp = a[src_ul];
    a[src_ul] = a[tgt_ul];
    a[tgt_ul] = tmp;

    tmp = a[src_ur];
    a[src_ur] = a[tgt_ur];
    a[tgt_ur] = tmp;
}

/**
 * grid stride loop
 */

void shift_h(int const nx, int const ny, dcomplex* const __restrict__ data) {
   dreal* a = reinterpret_cast<dreal*>(data);
#pragma omp parallel for
    for (auto x = 0; x < nx/2; ++x) {
        for (auto y = 0; y < ny/2; ++y) {
            shift_core(x, y, nx, ny, a);
        }
    }
}



/**
 * Shift quadrants of a rectangular matrix of size (nrow, ncol).
 * Swap the upper quadrant with the lower quadrant.
 *
 * For cache efficiency, may have to do loop tiling; i.e., the source and target
 * should fit into the cache. If the image is too large, only part of a row may
 * fit. This is a responsibility of the caller.
 **/
inline void shift_axis0_core(int const idx_x, int const idx_y, int const nrow, int const ncol, dcomplex* const __restrict__ matrix) {
    /* row-wise access */

    // from top-half to bottom-half
    auto const src_u = idx_x*ncol + idx_y;
    auto const tgt_u = src_u + nrow/2*ncol;

    // swap the values
    auto tmp = matrix[src_u];
    matrix[src_u] = matrix[tgt_u];
    matrix[tgt_u] = tmp;
}

/**
 * grid stride loop
 */

void shift_axis0_h(int const nrow, int const ncol, dcomplex* const __restrict__ matrix) {
#pragma omp parallel for
    for (auto x = 0; x < nrow/2; ++x) {
        for (auto y = 0; y < ncol; ++y) {
            shift_axis0_core(x, y, nrow, ncol, matrix);
        }
    }
}



/**
 * Bilinear interpolation in 2D according to Numerical Recipes.
 *
 * Interpolation of a matrix `data` in the generic point (u, v).
 *
 *     vis_int(u, v) = (1-t)(1-q)y0 + t(1-q)y1 + t*q*y2 + (1-t)*q*y3
 *                = t*q*(y0-y1+y2-y3) + t(-y0+y1) + q(-y0 + y3) + y0
 *
 * `y0` is bottom-left grid point, `y1` the bottom-right etc. forming a
 * a grid square around (u, v), ordered counter-clockwise.
 * `q` and `t` are the fractions of the desired location from left (bottom)
 * to right (upper) grid point.
 *
 * @param nrow, ncol : shape of the matrix `data`
 * @param data : complex 2D matrix containing the Real to Complex transform of an input image
 * @param u : x-axis coordinate of the point where `data` has to be interpolated
 * @param v : y-axis coordinate of the point where `data` has to be interpolated
 * @param duv : pixel size in the Fourier space, assumed to be uniform and the same in u and v direction
 * @returns: the interpolated point.
 *
 * Notes
 * The u and v coordinate axes follow the convention for radio interferometry for which
 * an input image has Right Ascension (x axis) increasing from Right to Left, and Declination
 * (y axis) increasing from Bottom to Top. u and v are parallel to Right Ascension and Declination, respectively.
 */
inline dcomplex interpolate_core(int const nrow, int const ncol, const dcomplex *const data, dreal const v_origin,
                                 const dreal u, const dreal v, const dreal duv) {

    const int half_nrow = nrow/2;

    // compute indices
    dreal const indu = fabs(u)/duv;
    dreal indv;  // also indv is const

    dreal const sign_u = copysign(1., u);

    indv = half_nrow + v_origin * sign_u * v / duv;

    // notations as in (3.6.5) of Numerical Recipes. They put the origin in the
    // lower-left.
    int const fl_u = floor(indu);
    int const fl_v = floor(indv);
    dcomplex const t = {indv - fl_v, 0.0};
    dcomplex const q = {indu - fl_u, 0.0};

    // linear index of y0
    int const base = fl_u + fl_v * ncol;

    /* the four grid points around the target point */
    const dcomplex& y0 = data[base];
    const dcomplex& y1 = data[base + ncol];
    const dcomplex& y2 = data[base + ncol + 1];
    const dcomplex& y3 = data[base + 1];

    /* ~ t*q */
    dcomplex const add1 = CMPLXADD(y0, y2);
    dcomplex const add2 = CMPLXADD(y1, y3);
    dcomplex const df1 = CMPLXSUB(add1, add2);
    dcomplex const mul1 = CMPLXMUL(q, df1);
    dcomplex const term1 = CMPLXMUL(t, mul1);

    /* ~ t */
    dcomplex const term2_sub = CMPLXSUB(y1, y0);
    dcomplex const term2 = CMPLXMUL(t, term2_sub);

    /* ~ q */
    dcomplex const term3_sub = CMPLXSUB(y3, y0);
    dcomplex const term3 = CMPLXMUL(q, term3_sub);

    /* add up all 4 terms */
    dcomplex const final_add2 = CMPLXADD(term2, term3);
    dcomplex const final_add1 = CMPLXADD(term1, final_add2);

    dreal const interp_phase = CMPLXARG(CMPLXADD(final_add1, y0)) * sign_u;

    dreal const tr = indv - fl_v;
    dreal const qr = indu - fl_u;

    dreal const y0r = CMPLXABS(y0);
    dreal const y1r = CMPLXABS(y1);
    dreal const y2r = CMPLXABS(y2);
    dreal const y3r = CMPLXABS(y3);

    dreal interp_amp = y0r;
    interp_amp += (y3r-y0r)*qr;
    interp_amp += (y1r-y0r)*tr;
    interp_amp += (y0r-y1r+y2r-y3r)*tr*qr;

    dcomplex interpolated = dcomplex{interp_amp*dreal(cos(interp_phase)), interp_amp*dreal(sin(interp_phase))};

    return interpolated;
}


void interpolate_h(int const nrow, int const ncol, const dcomplex* const data, dreal const v_origin, int const nd, const dreal* const u, const dreal* const v, dreal const duv, dcomplex* vis_int) {

#pragma omp parallel for
    for (auto idx = 0; idx < nd; ++idx) {
        vis_int[idx] = interpolate_core(nrow, ncol, data, v_origin, u[idx], v[idx], duv);
    }
}

namespace galario_internal {

void fft2d_impl(int nx, int ny, void* raw_data) {
    dcomplex* data = static_cast<dcomplex*>(raw_data);
    CHECK_INPUTXY(nx, ny);
    fft_h(nx, ny, data);
}

void fftshift_impl(int nx, int ny, void* raw_data) {
    dcomplex* data = static_cast<dcomplex*>(raw_data);
    CHECK_INPUTXY(nx, ny);
    shift_h(nx, ny, data);
}

void fftshift_axis0_impl(int nrow, int ncol, void* raw_matrix) {
    dcomplex* matrix = static_cast<dcomplex*>(raw_matrix);
    CHECK_INPUT(nrow);
    shift_axis0_h(nrow, ncol, matrix);
}

void interpolate_impl(
    int nrow,
    int ncol,
    const void* raw_data,
    dreal v_origin,
    int nd,
    const dreal* u,
    const dreal* v,
    dreal duv,
    void* raw_vis_int
) {
    const dcomplex* data = static_cast<const dcomplex*>(raw_data);
    dcomplex* vis_int = static_cast<dcomplex*>(raw_vis_int);
    interpolate_h(
        nrow,
        ncol,
        data,
        v_origin,
        nd,
        u,
        v,
        duv,
        vis_int
    );
}

}



inline void uv_rotate_core(dreal cos_PA, dreal sin_PA, dreal x, dreal y, dreal& xrot, dreal& yrot);

inline dcomplex phase_from_angle_core(dreal const angle);

inline dcomplex sample_image_visibility_core(int const nx, int const ny, const dreal* const image, dreal const v_origin,
                                             dreal const dxy, dreal const cos_PA, dreal const sin_PA,
                                             dreal const dRArot, dreal const dDecrot, dreal const u, dreal const v);

// APPLY_PHASE TO SAMPLED POINTS //
inline void apply_phase_sampled_core(int const idx_x, const dreal* const u, const dreal* const v, dcomplex* const __restrict__ vis_int, dreal const dRA, dreal const dDec) {

    dreal const angle = u[idx_x]*dRA + v[idx_x]*dDec;

    dcomplex const phase = dcomplex{dreal(cos(angle)), dreal(sin(angle))};

    vis_int[idx_x] = CMPLXMUL(vis_int[idx_x], phase);
}


void apply_phase_sampled_h(dreal dRA, dreal dDec, int const nd, const dreal* const u, const dreal* const v, dcomplex* const __restrict__ vis_int) {

    if ((dRA==0.) && (dDec==0.)) {
        return;
    }

    dRA *= 2.*(dreal)M_PI;
    dDec *= 2.*(dreal)M_PI;

#pragma omp parallel for shared(dRA, dDec) schedule(static)
    for (auto x = 0; x < nd; ++x) {
        apply_phase_sampled_core(x, u, v, vis_int, dRA, dDec);
    }
}

namespace galario {
void apply_phase_sampled(dreal dRA, dreal dDec, int const nd, const dreal* const u, const dreal* const v, dcomplex* const __restrict__ vis_int) {
    apply_phase_sampled_h(dRA, dDec, nd, u, v, vis_int);
}

void _apply_phase_sampled(dreal dRA, dreal dDec, int nd, void* const u,
                                  void* const v, void* __restrict__ vis_int) {
    apply_phase_sampled(dRA, dDec, nd, static_cast<dreal*>(u),
                                static_cast<dreal*>(v), static_cast<dcomplex*>(vis_int));
}
}

/**
 * Rotates the RA, Dec offsets and the u and v coordinates by Position Angle PA
 */
inline void uv_rotate_core(dreal cos_PA, dreal sin_PA, const dreal u, const dreal v, dreal& urot, dreal& vrot) {

    urot = u * cos_PA - v * sin_PA;
    vrot = u * sin_PA + v * cos_PA;

}


void uv_rotate_h(dreal PA, dreal dRA, dreal dDec, dreal* dRArot, dreal* dDecrot, int const nd, const dreal* const u, const dreal* const v,
                 dreal* const urot, dreal* vrot) {
    CPUTimer t;

    if (PA==0.) {
        *dRArot = dRA;
        *dDecrot = dDec;
        memcpy(urot, u, sizeof(dreal)*nd);
        memcpy(vrot, v, sizeof(dreal)*nd);
        return;
    }

    const dreal cos_PA = cos(PA);
    const dreal sin_PA = sin(PA);

#pragma omp parallel for
    for (auto i = 0; i < nd; ++i) {
        uv_rotate_core(cos_PA, sin_PA, u[i], v[i], urot[i], vrot[i]);
    }

    uv_rotate_core(cos_PA, sin_PA, dRA, dDec, *dRArot, *dDecrot);

    t.Elapsed("uv_rotate_h");
}

namespace galario {

void uv_rotate(dreal PA, dreal dRA, dreal dDec, dreal* dRArot, dreal* dDecrot, int const nd, const dreal* const u, const dreal* const v,
                       dreal* const urot, dreal* const vrot) {
    uv_rotate_h(PA, dRA, dDec, dRArot, dDecrot, nd, u, v, urot, vrot);
}

void _uv_rotate(dreal PA, dreal dRA, dreal dDec, void* dRArot, void* dDecrot, int nd, void* const u,
                                  void* const v, void* const urot, void* const vrot) {
    uv_rotate(PA, dRA, dDec, static_cast<dreal*>(dRArot), static_cast<dreal*>(dDecrot), nd, static_cast<dreal*>(u),
                                static_cast<dreal*>(v), static_cast<dreal*>(urot), static_cast<dreal*>(vrot));
}
}

inline dcomplex phase_from_angle_core(dreal const angle) {
    return dcomplex{dreal(cos(angle)), dreal(sin(angle))};
}

inline dcomplex sample_image_visibility_core(int const nx, int const ny, const dreal* const image, dreal const v_origin,
                                             dreal const dxy, dreal const cos_PA, dreal const sin_PA,
                                             dreal const dRArot, dreal const dDecrot, dreal const u, dreal const v) {
    dreal urot;
    dreal vrot;
    uv_rotate_core(cos_PA, sin_PA, u, v, urot, vrot);

    dreal const x0 = (ny / 2) * dxy;
    dreal const y0 = v_origin * (nx / 2) * dxy;

    dcomplex const col_init = phase_from_angle_core(-2. * (dreal)M_PI * urot * x0);
    dcomplex const col_step = phase_from_angle_core( 2. * (dreal)M_PI * urot * dxy);
    dcomplex const row_step = phase_from_angle_core( 2. * (dreal)M_PI * vrot * v_origin * dxy);

    dcomplex row_phase = phase_from_angle_core(-2. * (dreal)M_PI * vrot * y0);
    dcomplex vis = dcomplex{};

    for (auto i = 0; i < nx; ++i) {
        dcomplex phase = CMPLXMUL(row_phase, col_init);
        auto const row_offset = i * ny;

        for (auto j = 0; j < ny; ++j) {
            dcomplex const pixel = dcomplex{image[row_offset + j], 0.0};
            vis = CMPLXADD(vis, CMPLXMUL(pixel, phase));
            phase = CMPLXMUL(phase, col_step);
        }

        row_phase = CMPLXMUL(row_phase, row_step);
    }

    dcomplex const shift_phase = phase_from_angle_core(2. * (dreal)M_PI * (urot * dRArot + vrot * dDecrot));
    return CMPLXMUL(vis, shift_phase);
}

void sample_image_direct_h(int const nx, int const ny, const dreal* const image, dreal const v_origin,
                           dreal const dxy, dreal const PA, dreal const dRA, dreal const dDec, int const nd,
                           const dreal* const u, const dreal* const v, dcomplex* const vis_int) {
    dreal const cos_PA = cos(PA);
    dreal const sin_PA = sin(PA);
    dreal dRArot;
    dreal dDecrot;
    uv_rotate_core(cos_PA, sin_PA, dRA, dDec, dRArot, dDecrot);

#pragma omp parallel for
    for (auto idx = 0; idx < nd; ++idx) {
        vis_int[idx] = sample_image_visibility_core(nx, ny, image, v_origin, dxy, cos_PA, sin_PA,
                                                    dRArot, dDecrot, u[idx], v[idx]);
    }
}


void sample_h(int nx, int ny, dcomplex* data, const dreal v_origin, dreal dRA, dreal dDec, int nd, dreal duv, const dreal PA, const dreal* u, const dreal* v, dcomplex* vis_int) {
    CPUTimer t_start;

    int const ncol = ny/2+1;

    OPENMPTIME(shift_h(nx, ny, data), "sample::1st_shift");

    OPENMPTIME(fft_h(nx, ny, data), "sample::FFT");

    OPENMPTIME(shift_axis0_h(nx, ncol, data), "sample::2nd_shift");

    auto urot = reinterpret_cast<dreal*>(FFTW(alloc_real)(nd));
    auto vrot = reinterpret_cast<dreal*>(FFTW(alloc_real)(nd));
    dreal dRArot;
    dreal dDecrot;
    uv_rotate_h(PA, dRA, dDec, &dRArot, &dDecrot, nd, u, v, urot, vrot);

    // interpolate
    OPENMPTIME(interpolate_h(nx, ncol, data, v_origin, nd, urot, vrot, duv, vis_int), "sample::interpolate");

    // apply phase to the sampled points
    OPENMPTIME(apply_phase_sampled_h(dRArot, dDecrot, nd, urot, vrot, vis_int), "sample::apply_phase_sampled");

    galario::galario_free(urot);
    galario::galario_free(vrot);
    t_start.Elapsed("sample_tot");
}

inline void conjugate_vis_h(int const nd, dcomplex* vis) {
#pragma omp parallel for
    for (auto idx = 0; idx < nd; ++idx) {
        vis[idx] = CMPLXCONJ(vis[idx]);
    }
}

void sample_h_cached(galario::Chi2ImageContext* context, const dreal v_origin, dreal dRA, dreal dDec, dreal duv, const dreal PA) {
    CPUTimer t_start;
    if (context->backend == galario::BACKEND_DFT) {
        sample_image_direct_h(context->nx, context->ny, reinterpret_cast<dreal*>(context->data), v_origin, 1. / (duv * context->nx),
                              PA, dRA, dDec, context->nd, context->u, context->v, context->vis_int);
    } else {
        dreal dRArot;
        dreal dDecrot;
        dreal const duv_backend = duv * context->nx / context->work_nx;

        OPENMPTIME(shift_h(context->work_nx, context->work_ny, context->data), "sample_cached::1st_shift");
        // FFTW plans are bound to their buffers. The context owns both and
        // rebuilds the plan only when the configured thread count changes.
        OPENMPTIME(execute_context_fft(context), "sample_cached::FFT");
        OPENMPTIME(shift_axis0_h(context->work_nx, context->work_ncol, context->data), "sample_cached::2nd_shift");

        uv_rotate_h(PA, dRA, dDec, &dRArot, &dDecrot, context->nd, context->u, context->v, context->urot, context->vrot);

        OPENMPTIME(interpolate_h(context->work_nx, context->work_ncol, context->data, v_origin, context->nd, context->urot, context->vrot, duv_backend, context->vis_int), "sample_cached::interpolate");
        OPENMPTIME(apply_phase_sampled_h(dRArot, dDecrot, context->nd, context->urot, context->vrot, context->vis_int), "sample_cached::apply_phase_sampled");
        if (context->backend == galario::BACKEND_NUFFT) {
            OPENMPTIME(conjugate_vis_h(context->nd, context->vis_int), "sample_cached::conjugate");
        }
    }

    t_start.Elapsed("sample_cached_tot");
}



namespace galario {
/**
 * return result in `vis_int`
 */
void sample_image(int nx, int ny, const dreal* realdata, dreal v_origin, dreal dRA, dreal dDec, dreal duv,
                          const dreal PA, int nd, const dreal* u, const dreal* v, dcomplex* vis_int,
                          int backend, dreal nufft_oversample) {
    CPUTimer t_start;

    // Initialization for uv_idx and interpolate
    CHECK_INPUT(nx);
    int const resolved_backend = resolve_image_backend(nx, ny, nd, backend);

    if (resolved_backend == galario::BACKEND_DFT) {
        sample_image_direct_h(nx, ny, realdata, v_origin, 1. / (duv * nx), PA, dRA, dDec, nd, u, v, vis_int);
    } else {
        CPUTimer t;
        int const work_nx = resolved_backend == galario::BACKEND_NUFFT ? resolve_padded_size(nx, nufft_oversample) : nx;
        int const work_ny = work_nx;
        dreal const duv_backend = duv * nx / work_nx;
        dcomplex* data = reinterpret_cast<dcomplex*>(FFTW(alloc_complex)(work_nx * (work_ny / 2 + 1)));
        if (!data) {
            throw std::bad_alloc();
        }
        if (resolved_backend == galario::BACKEND_NUFFT) {
            auto padded_real = pad_real_image(nx, ny, realdata, work_nx, work_ny);
            copy_input_h_into(work_nx, work_ny, padded_real.data(), data);
        } else {
            copy_input_h_into(nx, ny, realdata, data);
        }
        sample_h(work_nx, work_ny, data, v_origin, dRA, dDec, nd, duv_backend, PA, u, v, vis_int);
        if (resolved_backend == galario::BACKEND_NUFFT) {
            conjugate_vis_h(nd, vis_int);
        }
        t = CPUTimer(); galario_free(data); t.Elapsed("sample_image::free_data");
    }
    t_start.Elapsed("sample_image_tot");
}

void _sample_image(int nx, int ny, void* data, dreal v_origin, dreal dRA, dreal dDec, dreal duv, dreal PA, int nd, void* u, void* v, void* vis_int, int backend, dreal nufft_oversample) {
    sample_image(nx, ny, static_cast<dreal*>(data), v_origin, dRA, dDec, duv, PA, nd, static_cast<dreal*>(u), static_cast<dreal*>(v), static_cast<dcomplex*>(vis_int),
                 backend, nufft_oversample);
}

}


/**
 * Compute weighted difference between observations (`vis_obs_re` and `vis_obs_im`) and model predictions `vis_int`, write to `vis_int`
 */
namespace galario {
dreal reduce_chi2(int nd, const dreal* vis_obs_re, const dreal* vis_obs_im, const dcomplex* vis_int, const dreal* weights) {
     CPUTimer t_start;
     dreal chi2 = 0.;
     // compute chi2 by hand in a single pass over data, avoiding creation of
     // intermediate complex values
#pragma omp parallel for reduction(+:chi2)
     for (auto idx = 0; idx < nd; ++idx) {
         dreal const dr = cmplx_real_part(vis_int[idx]) - vis_obs_re[idx];
         dreal const di = cmplx_imag_part(vis_int[idx]) - vis_obs_im[idx];
         // Avoid sqrt(weight) and temporary complex values in this hot loop.
         chi2 += (dr * dr + di * di) * weights[idx];
     }
     t_start.Elapsed("reduce_chi2_tot");

     return chi2;
}

dreal _reduce_chi2(int nd, void* vis_obs_re, void* vis_obs_im, void* vis_int, void* weights) {
    return reduce_chi2(nd, static_cast<dreal*>(vis_obs_re), static_cast<dreal*>(vis_obs_im), static_cast<dcomplex*>(vis_int), static_cast<dreal*>(weights));
}

int ngpus()
{
    int num_devices = 0;
    return num_devices;
}

void use_gpu(int device_id)
{
}

dreal chi2_image_from_context(Chi2ImageContext* context, const dreal* realdata, const dreal v_origin, dreal dRA, dreal dDec, dreal duv, dreal PA) {
    CPUTimer t_start;

    CHECK_INPUTXY(context->nx, context->ny);
    dreal chi2 = 0;
    if (context->backend == galario::BACKEND_DFT) {
        std::memcpy(reinterpret_cast<dreal*>(context->data), realdata, sizeof(dreal) * context->nx * context->ny);
    } else if (context->backend == galario::BACKEND_NUFFT) {
        auto padded_real = pad_real_image(context->nx, context->ny, realdata, context->work_nx, context->work_ny);
        copy_input_h_into(context->work_nx, context->work_ny, padded_real.data(), context->data);
    } else {
        copy_input_h_into(context->nx, context->ny, realdata, context->data);
    }
    sample_h_cached(context, v_origin, dRA, dDec, duv, PA);
    chi2 = reduce_chi2(context->nd, context->vis_obs_re, context->vis_obs_im, context->vis_int, context->weights);
    t_start.Elapsed("chi2_image_from_context_tot");
    flush_timing();

    return chi2;
}

void chi2_profile_from_context_batch(
    Chi2ImageContext* context,
    int nr,
    const dreal* intensity_batch,
    int batch_size,
    dreal r_min,
    dreal dr,
    int nxy,
    dreal dxy,
    const dreal* inc_batch,
    const dreal* dRA_batch,
    const dreal* dDec_batch,
    dreal duv,
    const dreal* PA_batch,
    dreal* chi2_out
) {
    std::vector<dcomplex> packed_image(
        static_cast<size_t>(nxy) * (nxy / 2 + 1)
    );
    for (int idx = 0; idx < batch_size; ++idx) {
        galario_profile_detail::sweep_image(
            nr,
            intensity_batch + static_cast<size_t>(idx) * nr,
            r_min, dr, nxy, dxy, inc_batch[idx],
            packed_image.data()
        );
        std::vector<dreal> image =
            galario_profile_detail::unpack_image(
                nxy, packed_image.data()
            );
        chi2_out[idx] = chi2_image_from_context(
            context, image.data(), 1.0,
            dRA_batch[idx], dDec_batch[idx], duv, PA_batch[idx]
        );
    }
}

namespace {
dreal chi2_image_from_context_rasterized(galario::Chi2ImageContext* context, const dreal v_origin,
                                        dreal dRA, dreal dDec, dreal duv, dreal PA) {
    dreal chi2 = 0;
    if (context->backend == galario::BACKEND_DFT) {
        std::memcpy(reinterpret_cast<dreal*>(context->data), context->model_image_h.data(),
                    sizeof(dreal) * context->nx * context->ny);
    } else {
        copy_input_h_into(context->work_nx, context->work_ny, context->model_image_h.data(), context->data);
    }
    sample_h_cached(context, v_origin, dRA, dDec, duv, PA);
    chi2 = reduce_chi2(context->nd, context->vis_obs_re, context->vis_obs_im, context->vis_int, context->weights);
    return chi2;
}

} // anonymous namespace

dreal chi2_image_from_context_components(Chi2ImageContext* context, dreal dxy,
                                   int ngauss, const dreal* gauss_params,
                                   int nrings, const dreal* ring_params,
                                   int narcs, const dreal* arc_params,
                                   dreal inc, const dreal v_origin,
                                   dreal dRA, dreal dDec, dreal duv, dreal PA) {
    CPUTimer t_start;

    CHECK_INPUTXY(context->nx, context->ny);
    galario_internal::rasterize_component_image(context->nx, context->ny, dxy,
                              ngauss, gauss_params,
                              nrings, ring_params,
                              narcs, arc_params,
                              inc,
                              context->backend == galario::BACKEND_NUFFT ? context->work_nx : context->nx,
                              context->backend == galario::BACKEND_NUFFT ? context->work_ny : context->ny,
                              context->model_image_h.data());

    dreal chi2 = chi2_image_from_context_rasterized(context, v_origin, dRA, dDec, duv, PA);
    t_start.Elapsed("chi2_image_from_context_components_tot");
    flush_timing();

    return chi2;
}

void chi2_image_from_context_components_batch(Chi2ImageContext* context, dreal dxy,
                                        int batch_size,
                                        int ngauss, const dreal* gauss_params_batch,
                                        int nrings, const dreal* ring_params_batch,
                                        int narcs, const dreal* arc_params_batch,
                                        const dreal* inc_batch, const dreal v_origin,
                                        const dreal* dRA_batch, const dreal* dDec_batch,
                                        dreal duv, const dreal* PA_batch, dreal* chi2_out) {
    CPUTimer t_start;
    CHECK_INPUTXY(context->nx, context->ny);
    int const effective_backend = resolve_batched_image_backend(context->nx, context->ny, context->nd,
                                                                batch_size, context->requested_backend);

    (void)effective_backend;

    auto const gauss_stride = ngauss * 2;
    auto const ring_stride = nrings * 3;
    auto const arc_stride = narcs * 5;
    auto const work_nx = context->backend == galario::BACKEND_NUFFT ? context->work_nx : context->nx;
    auto const work_ny = context->backend == galario::BACKEND_NUFFT ? context->work_ny : context->ny;

    for (int idx = 0; idx < batch_size; ++idx) {
        const dreal* gauss_params = gauss_stride > 0 ? gauss_params_batch + static_cast<size_t>(idx) * gauss_stride : nullptr;
        const dreal* ring_params = ring_stride > 0 ? ring_params_batch + static_cast<size_t>(idx) * ring_stride : nullptr;
        const dreal* arc_params = arc_stride > 0 ? arc_params_batch + static_cast<size_t>(idx) * arc_stride : nullptr;
        galario_internal::rasterize_component_image(context->nx, context->ny, dxy,
                                  ngauss, gauss_params,
                                  nrings, ring_params,
                                  narcs, arc_params,
                                  inc_batch[idx],
                                  work_nx, work_ny,
                                  context->model_image_h.data());
        chi2_out[idx] = chi2_image_from_context_rasterized(context, v_origin, dRA_batch[idx], dDec_batch[idx], duv, PA_batch[idx]);
    }

    t_start.Elapsed("chi2_image_from_context_components_tot");
    flush_timing();
}
}
