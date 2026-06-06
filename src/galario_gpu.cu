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

#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuComplex.h>

#include <cublas_v2.h>
#include <cufft.h>

#include <cstdio>
#include <cstdlib>
#include <mutex>

/*
 * CUDA backend.
 *
 * This file is longer than the CPU backend because GPU execution requires
 * explicit device-memory ownership, transfers, launch geometry, kernels,
 * synchronization, error translation, and reusable cuFFT workspaces. Its
 * public behavior still mirrors galario_cpu.cpp through galario_internal.h.
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


    void throw_exception(const char *file, const int line, const char* source, const int err) {
        std::stringstream ss;
        ss << "Failed with error code " << err;
        throw_exception(file, line, source, ss.str());
    }

    #define CCheck(err) __cudaSafeCall((err), __FILE__, __LINE__)
    inline void __cudaSafeCall(cudaError err, const char *file, const int line)  {
        if (err == cudaErrorInitializationError) {
            throw_exception(file, line, "cuda", "Could not initialize cuda. Is a CUDA GPU available at all?");
        }
        if (err == cudaErrorMemoryAllocation) {
            throw std::bad_alloc();
        }
        if (cudaSuccess != err) {
            throw_exception(file, line, "cuda", cudaGetErrorString(err));
        }
    }

    #define CBlasCheck(err) __cublasSafeCall((err), __FILE__, __LINE__)
    inline void __cublasSafeCall(cublasStatus_t err, const char *file, const int line) {
        if (err == CUBLAS_STATUS_NOT_INITIALIZED) {
            throw_exception(file, line, "cublas", "Could not initialize cublas. Is a cuda GPU available at all? Or is it ouf memory?");
        }
        if (err == CUBLAS_STATUS_ALLOC_FAILED) {
            throw std::bad_alloc();
        }
        if (CUBLAS_STATUS_SUCCESS != err) {
            throw_exception(file, line, "cublas", err);
        }
    }

    #define CUFFTCheck(err) __cufftwSafeCall((err), __FILE__, __LINE__)
    inline void __cufftwSafeCall(cufftResult_t err, const char *file, const int line) {
        if (err == CUFFT_ALLOC_FAILED) {
            throw std::bad_alloc();
        }
       if (CUFFT_SUCCESS != err) {
           throw_exception(file, line, "cufftw", err);
       }
    }

    cublasHandle_t cublasHandle = nullptr;
    std::mutex cublasHandle_mutex;

    bool cublas_initialized() {
        return cublasHandle != nullptr;
    }

    void cublas_init() {
        // lock to prevent data race
        std::lock_guard<std::mutex> lock(cublasHandle_mutex);

        // check if handle initialized to avoid 2nd thread in race condition to initialize again
        if (cublas_initialized()) {
            return;
        }

        // actually init
        CBlasCheck(cublasCreate(&cublasHandle));
    }

    cublasHandle_t& cublas_handle() {
        if (!cublas_initialized()) {
            cublas_init();
        }
        return cublasHandle;
    }

    /**
     * A simple RAII wrapper around cuda memory for exception safety
     */
    template <typename T>
    struct CudaMemory {
        CudaMemory(size_t n) : nbytes(sizeof(T) * n) {
            const auto error = cudaMalloc(&ptr, nbytes);
            if (error != cudaSuccess) {
                // If this fails, it hides the first error from allocation
                CCheck(cudaFree(ptr));

                // safe to throw an error now, no memory dangling
                CCheck(error);
            }
        }

        /**
         * Allocate and copy `n` elements of type `T` from `source` to device
         */
        CudaMemory(size_t n, const T* source) : CudaMemory(n) {
            CCheck(cudaMemcpy(ptr, source, nbytes, cudaMemcpyHostToDevice));
        }

        // forbid copy operations to avoid double ownership
        CudaMemory(const CudaMemory&) = delete;
        CudaMemory& operator=(const CudaMemory&) = delete;

        // move operations transfer ownership
        CudaMemory(CudaMemory&&) = default;
        CudaMemory& operator=(CudaMemory&&) = default;

        // Should not throw an exception inside destructor, so we don't `CCheck`
        ~CudaMemory() {
            cudaFree(ptr);
        }

        /// Copy back from device to host destination
        void Retrieve(T* destination) {
            CCheck(cudaMemcpy(destination, ptr, nbytes, cudaMemcpyDeviceToHost));
        }

        /// Copy back `count` elements from device to host destination.
        void RetrieveCount(T* destination, size_t count) {
            auto const bytes = sizeof(T) * count;
            if (bytes > nbytes) {
                throw std::out_of_range("CudaMemory::RetrieveCount exceeds allocation");
            }
            CCheck(cudaMemcpy(destination, ptr, bytes, cudaMemcpyDeviceToHost));
        }

        /// Copy up to `count` elements from host to device.
        void CopyFromHost(const T* source, size_t count) {
            auto const bytes = sizeof(T) * count;
            if (bytes > nbytes) {
                throw std::out_of_range("CudaMemory::CopyFromHost exceeds allocation");
            }
            CCheck(cudaMemcpy(ptr, source, bytes, cudaMemcpyHostToDevice));
        }

        /// The device pointer
        T* ptr;

        /// The size of the memory allocation
        const size_t nbytes;
    };

    #ifdef GALARIO_TIMING
        struct GPUTimer
        {
            cudaEvent_t start;
            cudaEvent_t stop;

            GPUTimer() {
                CCheck(cudaEventCreate(&start));
                CCheck(cudaEventCreate(&stop));
                Start();
            }

            ~GPUTimer() {
                CCheck(cudaEventDestroy(start));
                CCheck(cudaEventDestroy(stop));
            }

            void Start() {
                CCheck(cudaEventRecord(start, 0));
            }

            void Elapsed(const std::string& msg) {
                CCheck(cudaEventRecord(stop, 0));
                CCheck(cudaEventSynchronize(stop));
                float elapsed;
                CCheck(cudaEventElapsedTime(&elapsed, start, stop));
                ::out() << "[GPU] " << msg << ": " << elapsed << " ms\n";
                Start();
            }
        };
    #else
        struct GPUTimer
        {
            GPUTimer() {
                // call empty Start() just to avoid warning about unused function
                Start();
            }
            void Start() {}
            void Elapsed(const std::string& msg) {}
        };
    #endif // TIMING

#define CUFFTEXEC cufftExecD2Z
#define CUFFTTYPE CUFFT_D2Z
#define CMPLX(a, b) (make_cuDoubleComplex(a,b))
#define CMPLXSUB cuCsub
#define CMPLXADD cuCadd
#define CMPLXMUL cuCmul
#define CMPLXCONJ cuConj
#define CUBLASNRM2 cublasDznrm2

#define CMPLXABS cuCabs
#define CMPLXARG(a) atan2(cuCimag(a),cuCreal(a))


__host__ __device__
inline dreal cmplx_real_part(dcomplex const z) {
    return cuCreal(z);
}

__host__ __device__
inline dreal cmplx_imag_part(dcomplex const z) {
    return cuCimag(z);
}

#define SQRT sqrt
#define FFTW(name) fftw_ ## name

} // anonymous namespace

namespace galario {
int threads(int num) {
    // mynthreads^2 is used per block
    static int mynthreads = 16;
    // `num^2`: number of threads per block for 2D operations
    if (num > 0)
        mynthreads = num;
    return mynthreads;
}

void init() {
    // Avoid initializing cublas unconditionally. It takes a lot of memory and
    // fails if cuda is not available. Let the initialization be done only if
    // cublas is actually needed.
    // cublas_handle();
}

void cleanup() {
    if (cublas_initialized()) {
        CBlasCheck(cublasDestroy(cublas_handle()));
    }
}

void galario_free(void* data) {
    free(data);
}
}

/**
 * Return complex image on the device made from real image with array size `nx*ny` on the host.
 *
 */
CudaMemory<dcomplex> copy_input_d(int nx, int ny, const dreal* realdata) {
    GPUTimer t;
    auto const ncol = ny/2+1;
    auto const rowsize_real = sizeof(dreal)*ny;
    auto const rowsize_complex = sizeof(dcomplex)*ncol;

    // create destination array
    CudaMemory<dcomplex> data_d(nx * ncol);

    // set the padding by defining different sizes of a row in bytes
    CCheck(cudaMemcpy2D(data_d.ptr, rowsize_complex, realdata, rowsize_real, rowsize_real, nx, cudaMemcpyHostToDevice));
    t.Elapsed("copy_input_H->D");
    return data_d;
}

void copy_input_d_into(int nx, int ny, const dreal* realdata, dcomplex* data_d) {
    GPUTimer t;
    auto const ncol = ny/2+1;
    auto const rowsize_real = sizeof(dreal)*ny;
    auto const rowsize_complex = sizeof(dcomplex)*ncol;

    CCheck(cudaMemcpy2D(data_d, rowsize_complex, realdata, rowsize_real, rowsize_real, nx, cudaMemcpyHostToDevice));
    t.Elapsed("copy_input_H->D");
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

    auto buffer = static_cast<dcomplex*>(malloc(sizeof(dcomplex)*nx*ncol));

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


__host__ __device__
inline dreal wrap_angle_pi(dreal angle) {
    dreal const two_pi = 2. * (dreal)M_PI;
    return angle - two_pi * floor((angle + (dreal)M_PI) / two_pi);
}

__host__ __device__
inline dreal gaussian_ring_value(dreal radius, dreal flux, dreal ring_radius, dreal sigma) {
    dreal const diff = radius - ring_radius;
    return flux * exp(-(diff * diff) / (2. * sigma * sigma));
}

__host__ __device__
inline dreal gaussian_arc_value(dreal radius, dreal phi, dreal flux, dreal ring_radius, dreal sigma_radius,
                                dreal phi_center, dreal sigma_phi) {
    dreal const phi_wrapped = wrap_angle_pi(phi - phi_center);
    dreal const ring_term = gaussian_ring_value(radius, flux, ring_radius, sigma_radius);
    return ring_term * exp(-(phi_wrapped * phi_wrapped) / (2. * sigma_phi * sigma_phi));
}

__global__ void rasterize_component_image_batch_d(int nx_model, int ny_model, dreal dxy,
                                                  int batch_size,
                                                  int ngauss, const dreal* gauss_params_batch,
                                                  int nrings, const dreal* ring_params_batch,
                                                  int narcs, const dreal* arc_params_batch,
                                                  const dreal* inc_batch,
                                                  int nx_out, int ny_out, dreal* out_batch) {
    int const batch_idx = blockIdx.z;
    if (batch_idx >= batch_size) {
        return;
    }

    int const i = blockDim.y * blockIdx.y + threadIdx.y;
    int const j = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= nx_out || j >= ny_out) {
        return;
    }

    dreal* const out = out_batch + static_cast<size_t>(batch_idx) * nx_out * ny_out;
    auto const row_offset = (nx_out - nx_model) / 2;
    auto const col_offset = (ny_out - ny_model) / 2;
    if (i < row_offset || i >= row_offset + nx_model || j < col_offset || j >= col_offset + ny_model) {
        out[static_cast<size_t>(i) * ny_out + j] = 0.0;
        return;
    }

    int const ii = i - row_offset;
    int const jj = j - col_offset;
    dreal const x_min = -0.5 * ny_model * dxy;
    dreal const y_min = -0.5 * nx_model * dxy;
    dreal const x_step = ny_model > 1 ? (ny_model * dxy) / (ny_model - 1) : 0.0;
    dreal const y_step = nx_model > 1 ? (nx_model * dxy) / (nx_model - 1) : 0.0;
    dreal const cos_inc = cos(inc_batch[batch_idx]);
    dreal const x = x_min + jj * x_step;
    dreal const y = y_min + ii * y_step;
    dreal const x_deproj = x / cos_inc;
    dreal const radius = sqrt(x_deproj * x_deproj + y * y);
    dreal const phi = atan2(y, x_deproj);
    dreal value = 0.0;

    if (ngauss > 0) {
        const dreal* const gauss_params = gauss_params_batch + static_cast<size_t>(batch_idx) * ngauss * 2;
        for (int g = 0; g < ngauss; ++g) {
            auto const base = 2 * g;
            dreal const flux = gauss_params[base];
            dreal const sigma = gauss_params[base + 1];
            value += flux * exp(-(radius * radius) / (2. * sigma * sigma));
        }
    }

    if (nrings > 0) {
        const dreal* const ring_params = ring_params_batch + static_cast<size_t>(batch_idx) * nrings * 3;
        for (int r = 0; r < nrings; ++r) {
            auto const base = 3 * r;
            value += gaussian_ring_value(radius, ring_params[base], ring_params[base + 1], ring_params[base + 2]);
        }
    }

    if (narcs > 0) {
        const dreal* const arc_params = arc_params_batch + static_cast<size_t>(batch_idx) * narcs * 5;
        for (int a = 0; a < narcs; ++a) {
            auto const base = 5 * a;
            value += gaussian_arc_value(radius, phi, arc_params[base], arc_params[base + 1], arc_params[base + 2],
                                        arc_params[base + 3], arc_params[base + 4]);
        }
    }

    out[static_cast<size_t>(i) * ny_out + j] = value;
}

__global__ void rasterize_profile_image_batch_d(
    int nr,
    const dreal* intensity_batch,
    int batch_size,
    dreal r_min,
    dreal dr,
    int nxy,
    dreal dxy,
    const dreal* inc_batch,
    int nx_out,
    int ny_out,
    dreal* out_batch
) {
    int const batch_idx = blockIdx.z;
    int const i = blockDim.y * blockIdx.y + threadIdx.y;
    int const j = blockDim.x * blockIdx.x + threadIdx.x;
    if (batch_idx >= batch_size || i >= nx_out || j >= ny_out) {
        return;
    }

    dreal* const out =
        out_batch + static_cast<size_t>(batch_idx) * nx_out * ny_out;
    int const row_offset = (nx_out - nxy) / 2;
    int const col_offset = (ny_out - nxy) / 2;
    int const ii = i - row_offset;
    int const jj = j - col_offset;
    if (ii < 0 || ii >= nxy || jj < 0 || jj >= nxy) {
        out[static_cast<size_t>(i) * ny_out + j] = 0.0;
        return;
    }

    dreal const* const intensity =
        intensity_batch + static_cast<size_t>(batch_idx) * nr;
    int const rmax = min(
        static_cast<int>(ceil((r_min + nr * dr) / dxy)),
        nxy / 2
    );
    int const offset = nxy / 2 - rmax;
    int const local_i = ii - offset;
    int const local_j = jj - offset;
    if (
        local_i < 0 || local_i >= 2 * rmax
        || local_j < 0 || local_j >= 2 * rmax
    ) {
        out[static_cast<size_t>(i) * ny_out + j] = 0.0;
        return;
    }

    dreal const sr_to_px = dxy * dxy;
    if (ii == nxy / 2 && jj == nxy / 2) {
        int const inner_index =
            static_cast<int>(floor((dxy / 2.0 - r_min) / dr));
        dreal flux = 0.0;
        for (int k = 1; k < inner_index; ++k) {
            flux += (r_min + dr * k) * intensity[k];
        }
        flux *= 2.0;
        flux += r_min * intensity[0]
            + (r_min + inner_index * dr) * intensity[inner_index];
        flux *= dr;

        dreal const interpolated = (
            intensity[inner_index + 1] - intensity[inner_index]
        ) / dr * (
            dxy / 2.0 - (r_min + dr * inner_index)
        ) + intensity[inner_index];
        flux += (
            (r_min + inner_index * dr) * intensity[inner_index]
            + dxy / 2.0 * interpolated
        ) * (
            dxy / 2.0 - (r_min + inner_index * dr)
        );
        dreal const area =
            (dxy / 2.0) * (dxy / 2.0) - r_min * r_min;
        out[static_cast<size_t>(i) * ny_out + j] =
            sr_to_px * flux / area;
        return;
    }

    dreal const x = (rmax - local_j) * dxy;
    dreal const y = (rmax - local_i) * dxy;
    dreal const cos_inc = cos(inc_batch[batch_idx]);
    dreal const radius = sqrt(
        (x / cos_inc) * (x / cos_inc) + y * y
    );
    int const radial_index = max(
        static_cast<int>(floor((radius - r_min) / dr)),
        0
    );
    dreal value = 0.0;
    if (radial_index <= nr - 2) {
        value = sr_to_px * (
            intensity[radial_index]
            + (radius - radial_index * dr - r_min)
                * (intensity[radial_index + 1] - intensity[radial_index])
                / dr
        );
    }
    out[static_cast<size_t>(i) * ny_out + j] = value;
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
    CudaMemory<dreal> u_d;
    CudaMemory<dreal> v_d;
    CudaMemory<dreal> vis_obs_re_d;
    CudaMemory<dreal> vis_obs_im_d;
    CudaMemory<dreal> weights_d;

    CudaMemory<dreal> urot_d;
    CudaMemory<dreal> vrot_d;
    CudaMemory<dcomplex> vis_int_d;
    CudaMemory<dcomplex> data_d;
    std::vector<dreal> batch_inc_h;
    std::vector<dreal> batch_pa_h;
    std::vector<dreal> batch_dRA_h;
    std::vector<dreal> batch_dDec_h;
    std::vector<dreal> batch_cos_pa_h;
    std::vector<dreal> batch_sin_pa_h;
    std::vector<dreal> batch_dRArot_h;
    std::vector<dreal> batch_dDecrot_h;
    std::vector<dreal> batch_gauss_h;
    std::vector<dreal> batch_ring_h;
    std::vector<dreal> batch_arc_h;
    // Batch allocations and plans are retained because cudaMalloc and cuFFT
    // planning are significant costs in repeated optimizer/MCMC evaluations.
    std::unique_ptr<CudaMemory<dreal>> batch_model_images_d;
    std::unique_ptr<CudaMemory<dcomplex>> batch_fft_images_d;
    std::unique_ptr<CudaMemory<dreal>> batch_inc_d;
    std::unique_ptr<CudaMemory<dreal>> batch_cos_pa_d;
    std::unique_ptr<CudaMemory<dreal>> batch_sin_pa_d;
    std::unique_ptr<CudaMemory<dreal>> batch_dRArot_d;
    std::unique_ptr<CudaMemory<dreal>> batch_dDecrot_d;
    std::unique_ptr<CudaMemory<dreal>> batch_chi2_d;
    std::unique_ptr<CudaMemory<dreal>> batch_gauss_params_d;
    std::unique_ptr<CudaMemory<dreal>> batch_ring_params_d;
    std::unique_ptr<CudaMemory<dreal>> batch_arc_params_d;
    std::unique_ptr<CudaMemory<dreal>> batch_intensity_d;
    int batch_workspace_capacity;
    int batch_gauss_capacity;
    int batch_ring_capacity;
    int batch_arc_capacity;
    int batch_intensity_capacity;
    int cached_fft_chunk_request;
    int cached_fft_chunk_size;
    cufftHandle batch_fft_plan;
    int batch_fft_plan_size;
    bool batch_fft_plan_initialized;

    Chi2ImageContext(int nx_, int ny_, int nd_, const dreal* u, const dreal* v,
                     const dreal* vis_obs_re, const dreal* vis_obs_im, const dreal* weights,
                     int backend_in, dreal nufft_oversample_in)
        : nx(nx_), ny(ny_), nd(nd_), requested_backend(backend_in),
          backend(resolve_image_backend(nx_, ny_, nd_, backend_in)),
          nufft_oversample(clamp_nufft_oversample(nufft_oversample_in)),
          work_nx(backend == galario::BACKEND_NUFFT ? resolve_padded_size(nx_, nufft_oversample) : nx_),
          work_ny(work_nx), work_ncol(work_ny / 2 + 1),
          model_image_h(static_cast<size_t>(work_nx) * work_ny),
          u_d(nd_, u), v_d(nd_, v),
          vis_obs_re_d(nd_, vis_obs_re), vis_obs_im_d(nd_, vis_obs_im), weights_d(nd_, weights),
          urot_d(nd_), vrot_d(nd_), vis_int_d(nd_), data_d(work_nx * work_ncol),
          batch_workspace_capacity(0), batch_gauss_capacity(0), batch_ring_capacity(0), batch_arc_capacity(0),
          batch_intensity_capacity(0),
          cached_fft_chunk_request(0), cached_fft_chunk_size(0),
          batch_fft_plan_size(0), batch_fft_plan_initialized(false) {}

    ~Chi2ImageContext() {
        if (batch_fft_plan_initialized) {
            cufftDestroy(batch_fft_plan);
        }
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

void fft_d(int nx, int ny, dcomplex* data_d) {
     cufftHandle plan;

     /* Create a 2D FFT plan and execute it. */
     // TODO: find a way to store the plan
     CUFFTCheck(cufftPlan2d(&plan, nx, ny, CUFFTTYPE));
     CUFFTCheck(CUFFTEXEC(plan, reinterpret_cast<dreal*>(data_d), data_d));

     // cufft calls are asynchronous but in default stream
     CCheck(cudaDeviceSynchronize());
     CUFFTCheck(cufftDestroy(plan));
}

void fft_batch_d(int nx, int ny, int batch_size, dreal* input_d, dcomplex* output_d) {
     cufftHandle plan;
     int n[2] = {nx, ny};
     int inembed[2] = {nx, ny};
     int onembed[2] = {nx, ny / 2 + 1};

     CUFFTCheck(cufftPlanMany(&plan, 2, n,
                              inembed, 1, nx * ny,
                              onembed, 1, nx * (ny / 2 + 1),
                              CUFFTTYPE, batch_size));
     CUFFTCheck(CUFFTEXEC(plan, input_d, output_d));

     CCheck(cudaDeviceSynchronize());
     CUFFTCheck(cufftDestroy(plan));
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
__host__ __device__
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

__host__ __device__
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
__global__ void shift_d(const int nx, const int ny, dcomplex* const __restrict__ data) {

    dreal* a = reinterpret_cast<dreal*>(data);

    // indices
    int const x0 = blockDim.x * blockIdx.x + threadIdx.x;
    int const y0 = blockDim.y * blockIdx.y + threadIdx.y;

    // stride
    int const sx = blockDim.x * gridDim.x;
    int const sy = blockDim.y * gridDim.y;

    for (auto x = x0; x < nx/2; x += sx) {
        for (auto y = y0; y < ny/2; y += sy) {
            shift_core(x, y, nx, ny, a);
        }
    }
}

__global__ void shift_real_batch_d(int const nx, int const ny, int const batch_size,
                                   dreal* const __restrict__ data) {
    int const batch_idx = blockIdx.z;
    if (batch_idx >= batch_size) {
        return;
    }

    dreal* const batch_data = data + static_cast<size_t>(batch_idx) * nx * ny;
    int const x0 = blockDim.x * blockIdx.x + threadIdx.x;
    int const y0 = blockDim.y * blockIdx.y + threadIdx.y;
    int const sx = blockDim.x * gridDim.x;
    int const sy = blockDim.y * gridDim.y;

    for (auto x = x0; x < nx / 2; x += sx) {
        for (auto y = y0; y < ny / 2; y += sy) {
            shift_real_core(x, y, nx, ny, batch_data);
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
__host__ __device__
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
__global__ void shift_axis0_d(int const nrow, int const ncol, dcomplex* const __restrict__ matrix) {
    // indices
    int const x0 = blockDim.x * blockIdx.x + threadIdx.x;
    int const y0 = blockDim.y * blockIdx.y + threadIdx.y;

    // stride
    int const sx = blockDim.x * gridDim.x;
    int const sy = blockDim.y * gridDim.y;

    for (auto x = x0; x < nrow/2; x += sx) {
        for (auto y = y0; y < ncol; y += sy) {
            shift_axis0_core(x, y, nrow, ncol, matrix);
        }
    }
}

__global__ void shift_axis0_batch_d(int const nrow, int const ncol, int const batch_size,
                                    dcomplex* const __restrict__ matrix) {
    int const batch_idx = blockIdx.z;
    if (batch_idx >= batch_size) {
        return;
    }

    dcomplex* const batch_matrix = matrix + static_cast<size_t>(batch_idx) * nrow * ncol;
    int const x0 = blockDim.x * blockIdx.x + threadIdx.x;
    int const y0 = blockDim.y * blockIdx.y + threadIdx.y;
    int const sx = blockDim.x * gridDim.x;
    int const sy = blockDim.y * gridDim.y;

    for (auto x = x0; x < nrow / 2; x += sx) {
        for (auto y = y0; y < ncol; y += sy) {
            shift_axis0_core(x, y, nrow, ncol, batch_matrix);
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
__host__ __device__
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

__global__ void interpolate_d(int const nrow, int const ncol, const dcomplex* const __restrict__ data, dreal const v_origin, int const nd, const dreal* const u, const dreal* const v, dreal const duv, dcomplex* const __restrict__ vis_int)
{
    //index
    int const idx_0 = blockDim.x * blockIdx.x + threadIdx.x;

    // stride
    int const sx = blockDim.x * gridDim.x;

    for (auto idx = idx_0; idx < nd; idx += sx) {
        vis_int[idx] = interpolate_core(nrow, ncol, data, v_origin, u[idx], v[idx], duv);
    }
}

namespace galario_internal {

void fft2d_impl(int nx, int ny, void* raw_data) {
    dcomplex* data = static_cast<dcomplex*>(raw_data);
    CHECK_INPUTXY(nx, ny);
    CudaMemory<dcomplex> data_d(nx * (ny / 2 + 1), data);
    fft_d(nx, ny, data_d.ptr);
    data_d.Retrieve(data);
}

void fftshift_impl(int nx, int ny, void* raw_data) {
    dcomplex* data = static_cast<dcomplex*>(raw_data);
    CHECK_INPUTXY(nx, ny);
    CudaMemory<dcomplex> data_d(nx * (ny / 2 + 1), data);
    shift_d<<<
        dim3(nx / 2 / tpb + 1, ny / 2 / tpb + 1),
        dim3(tpb, tpb)
    >>>(nx, ny, data_d.ptr);
    CCheck(cudaDeviceSynchronize());
    data_d.Retrieve(data);
}

void fftshift_axis0_impl(int nrow, int ncol, void* raw_matrix) {
    dcomplex* matrix = static_cast<dcomplex*>(raw_matrix);
    CHECK_INPUT(nrow);
    CudaMemory<dcomplex> matrix_d(nrow * ncol, matrix);
    shift_axis0_d<<<
        dim3(nrow / 2 / tpb + 1, ncol / tpb + 1),
        dim3(tpb, tpb)
    >>>(nrow, ncol, matrix_d.ptr);
    CCheck(cudaDeviceSynchronize());
    matrix_d.Retrieve(matrix);
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
    CudaMemory<dcomplex> data_d(nrow * ncol, data);
    CudaMemory<dreal> u_d(nd, u);
    CudaMemory<dreal> v_d(nd, v);
    CudaMemory<dcomplex> vis_int_d(nd);

    int const nthreads = tpb * tpb;
    interpolate_d<<<nd / nthreads + 1, nthreads>>>(
        nrow,
        ncol,
        data_d.ptr,
        v_origin,
        nd,
        u_d.ptr,
        v_d.ptr,
        duv,
        vis_int_d.ptr
    );
    CCheck(cudaDeviceSynchronize());
    vis_int_d.Retrieve(vis_int);
}

}



__host__ __device__
inline void uv_rotate_core(dreal cos_PA, dreal sin_PA, dreal x, dreal y, dreal& xrot, dreal& yrot);

__host__ __device__
inline dcomplex phase_from_angle_core(dreal const angle);

__host__ __device__
inline dcomplex sample_image_visibility_core(int const nx, int const ny, const dreal* const image, dreal const v_origin,
                                             dreal const dxy, dreal const cos_PA, dreal const sin_PA,
                                             dreal const dRArot, dreal const dDecrot, dreal const u, dreal const v);

// APPLY_PHASE TO SAMPLED POINTS //
__host__ __device__
inline void apply_phase_sampled_core(int const idx_x, const dreal* const u, const dreal* const v, dcomplex* const __restrict__ vis_int, dreal const dRA, dreal const dDec) {

    dreal const angle = u[idx_x]*dRA + v[idx_x]*dDec;

    dcomplex const phase = dcomplex{dreal(cos(angle)), dreal(sin(angle))};

    vis_int[idx_x] = CMPLXMUL(vis_int[idx_x], phase);
}

__global__ void apply_phase_sampled_d(dreal dRA, dreal dDec, int const nd, const dreal* const u, const dreal* const v, dcomplex* __restrict__ vis_int) {

    if ((dRA==0.) && (dDec==0.)) {
        return;
    }

    dRA *= 2.*(dreal)M_PI;
    dDec *= 2.*(dreal)M_PI;

    //index
    int const idx_x0 = blockDim.x * blockIdx.x + threadIdx.x;

    // stride
    int const sx = blockDim.x * gridDim.x;

    for (auto x = idx_x0; x < nd; x += sx) {
        apply_phase_sampled_core(x, u, v, vis_int, dRA, dDec);
    }
}

__global__ void interpolate_chi2_batch_d(int const nrow, int const ncol,
                                         const dcomplex* const __restrict__ data_batch,
                                         int const batch_size, dreal const v_origin, int const nd,
                                         const dreal* const __restrict__ u,
                                         const dreal* const __restrict__ v,
                                         dreal const duv,
                                         bool const conjugate_model,
                                         const dreal* const __restrict__ cos_pa_batch,
                                         const dreal* const __restrict__ sin_pa_batch,
                                         const dreal* const __restrict__ dRArot_batch,
                                         const dreal* const __restrict__ dDecrot_batch,
                                         const dreal* const __restrict__ vis_obs_re,
                                         const dreal* const __restrict__ vis_obs_im,
                                         const dreal* const __restrict__ weights,
                                         dreal* const __restrict__ chi2_out) {
    extern __shared__ dreal shared_sum[];

    int const batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) {
        return;
    }

    int const tid = threadIdx.x;
    int const idx_0 = blockDim.x * blockIdx.x + tid;
    int const sx = blockDim.x * gridDim.x;
    size_t const image_stride = static_cast<size_t>(nrow) * ncol;
    const dcomplex* const image = data_batch + static_cast<size_t>(batch_idx) * image_stride;
    dreal const cos_PA = cos_pa_batch[batch_idx];
    dreal const sin_PA = sin_pa_batch[batch_idx];
    dreal const dRArot = dRArot_batch[batch_idx];
    dreal const dDecrot = dDecrot_batch[batch_idx];

    dreal sum = 0.;
    for (auto idx = idx_0; idx < nd; idx += sx) {
        dreal urot;
        dreal vrot;
        uv_rotate_core(cos_PA, sin_PA, u[idx], v[idx], urot, vrot);

        dcomplex vis_model = interpolate_core(nrow, ncol, image, v_origin, urot, vrot, duv);
        if ((dRArot != 0.) || (dDecrot != 0.)) {
            dreal const angle = 2. * (dreal)M_PI * (urot * dRArot + vrot * dDecrot);
            vis_model = CMPLXMUL(vis_model, phase_from_angle_core(angle));
        }
        if (conjugate_model) {
            vis_model = CMPLXCONJ(vis_model);
        }

        dcomplex const diff = CMPLXSUB(vis_model, dcomplex{vis_obs_re[idx], vis_obs_im[idx]});
        dreal const diff_re = cmplx_real_part(diff);
        dreal const diff_im = cmplx_imag_part(diff);
        sum += weights[idx] * (diff_re * diff_re + diff_im * diff_im);
    }

    shared_sum[tid] = sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            shared_sum[tid] += shared_sum[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&chi2_out[batch_idx], shared_sum[0]);
    }
}

__global__ void direct_chi2_batch_d(int const nx, int const ny,
                                    const dreal* const __restrict__ image_batch,
                                    int const batch_size, dreal const v_origin, dreal const dxy,
                                    int const nd, const dreal* const __restrict__ u,
                                    const dreal* const __restrict__ v,
                                    const dreal* const __restrict__ cos_pa_batch,
                                    const dreal* const __restrict__ sin_pa_batch,
                                    const dreal* const __restrict__ dRArot_batch,
                                    const dreal* const __restrict__ dDecrot_batch,
                                    const dreal* const __restrict__ vis_obs_re,
                                    const dreal* const __restrict__ vis_obs_im,
                                    const dreal* const __restrict__ weights,
                                    dreal* const __restrict__ chi2_out) {
    extern __shared__ dreal shared_sum[];

    int const batch_idx = blockIdx.y;
    if (batch_idx >= batch_size) {
        return;
    }

    int const tid = threadIdx.x;
    int const idx_0 = blockDim.x * blockIdx.x + tid;
    int const sx = blockDim.x * gridDim.x;
    const dreal* const image = image_batch + static_cast<size_t>(batch_idx) * nx * ny;
    dreal const cos_PA = cos_pa_batch[batch_idx];
    dreal const sin_PA = sin_pa_batch[batch_idx];
    dreal const dRArot = dRArot_batch[batch_idx];
    dreal const dDecrot = dDecrot_batch[batch_idx];

    dreal sum = 0.;
    for (auto idx = idx_0; idx < nd; idx += sx) {
        dcomplex const vis_model = sample_image_visibility_core(nx, ny, image, v_origin, dxy,
                                                                cos_PA, sin_PA, dRArot, dDecrot,
                                                                u[idx], v[idx]);
        dcomplex const diff = CMPLXSUB(vis_model, dcomplex{vis_obs_re[idx], vis_obs_im[idx]});
        dreal const diff_re = cmplx_real_part(diff);
        dreal const diff_im = cmplx_imag_part(diff);
        sum += weights[idx] * (diff_re * diff_re + diff_im * diff_im);
    }

    shared_sum[tid] = sum;
    __syncthreads();

    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (tid < offset) {
            shared_sum[tid] += shared_sum[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(&chi2_out[batch_idx], shared_sum[0]);
    }
}

namespace galario {
void apply_phase_sampled(dreal dRA, dreal dDec, int const nd, const dreal* const u, const dreal* const v, dcomplex* const __restrict__ vis_int) {

     CudaMemory<dreal> u_d(nd, u);
     CudaMemory<dreal> v_d(nd, v);
     CudaMemory<dcomplex> vis_int_d(nd, vis_int);

     auto const nthreads = tpb * tpb;
     apply_phase_sampled_d<<<nd/nthreads+1, nthreads>>>(dRA, dDec, nd, u_d.ptr, v_d.ptr, vis_int_d.ptr);

     CCheck(cudaDeviceSynchronize());
     vis_int_d.Retrieve(vis_int);
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
__host__ __device__
inline void uv_rotate_core(dreal cos_PA, dreal sin_PA, const dreal u, const dreal v, dreal& urot, dreal& vrot) {

    urot = u * cos_PA - v * sin_PA;
    vrot = u * sin_PA + v * cos_PA;

}

__global__ void uv_rotate_d(dreal cos_PA, dreal sin_PA, int const nd, const dreal* const u, const dreal* const v, dreal* const urot, dreal* vrot) {
    //index
    int const idx_x0 = blockDim.x * blockIdx.x + threadIdx.x;

    // stride
    int const sx = blockDim.x * gridDim.x;

    for (auto i = idx_x0; i < nd; i += sx) {
        uv_rotate_core(cos_PA, sin_PA, u[i], v[i], urot[i], vrot[i]);
    }
}

namespace galario {

void uv_rotate(dreal PA, dreal dRA, dreal dDec, dreal* dRArot, dreal* dDecrot, int const nd, const dreal* const u, const dreal* const v,
                       dreal* const urot, dreal* const vrot) {
     CudaMemory<dreal> u_d(nd, u);
     CudaMemory<dreal> v_d(nd, v);

     CudaMemory<dreal> urot_d(nd);
     CudaMemory<dreal> vrot_d(nd);

     if (PA==0.) {
        *dRArot = dRA;
        *dDecrot = dDec;
        cudaMemcpy(urot_d.ptr, u_d.ptr, u_d.nbytes, cudaMemcpyDeviceToDevice);
        cudaMemcpy(vrot_d.ptr, v_d.ptr, v_d.nbytes, cudaMemcpyDeviceToDevice);
     } else {
        const dreal cos_PA = cos(PA);
        const dreal sin_PA = sin(PA);

        auto const nthreads = tpb * tpb;
        uv_rotate_d<<<nd/nthreads +1, nthreads>>>(cos_PA, sin_PA, nd, u_d.ptr, v_d.ptr, urot_d.ptr, vrot_d.ptr);
        uv_rotate_core(cos_PA, sin_PA, dRA, dDec, *dRArot, *dDecrot);
     }
     CCheck(cudaDeviceSynchronize());
     urot_d.Retrieve(urot);
     vrot_d.Retrieve(vrot);
}

void _uv_rotate(dreal PA, dreal dRA, dreal dDec, void* dRArot, void* dDecrot, int nd, void* const u,
                                  void* const v, void* const urot, void* const vrot) {
    uv_rotate(PA, dRA, dDec, static_cast<dreal*>(dRArot), static_cast<dreal*>(dDecrot), nd, static_cast<dreal*>(u),
                                static_cast<dreal*>(v), static_cast<dreal*>(urot), static_cast<dreal*>(vrot));
}
}

__host__ __device__
inline dcomplex phase_from_angle_core(dreal const angle) {
    return dcomplex{dreal(cos(angle)), dreal(sin(angle))};
}

__host__ __device__
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

__global__ void sample_image_direct_d(int const nx, int const ny, const dreal* const __restrict__ image,
                                      dreal const v_origin, dreal const dxy, dreal const cos_PA,
                                      dreal const sin_PA, dreal const dRArot, dreal const dDecrot,
                                      int const nd, const dreal* const __restrict__ u,
                                      const dreal* const __restrict__ v,
                                      dcomplex* const __restrict__ vis_int) {
    int const idx_0 = blockDim.x * blockIdx.x + threadIdx.x;
    int const sx = blockDim.x * gridDim.x;

    for (auto idx = idx_0; idx < nd; idx += sx) {
        vis_int[idx] = sample_image_visibility_core(nx, ny, image, v_origin, dxy, cos_PA, sin_PA,
                                                    dRArot, dDecrot, u[idx], v[idx]);
    }
}

inline void sample_d(int nx, int ny, dcomplex* data_d, const dreal v_origin, dreal dRA, dreal dDec, int nd, dreal duv, const dreal PA, const dreal* u, const dreal* v, dcomplex* vis_int_d)
{
    GPUTimer t_start;

    int const ncol = ny/2+1;

    // ################################
    // ### ALLOCATION, INITIALIZATION ###
    // ################################

    /* async memory copy:, see issue https://github.com/mtazzari/galario/issues/40
       TODO copy memory asynchronously or create streams to define dependencies
       use nonzero cudaStream_t
       kernel<<< blocks, threads, bytes=0, stream =! 0>>>();

       all cufft calls are asynchronous, can specify the stream explicitly (cf. doc)
       same for cublas
       draw dependencies on paper: first thing is to do fft while other data is transferred
    */

    GPUTimer t;
    CudaMemory<dreal> u_d(nd, u);
    CudaMemory<dreal> v_d(nd, v);

    CudaMemory<dreal> urot_d(nd);
    CudaMemory<dreal> vrot_d(nd);
    t.Elapsed("sample::copy_uv_H->D");

    auto const nthreads = tpb * tpb;
    dreal dRArot = 0.;
    dreal dDecrot = 0.;

    // ################################
    // ########### KERNELS ############
    // ################################
    // rotate uv points
     if (PA==0.) {
        dRArot = dRA;
        dDecrot = dDec;
        cudaMemcpy(urot_d.ptr, u_d.ptr, u_d.nbytes, cudaMemcpyDeviceToDevice);
        cudaMemcpy(vrot_d.ptr, v_d.ptr, u_d.nbytes, cudaMemcpyDeviceToDevice);
        t.Elapsed("sample::copy_uvrot_D->D");
     } else {
        const dreal cos_PA = cos(PA);
        const dreal sin_PA = sin(PA);

        uv_rotate_d<<<nd/nthreads +1, nthreads>>>(cos_PA, sin_PA, nd, u_d.ptr, v_d.ptr, urot_d.ptr, vrot_d.ptr);
        uv_rotate_core(cos_PA, sin_PA, dRA, dDec, dRArot, dDecrot);
        t.Elapsed("sample::uv_rotate");
     }

    // Kernel for shift --> FFT --> shift
    shift_d<<<dim3(nx/2/tpb+1, ny/2/tpb+1), dim3(tpb, tpb)>>>(nx, ny, data_d); t.Elapsed("sample::1st_shift");
    fft_d(nx, ny, (dcomplex*) data_d); t.Elapsed("sample::FFT");
    shift_axis0_d<<<dim3(nx/2/tpb+1, ncol/2/tpb+1), dim3(tpb, tpb)>>>(nx, ncol, data_d); t.Elapsed("sample::2nd_shift");

    // oversubscribe blocks because we don't know if #(data points) divisible by nthreads
    interpolate_d<<<nd / nthreads + 1, nthreads>>>(nx, ncol, data_d, v_origin, nd, urot_d.ptr, vrot_d.ptr, duv, vis_int_d); t.Elapsed("sample::interpolate");

    // apply phase to the sampled points
    apply_phase_sampled_d<<<nd / nthreads + 1, nthreads>>>(dRArot, dDecrot, nd, urot_d.ptr, vrot_d.ptr, vis_int_d); t.Elapsed("sample::apply_phase_sampled");

    t_start.Elapsed("sample_tot");
}

__global__ void conjugate_vis_d(int const nd, dcomplex* const __restrict__ vis) {
    int const idx_0 = blockDim.x * blockIdx.x + threadIdx.x;
    int const sx = blockDim.x * gridDim.x;

    for (auto idx = idx_0; idx < nd; idx += sx) {
        vis[idx] = CMPLXCONJ(vis[idx]);
    }
}

inline void conjugate_vis_inplace_d(int const nd, dcomplex* vis) {
    auto const nthreads = tpb * tpb;
    conjugate_vis_d<<<nd / nthreads + 1, nthreads>>>(nd, vis);
}

inline void sample_d_cached(galario::Chi2ImageContext* context, const dreal v_origin, dreal dRA, dreal dDec, dreal duv, const dreal PA)
{
    GPUTimer t_start;
    GPUTimer t;

    auto const nthreads = tpb * tpb;
    if (context->backend == galario::BACKEND_DFT) {
        dreal const cos_PA = cos(PA);
        dreal const sin_PA = sin(PA);
        dreal dRArot;
        dreal dDecrot;
        uv_rotate_core(cos_PA, sin_PA, dRA, dDec, dRArot, dDecrot);

        sample_image_direct_d<<<context->nd / nthreads + 1, nthreads>>>(context->nx, context->ny,
                                                                         reinterpret_cast<dreal*>(context->data_d.ptr),
                                                                         v_origin, 1. / (duv * context->nx),
                                                                         cos_PA, sin_PA, dRArot, dDecrot,
                                                                         context->nd, context->u_d.ptr, context->v_d.ptr,
                                                                         context->vis_int_d.ptr);
        t.Elapsed("sample_cached::direct_dft");
    } else {
        dreal dRArot = 0.;
        dreal dDecrot = 0.;
        dreal const duv_backend = duv * context->nx / context->work_nx;

        if (PA==0.) {
            dRArot = dRA;
            dDecrot = dDec;
            CCheck(cudaMemcpy(context->urot_d.ptr, context->u_d.ptr, context->u_d.nbytes, cudaMemcpyDeviceToDevice));
            CCheck(cudaMemcpy(context->vrot_d.ptr, context->v_d.ptr, context->v_d.nbytes, cudaMemcpyDeviceToDevice));
            t.Elapsed("sample_cached::copy_uvrot_D->D");
        } else {
            const dreal cos_PA = cos(PA);
            const dreal sin_PA = sin(PA);

            uv_rotate_d<<<context->nd/nthreads +1, nthreads>>>(cos_PA, sin_PA, context->nd, context->u_d.ptr, context->v_d.ptr, context->urot_d.ptr, context->vrot_d.ptr);
            uv_rotate_core(cos_PA, sin_PA, dRA, dDec, dRArot, dDecrot);
            t.Elapsed("sample_cached::uv_rotate");
        }

        shift_d<<<dim3(context->work_nx/2/tpb+1, context->work_ny/2/tpb+1), dim3(tpb, tpb)>>>(context->work_nx, context->work_ny, context->data_d.ptr); t.Elapsed("sample_cached::1st_shift");
        fft_d(context->work_nx, context->work_ny, context->data_d.ptr); t.Elapsed("sample_cached::FFT");
        shift_axis0_d<<<dim3(context->work_nx/2/tpb+1, context->work_ncol/2/tpb+1), dim3(tpb, tpb)>>>(context->work_nx, context->work_ncol, context->data_d.ptr); t.Elapsed("sample_cached::2nd_shift");
        interpolate_d<<<context->nd / nthreads + 1, nthreads>>>(context->work_nx, context->work_ncol, context->data_d.ptr, v_origin, context->nd, context->urot_d.ptr, context->vrot_d.ptr, duv_backend, context->vis_int_d.ptr); t.Elapsed("sample_cached::interpolate");
        apply_phase_sampled_d<<<context->nd / nthreads + 1, nthreads>>>(dRArot, dDecrot, context->nd, context->urot_d.ptr, context->vrot_d.ptr, context->vis_int_d.ptr); t.Elapsed("sample_cached::apply_phase_sampled");
        if (context->backend == galario::BACKEND_NUFFT) {
            conjugate_vis_inplace_d(context->nd, context->vis_int_d.ptr);
            t.Elapsed("sample_cached::conjugate");
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
        GPUTimer t_total;
        dreal const cos_PA = cos(PA);
        dreal const sin_PA = sin(PA);
        dreal dRArot;
        dreal dDecrot;
        uv_rotate_core(cos_PA, sin_PA, dRA, dDec, dRArot, dDecrot);

        CudaMemory<dreal> image_d(nx * ny, realdata);
        CudaMemory<dreal> u_d(nd, u);
        CudaMemory<dreal> v_d(nd, v);
        CudaMemory<dcomplex> vis_int_d(nd);
        auto const nthreads = tpb * tpb;

        sample_image_direct_d<<<nd / nthreads + 1, nthreads>>>(nx, ny, image_d.ptr, v_origin, 1. / (duv * nx),
                                                               cos_PA, sin_PA, dRArot, dDecrot, nd, u_d.ptr, v_d.ptr,
                                                               vis_int_d.ptr);

        CCheck(cudaDeviceSynchronize());
        vis_int_d.Retrieve(vis_int);
        t_total.Elapsed("sample_image_tot");
    } else {
        int const work_nx = resolved_backend == galario::BACKEND_NUFFT ? resolve_padded_size(nx, nufft_oversample) : nx;
        int const work_ny = work_nx;
        dreal const duv_backend = duv * nx / work_nx;
        CudaMemory<dcomplex> data_d(work_nx * (work_ny / 2 + 1));
        CudaMemory<dcomplex> vis_int_d(nd);
        if (resolved_backend == galario::BACKEND_NUFFT) {
            auto padded_real = pad_real_image(nx, ny, realdata, work_nx, work_ny);
            copy_input_d_into(work_nx, work_ny, padded_real.data(), data_d.ptr);
        } else {
            copy_input_d_into(nx, ny, realdata, data_d.ptr);
        }
        sample_d(work_nx, work_ny, data_d.ptr, v_origin, dRA, dDec, nd, duv_backend, PA, u, v, vis_int_d.ptr);
        if (resolved_backend == galario::BACKEND_NUFFT) {
            conjugate_vis_inplace_d(nd, vis_int_d.ptr);
        }
        vis_int_d.Retrieve(vis_int);
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
__host__ __device__
inline void diff_weighted_core(int const idx_x, int const nd, const dreal* const __restrict__ vis_obs_re,
                               const dreal * const __restrict__ vis_obs_im, const dcomplex* const __restrict__ vis_int,
                               const dreal* const __restrict__ weights, dcomplex& res)
{
    dcomplex const vis_obs_cmplx = dcomplex { vis_obs_re[idx_x], vis_obs_im[idx_x] };
    dcomplex const sqrt_w_cmplx = dcomplex { SQRT(weights[idx_x]), 0.0 } ;
    res = CMPLXSUB(vis_int[idx_x], vis_obs_cmplx);
    res = CMPLXMUL(res, sqrt_w_cmplx);
}

__global__ void diff_weighted_d
(int const nd, const dreal* const __restrict__ vis_obs_re, const dreal* const __restrict__ vis_obs_im,  dcomplex* const __restrict__ vis_int, const dreal* const __restrict__ weights)
{
    //index
    int const idx_x0 = blockDim.x * blockIdx.x + threadIdx.x;

    // stride
    int const sidx_x = blockDim.x * gridDim.x;

    for (auto idx_x = idx_x0; idx_x < nd; idx_x += sidx_x) {
        // vis_int copied before, so it is ok to overwrite inside diff_weighted_core
        diff_weighted_core(idx_x, nd, vis_obs_re, vis_obs_im, vis_int, weights, vis_int[idx_x]);
    }
}

dreal reduce_chi2_d
(int nd, const dreal* const __restrict__ vis_obs_re, const dreal* const __restrict__ vis_obs_im, dcomplex * const __restrict__ vis_int, const dreal* const __restrict__ weights)
{
    GPUTimer t_start, t;

    auto const nthreads = tpb * tpb;

    /* compute weighted difference */
    diff_weighted_d<<<nd / nthreads + 1, nthreads>>>(nd, vis_obs_re, vis_obs_im, vis_int, weights);
    t.Elapsed("reduce_chi2::diff_weighted");

    // only device pointers!
    // compute the Euclidean norm
    dreal chi2 = 0;
    CUBLASNRM2(cublas_handle(), nd, vis_int, 1, &chi2);
    // but we want the square of the norm
    chi2 *= chi2;
    t.Elapsed("reduce_chi2::reduction");
    t_start.Elapsed("reduce_chi2_tot");

    return chi2;
}

namespace galario {
dreal reduce_chi2(int nd, const dreal* vis_obs_re, const dreal* vis_obs_im, const dcomplex* vis_int, const dreal* weights) {
     CPUTimer t_start;
     dreal chi2 = 0.;

    /* allocate and copy */
     CudaMemory<dreal> vis_obs_re_d(nd, vis_obs_re);
     CudaMemory<dreal> vis_obs_im_d(nd, vis_obs_im);
     CudaMemory<dcomplex> vis_int_d(nd, vis_int);
     CudaMemory<dreal> weights_d(nd, weights);

     chi2 = reduce_chi2_d(nd, vis_obs_re_d.ptr, vis_obs_im_d.ptr, vis_int_d.ptr, weights_d.ptr);
     t_start.Elapsed("reduce_chi2_tot");

     return chi2;
}

dreal _reduce_chi2(int nd, void* vis_obs_re, void* vis_obs_im, void* vis_int, void* weights) {
    return reduce_chi2(nd, static_cast<dreal*>(vis_obs_re), static_cast<dreal*>(vis_obs_im), static_cast<dcomplex*>(vis_int), static_cast<dreal*>(weights));
}

int ngpus()
{
    int num_devices = 0;
    CCheck(cudaGetDeviceCount(&num_devices));
    return num_devices;
}

void use_gpu(int device_id)
{
    CCheck(cudaSetDevice(device_id));
}

dreal chi2_image_from_context(Chi2ImageContext* context, const dreal* realdata, const dreal v_origin, dreal dRA, dreal dDec, dreal duv, dreal PA) {
    CPUTimer t_start;

    CHECK_INPUTXY(context->nx, context->ny);
    dreal chi2 = 0;
    if (context->backend == galario::BACKEND_DFT) {
        CCheck(cudaMemcpy(reinterpret_cast<dreal*>(context->data_d.ptr), realdata, sizeof(dreal) * context->nx * context->ny, cudaMemcpyHostToDevice));
    } else if (context->backend == galario::BACKEND_NUFFT) {
        auto padded_real = pad_real_image(context->nx, context->ny, realdata, context->work_nx, context->work_ny);
        copy_input_d_into(context->work_nx, context->work_ny, padded_real.data(), context->data_d.ptr);
    } else {
        copy_input_d_into(context->nx, context->ny, realdata, context->data_d.ptr);
    }
    sample_d_cached(context, v_origin, dRA, dDec, duv, PA);
    CCheck(cudaDeviceSynchronize());
    chi2 = reduce_chi2_d(context->nd, context->vis_obs_re_d.ptr, context->vis_obs_im_d.ptr, context->vis_int_d.ptr, context->weights_d.ptr);
    t_start.Elapsed("chi2_image_from_context_tot");
    flush_timing();

    return chi2;
}

namespace {
dreal chi2_image_from_context_rasterized(galario::Chi2ImageContext* context, const dreal v_origin,
                                        dreal dRA, dreal dDec, dreal duv, dreal PA) {
    dreal chi2 = 0;
    if (context->backend == galario::BACKEND_DFT) {
        CCheck(cudaMemcpy(reinterpret_cast<dreal*>(context->data_d.ptr), context->model_image_h.data(),
                          sizeof(dreal) * context->nx * context->ny, cudaMemcpyHostToDevice));
    } else {
        copy_input_d_into(context->work_nx, context->work_ny, context->model_image_h.data(), context->data_d.ptr);
    }
    sample_d_cached(context, v_origin, dRA, dDec, duv, PA);
    CCheck(cudaDeviceSynchronize());
    chi2 = reduce_chi2_d(context->nd, context->vis_obs_re_d.ptr, context->vis_obs_im_d.ptr,
                         context->vis_int_d.ptr, context->weights_d.ptr);
    return chi2;
}

int resolve_fft_batch_chunk_size(galario::Chi2ImageContext* context, int requested_batch_size) {
    if (context->cached_fft_chunk_request == requested_batch_size && context->cached_fft_chunk_size > 0) {
        return context->cached_fft_chunk_size;
    }

    if (requested_batch_size <= 1) {
        context->cached_fft_chunk_request = requested_batch_size;
        context->cached_fft_chunk_size = 1;
        return 1;
    }

    size_t free_mem = 0;
    size_t total_mem = 0;
    CCheck(cudaMemGetInfo(&free_mem, &total_mem));

    size_t const real_image_bytes = static_cast<size_t>(context->work_nx) * context->work_ny * sizeof(dreal);
    size_t const fft_image_bytes = static_cast<size_t>(context->work_nx) * context->work_ncol * sizeof(dcomplex);
    size_t const per_batch_bytes = real_image_bytes + fft_image_bytes;
    size_t const reserved_mem = free_mem / 2;
    size_t const gpu_limited_batch = std::max<size_t>(1, reserved_mem / std::max<size_t>(per_batch_bytes, 1));
    size_t const max_host_bytes = static_cast<size_t>(512) * 1024 * 1024;
    size_t const host_limited_batch = std::max<size_t>(1, max_host_bytes / std::max<size_t>(real_image_bytes, 1));

    int const resolved = static_cast<int>(std::max<size_t>(1, std::min<size_t>(requested_batch_size,
                                                                                std::min(gpu_limited_batch, host_limited_batch))));
    context->cached_fft_chunk_request = requested_batch_size;
    context->cached_fft_chunk_size = resolved;
    return resolved;
}

void ensure_fft_batch_plan(galario::Chi2ImageContext* context, int batch_size) {
    if (context->batch_fft_plan_initialized && context->batch_fft_plan_size == batch_size) {
        return;
    }

    if (context->batch_fft_plan_initialized) {
        CUFFTCheck(cufftDestroy(context->batch_fft_plan));
        context->batch_fft_plan_initialized = false;
    }

    int n[2] = {context->work_nx, context->work_ny};
    int inembed[2] = {context->work_nx, context->work_ny};
    int onembed[2] = {context->work_nx, context->work_ncol};
    CUFFTCheck(cufftPlanMany(&context->batch_fft_plan, 2, n,
                             inembed, 1, context->work_nx * context->work_ny,
                             onembed, 1, context->work_nx * context->work_ncol,
                             CUFFTTYPE, batch_size));
    context->batch_fft_plan_size = batch_size;
    context->batch_fft_plan_initialized = true;
}

void ensure_fft_batch_workspace(galario::Chi2ImageContext* context, int batch_size,
                                int gauss_stride, int ring_stride, int arc_stride) {
    if (context->batch_workspace_capacity < batch_size) {
        context->batch_model_images_d.reset(new CudaMemory<dreal>(static_cast<size_t>(batch_size) * context->work_nx * context->work_ny));
        context->batch_fft_images_d.reset(new CudaMemory<dcomplex>(static_cast<size_t>(batch_size) * context->work_nx * context->work_ncol));
        context->batch_inc_d.reset(new CudaMemory<dreal>(batch_size));
        context->batch_cos_pa_d.reset(new CudaMemory<dreal>(batch_size));
        context->batch_sin_pa_d.reset(new CudaMemory<dreal>(batch_size));
        context->batch_dRArot_d.reset(new CudaMemory<dreal>(batch_size));
        context->batch_dDecrot_d.reset(new CudaMemory<dreal>(batch_size));
        context->batch_chi2_d.reset(new CudaMemory<dreal>(batch_size));
        context->batch_workspace_capacity = batch_size;
        context->batch_inc_h.resize(batch_size);
        context->batch_cos_pa_h.resize(batch_size);
        context->batch_sin_pa_h.resize(batch_size);
        context->batch_dRArot_h.resize(batch_size);
        context->batch_dDecrot_h.resize(batch_size);
    }

    if (gauss_stride > 0 && context->batch_gauss_capacity < batch_size * gauss_stride) {
        context->batch_gauss_params_d.reset(new CudaMemory<dreal>(static_cast<size_t>(batch_size) * gauss_stride));
        context->batch_gauss_capacity = batch_size * gauss_stride;
    }
    if (ring_stride > 0 && context->batch_ring_capacity < batch_size * ring_stride) {
        context->batch_ring_params_d.reset(new CudaMemory<dreal>(static_cast<size_t>(batch_size) * ring_stride));
        context->batch_ring_capacity = batch_size * ring_stride;
    }
    if (arc_stride > 0 && context->batch_arc_capacity < batch_size * arc_stride) {
        context->batch_arc_params_d.reset(new CudaMemory<dreal>(static_cast<size_t>(batch_size) * arc_stride));
        context->batch_arc_capacity = batch_size * arc_stride;
    }
}

void chi2_image_from_context_components_batch_fft_like_d(galario::Chi2ImageContext* context, dreal dxy,
                                                   int batch_size,
                                                   int ngauss, const dreal* gauss_params_batch,
                                                   int nrings, const dreal* ring_params_batch,
                                                   int narcs, const dreal* arc_params_batch,
                                                   const dreal* inc_batch, const dreal v_origin,
                                                   const dreal* dRA_batch, const dreal* dDec_batch,
                                                   dreal duv, const dreal* PA_batch, dreal* chi2_out,
                                                   bool conjugate_model) {
    auto const gauss_stride = ngauss * 2;
    auto const ring_stride = nrings * 3;
    auto const arc_stride = narcs * 5;
    auto const work_nx = context->work_nx;
    auto const work_ny = context->work_ny;
    auto const work_ncol = context->work_ncol;
    auto const nthreads = tpb * tpb;
    auto const fft_blocks_x = context->nd / nthreads + 1;
    auto const duv_backend = duv * context->nx / context->work_nx;
    int const chunk_size = resolve_fft_batch_chunk_size(context, batch_size);

    std::fill_n(chi2_out, batch_size, dreal{0.});

    for (int start = 0; start < batch_size; start += chunk_size) {
        int const current_batch = std::min(chunk_size, batch_size - start);
        ensure_fft_batch_workspace(context, current_batch, gauss_stride, ring_stride, arc_stride);
        ensure_fft_batch_plan(context, current_batch);

        for (int local_idx = 0; local_idx < current_batch; ++local_idx) {
            int const batch_idx = start + local_idx;
            context->batch_inc_h[local_idx] = inc_batch[batch_idx];
            context->batch_cos_pa_h[local_idx] = cos(PA_batch[batch_idx]);
            context->batch_sin_pa_h[local_idx] = sin(PA_batch[batch_idx]);
            uv_rotate_core(context->batch_cos_pa_h[local_idx], context->batch_sin_pa_h[local_idx],
                           dRA_batch[batch_idx], dDec_batch[batch_idx],
                           context->batch_dRArot_h[local_idx], context->batch_dDecrot_h[local_idx]);
        }

        context->batch_inc_d->CopyFromHost(context->batch_inc_h.data(), current_batch);
        context->batch_cos_pa_d->CopyFromHost(context->batch_cos_pa_h.data(), current_batch);
        context->batch_sin_pa_d->CopyFromHost(context->batch_sin_pa_h.data(), current_batch);
        context->batch_dRArot_d->CopyFromHost(context->batch_dRArot_h.data(), current_batch);
        context->batch_dDecrot_d->CopyFromHost(context->batch_dDecrot_h.data(), current_batch);

        if (gauss_stride > 0) {
            context->batch_gauss_params_d->CopyFromHost(gauss_params_batch + static_cast<size_t>(start) * gauss_stride,
                                                        static_cast<size_t>(current_batch) * gauss_stride);
        }
        if (ring_stride > 0) {
            context->batch_ring_params_d->CopyFromHost(ring_params_batch + static_cast<size_t>(start) * ring_stride,
                                                       static_cast<size_t>(current_batch) * ring_stride);
        }
        if (arc_stride > 0) {
            context->batch_arc_params_d->CopyFromHost(arc_params_batch + static_cast<size_t>(start) * arc_stride,
                                                      static_cast<size_t>(current_batch) * arc_stride);
        }

        CCheck(cudaMemset(context->batch_chi2_d->ptr, 0, sizeof(dreal) * current_batch));
        rasterize_component_image_batch_d<<<dim3(work_ny / tpb + 1, work_nx / tpb + 1, current_batch), dim3(tpb, tpb)>>>(
            context->nx, context->ny, dxy,
            current_batch,
            ngauss, gauss_stride > 0 ? context->batch_gauss_params_d->ptr : nullptr,
            nrings, ring_stride > 0 ? context->batch_ring_params_d->ptr : nullptr,
            narcs, arc_stride > 0 ? context->batch_arc_params_d->ptr : nullptr,
            context->batch_inc_d->ptr,
            work_nx, work_ny, context->batch_model_images_d->ptr);
        shift_real_batch_d<<<dim3(work_nx / 2 / tpb + 1, work_ny / 2 / tpb + 1, current_batch), dim3(tpb, tpb)>>>(
            work_nx, work_ny, current_batch, context->batch_model_images_d->ptr);
        CUFFTCheck(CUFFTEXEC(context->batch_fft_plan, context->batch_model_images_d->ptr, context->batch_fft_images_d->ptr));
        shift_axis0_batch_d<<<dim3(work_nx / 2 / tpb + 1, work_ncol / tpb + 1, current_batch), dim3(tpb, tpb)>>>(
            work_nx, work_ncol, current_batch, context->batch_fft_images_d->ptr);
        interpolate_chi2_batch_d<<<dim3(fft_blocks_x, current_batch), nthreads, sizeof(dreal) * nthreads>>>(
            work_nx, work_ncol, context->batch_fft_images_d->ptr, current_batch, v_origin, context->nd,
            context->u_d.ptr, context->v_d.ptr, duv_backend, conjugate_model,
            context->batch_cos_pa_d->ptr, context->batch_sin_pa_d->ptr,
            context->batch_dRArot_d->ptr, context->batch_dDecrot_d->ptr,
            context->vis_obs_re_d.ptr, context->vis_obs_im_d.ptr, context->weights_d.ptr, context->batch_chi2_d->ptr);
        CCheck(cudaDeviceSynchronize());
        context->batch_chi2_d->RetrieveCount(chi2_out + start, current_batch);
    }
}

void chi2_image_from_context_components_batch_direct_d(galario::Chi2ImageContext* context, dreal dxy,
                                                 int batch_size,
                                                 int ngauss, const dreal* gauss_params_batch,
                                                 int nrings, const dreal* ring_params_batch,
                                                 int narcs, const dreal* arc_params_batch,
                                                 const dreal* inc_batch, const dreal v_origin,
                                                 const dreal* dRA_batch, const dreal* dDec_batch,
                                                 const dreal* PA_batch, dreal* chi2_out) {
    auto const gauss_stride = ngauss * 2;
    auto const ring_stride = nrings * 3;
    auto const arc_stride = narcs * 5;
    auto const nthreads = tpb * tpb;
    auto const blocks_x = context->nd / nthreads + 1;
    int const chunk_size = resolve_fft_batch_chunk_size(context, batch_size);

    std::fill_n(chi2_out, batch_size, dreal{0.});

    for (int start = 0; start < batch_size; start += chunk_size) {
        int const current_batch = std::min(chunk_size, batch_size - start);
        ensure_fft_batch_workspace(context, current_batch, gauss_stride, ring_stride, arc_stride);

        for (int local_idx = 0; local_idx < current_batch; ++local_idx) {
            int const batch_idx = start + local_idx;
            context->batch_inc_h[local_idx] = inc_batch[batch_idx];
            context->batch_cos_pa_h[local_idx] = cos(PA_batch[batch_idx]);
            context->batch_sin_pa_h[local_idx] = sin(PA_batch[batch_idx]);
            uv_rotate_core(context->batch_cos_pa_h[local_idx], context->batch_sin_pa_h[local_idx],
                           dRA_batch[batch_idx], dDec_batch[batch_idx],
                           context->batch_dRArot_h[local_idx], context->batch_dDecrot_h[local_idx]);
        }

        context->batch_inc_d->CopyFromHost(context->batch_inc_h.data(), current_batch);
        context->batch_cos_pa_d->CopyFromHost(context->batch_cos_pa_h.data(), current_batch);
        context->batch_sin_pa_d->CopyFromHost(context->batch_sin_pa_h.data(), current_batch);
        context->batch_dRArot_d->CopyFromHost(context->batch_dRArot_h.data(), current_batch);
        context->batch_dDecrot_d->CopyFromHost(context->batch_dDecrot_h.data(), current_batch);
        if (gauss_stride > 0) {
            context->batch_gauss_params_d->CopyFromHost(gauss_params_batch + static_cast<size_t>(start) * gauss_stride,
                                                        static_cast<size_t>(current_batch) * gauss_stride);
        }
        if (ring_stride > 0) {
            context->batch_ring_params_d->CopyFromHost(ring_params_batch + static_cast<size_t>(start) * ring_stride,
                                                       static_cast<size_t>(current_batch) * ring_stride);
        }
        if (arc_stride > 0) {
            context->batch_arc_params_d->CopyFromHost(arc_params_batch + static_cast<size_t>(start) * arc_stride,
                                                      static_cast<size_t>(current_batch) * arc_stride);
        }

        CCheck(cudaMemset(context->batch_chi2_d->ptr, 0, sizeof(dreal) * current_batch));
        rasterize_component_image_batch_d<<<dim3(context->ny / tpb + 1, context->nx / tpb + 1, current_batch), dim3(tpb, tpb)>>>(
            context->nx, context->ny, dxy,
            current_batch,
            ngauss, gauss_stride > 0 ? context->batch_gauss_params_d->ptr : nullptr,
            nrings, ring_stride > 0 ? context->batch_ring_params_d->ptr : nullptr,
            narcs, arc_stride > 0 ? context->batch_arc_params_d->ptr : nullptr,
            context->batch_inc_d->ptr,
            context->nx, context->ny, context->batch_model_images_d->ptr);
        direct_chi2_batch_d<<<dim3(blocks_x, current_batch), nthreads, sizeof(dreal) * nthreads>>>(
            context->nx, context->ny, context->batch_model_images_d->ptr, current_batch,
            v_origin, dxy, context->nd, context->u_d.ptr, context->v_d.ptr,
            context->batch_cos_pa_d->ptr, context->batch_sin_pa_d->ptr,
            context->batch_dRArot_d->ptr, context->batch_dDecrot_d->ptr,
            context->vis_obs_re_d.ptr, context->vis_obs_im_d.ptr, context->weights_d.ptr,
            context->batch_chi2_d->ptr);
        CCheck(cudaDeviceSynchronize());
        context->batch_chi2_d->RetrieveCount(chi2_out + start, current_batch);
    }
}
} // anonymous namespace

dreal chi2_image_from_context_components(Chi2ImageContext* context, dreal dxy,
                                   int ngauss, const dreal* gauss_params,
                                   int nrings, const dreal* ring_params,
                                   int narcs, const dreal* arc_params,
                                   dreal inc, const dreal v_origin,
                                   dreal dRA, dreal dDec, dreal duv, dreal PA) {
    // Reuse the GPU batch pipeline even for one model. This keeps component
    // rasterization off the CPU and avoids a full image transfer per call.
    dreal chi2 = 0.0;
    chi2_image_from_context_components_batch(
        context, dxy, 1,
        ngauss, gauss_params,
        nrings, ring_params,
        narcs, arc_params,
        &inc, v_origin, &dRA, &dDec, duv, &PA, &chi2
    );
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

    if (effective_backend == galario::BACKEND_DFT) {
        chi2_image_from_context_components_batch_direct_d(context, dxy, batch_size,
                                                    ngauss, gauss_params_batch,
                                                    nrings, ring_params_batch,
                                                    narcs, arc_params_batch,
                                                    inc_batch, v_origin,
                                                    dRA_batch, dDec_batch,
                                                    PA_batch, chi2_out);
        t_start.Elapsed("chi2_image_from_context_components_tot");
        flush_timing();
        return;
    }
    if (effective_backend == galario::BACKEND_FFT || effective_backend == galario::BACKEND_NUFFT) {
        chi2_image_from_context_components_batch_fft_like_d(context, dxy, batch_size,
                                                      ngauss, gauss_params_batch,
                                                      nrings, ring_params_batch,
                                                      narcs, arc_params_batch,
                                                      inc_batch, v_origin,
                                                      dRA_batch, dDec_batch,
                                                      duv, PA_batch, chi2_out,
                                                      effective_backend == galario::BACKEND_NUFFT);
        t_start.Elapsed("chi2_image_from_context_components_tot");
        flush_timing();
        return;
    }

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
    CPUTimer t_start;
    CHECK_INPUTXY(nxy, nxy);
    galario_profile_detail::check_sweep_inputs(
        nr, r_min, dr, nxy, dxy
    );
    if (nxy != context->nx || nxy != context->ny) {
        throw std::invalid_argument(
            "Profile image size does not match context"
        );
    }

    int const effective_backend = resolve_batched_image_backend(
        nxy, nxy, context->nd, batch_size, context->requested_backend
    );
    int const work_nx =
        effective_backend == galario::BACKEND_NUFFT
        ? resolve_padded_size(nxy, context->nufft_oversample)
        : nxy;
    int const work_ny = work_nx;
    int const work_ncol = work_ny / 2 + 1;
    int const nthreads = tpb * tpb;
    int const blocks_x = context->nd / nthreads + 1;
    dreal const duv_backend = duv * nxy / work_nx;
    int const chunk_size = resolve_fft_batch_chunk_size(
        context, batch_size
    );

    std::fill_n(chi2_out, batch_size, dreal{0.0});
    for (int start = 0; start < batch_size; start += chunk_size) {
        int const current_batch =
            std::min(chunk_size, batch_size - start);
        ensure_fft_batch_workspace(context, current_batch, 0, 0, 0);
        if (
            context->batch_intensity_capacity
            < current_batch * nr
        ) {
            context->batch_intensity_d.reset(
                new CudaMemory<dreal>(
                    static_cast<size_t>(current_batch) * nr
                )
            );
            context->batch_intensity_capacity = current_batch * nr;
        }

        for (int local_idx = 0; local_idx < current_batch; ++local_idx) {
            int const batch_idx = start + local_idx;
            context->batch_inc_h[local_idx] = inc_batch[batch_idx];
            context->batch_cos_pa_h[local_idx] = cos(PA_batch[batch_idx]);
            context->batch_sin_pa_h[local_idx] = sin(PA_batch[batch_idx]);
            uv_rotate_core(
                context->batch_cos_pa_h[local_idx],
                context->batch_sin_pa_h[local_idx],
                dRA_batch[batch_idx],
                dDec_batch[batch_idx],
                context->batch_dRArot_h[local_idx],
                context->batch_dDecrot_h[local_idx]
            );
        }

        context->batch_intensity_d->CopyFromHost(
            intensity_batch + static_cast<size_t>(start) * nr,
            static_cast<size_t>(current_batch) * nr
        );
        context->batch_inc_d->CopyFromHost(
            context->batch_inc_h.data(), current_batch
        );
        context->batch_cos_pa_d->CopyFromHost(
            context->batch_cos_pa_h.data(), current_batch
        );
        context->batch_sin_pa_d->CopyFromHost(
            context->batch_sin_pa_h.data(), current_batch
        );
        context->batch_dRArot_d->CopyFromHost(
            context->batch_dRArot_h.data(), current_batch
        );
        context->batch_dDecrot_d->CopyFromHost(
            context->batch_dDecrot_h.data(), current_batch
        );
        CCheck(cudaMemset(
            context->batch_chi2_d->ptr,
            0,
            sizeof(dreal) * current_batch
        ));

        rasterize_profile_image_batch_d<<<
            dim3(
                work_ny / tpb + 1,
                work_nx / tpb + 1,
                current_batch
            ),
            dim3(tpb, tpb)
        >>>(
            nr,
            context->batch_intensity_d->ptr,
            current_batch,
            r_min,
            dr,
            nxy,
            dxy,
            context->batch_inc_d->ptr,
            work_nx,
            work_ny,
            context->batch_model_images_d->ptr
        );

        if (effective_backend == galario::BACKEND_DFT) {
            direct_chi2_batch_d<<<
                dim3(blocks_x, current_batch),
                nthreads,
                sizeof(dreal) * nthreads
            >>>(
                nxy, nxy,
                context->batch_model_images_d->ptr,
                current_batch,
                1.0,
                dxy,
                context->nd,
                context->u_d.ptr,
                context->v_d.ptr,
                context->batch_cos_pa_d->ptr,
                context->batch_sin_pa_d->ptr,
                context->batch_dRArot_d->ptr,
                context->batch_dDecrot_d->ptr,
                context->vis_obs_re_d.ptr,
                context->vis_obs_im_d.ptr,
                context->weights_d.ptr,
                context->batch_chi2_d->ptr
            );
        } else {
            ensure_fft_batch_plan(context, current_batch);
            shift_real_batch_d<<<
                dim3(
                    work_nx / 2 / tpb + 1,
                    work_ny / 2 / tpb + 1,
                    current_batch
                ),
                dim3(tpb, tpb)
            >>>(
                work_nx, work_ny, current_batch,
                context->batch_model_images_d->ptr
            );
            CUFFTCheck(CUFFTEXEC(
                context->batch_fft_plan,
                context->batch_model_images_d->ptr,
                context->batch_fft_images_d->ptr
            ));
            shift_axis0_batch_d<<<
                dim3(
                    work_nx / 2 / tpb + 1,
                    work_ncol / tpb + 1,
                    current_batch
                ),
                dim3(tpb, tpb)
            >>>(
                work_nx, work_ncol, current_batch,
                context->batch_fft_images_d->ptr
            );
            interpolate_chi2_batch_d<<<
                dim3(blocks_x, current_batch),
                nthreads,
                sizeof(dreal) * nthreads
            >>>(
                work_nx,
                work_ncol,
                context->batch_fft_images_d->ptr,
                current_batch,
                1.0,
                context->nd,
                context->u_d.ptr,
                context->v_d.ptr,
                duv_backend,
                effective_backend == galario::BACKEND_NUFFT,
                context->batch_cos_pa_d->ptr,
                context->batch_sin_pa_d->ptr,
                context->batch_dRArot_d->ptr,
                context->batch_dDecrot_d->ptr,
                context->vis_obs_re_d.ptr,
                context->vis_obs_im_d.ptr,
                context->weights_d.ptr,
                context->batch_chi2_d->ptr
            );
        }

        CCheck(cudaDeviceSynchronize());
        context->batch_chi2_d->RetrieveCount(
            chi2_out + start, current_batch
        );
    }

    t_start.Elapsed("chi2_profile_from_context_batch_tot");
    flush_timing();
}
}
