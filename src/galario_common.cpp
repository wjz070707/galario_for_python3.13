/*
 * GALARIO
 * Copyright (C) 2017-2020, Marco Tazzari, Frederik Beaujean, Leonardo Testi.
 * Copyright (C) 2026, wjz070707.
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include "galario_internal.h"

#include <algorithm>
#include <cmath>
#include <cstring>

/*
 * Host-side logic shared by both backend libraries.
 *
 * Backend-selection heuristics and model rasterization belong here. Do not add
 * FFTW or CUDA calls: this file must compile unchanged into both libraries.
 */
namespace {

dreal wrap_angle_pi(dreal angle) {
    dreal const two_pi = 2.0 * static_cast<dreal>(M_PI);
    return angle - two_pi * std::floor(
        (angle + static_cast<dreal>(M_PI)) / two_pi
    );
}

dreal gaussian_ring_value(
    dreal radius,
    dreal flux,
    dreal ring_radius,
    dreal sigma
) {
    dreal const diff = radius - ring_radius;
    return flux * std::exp(-(diff * diff) / (2.0 * sigma * sigma));
}

dreal gaussian_arc_value(
    dreal radius,
    dreal phi,
    dreal flux,
    dreal ring_radius,
    dreal sigma_radius,
    dreal phi_center,
    dreal sigma_phi
) {
    dreal const phi_wrapped = wrap_angle_pi(phi - phi_center);
    dreal const ring_term = gaussian_ring_value(
        radius,
        flux,
        ring_radius,
        sigma_radius
    );
    return ring_term * std::exp(
        -(phi_wrapped * phi_wrapped) / (2.0 * sigma_phi * sigma_phi)
    );
}

}

dreal clamp_nufft_oversample(dreal oversample) {
    return oversample < 1.0 ? 1.0 : oversample;
}

int next_power_of_two(int value) {
    int result = 1;
    while (result < value) {
        result <<= 1;
    }
    return result;
}

int resolve_padded_size(int nxy, dreal oversample) {
    int padded = next_power_of_two(
        static_cast<int>(std::ceil(nxy * clamp_nufft_oversample(oversample)))
    );
    if (padded % 2 != 0) {
        ++padded;
    }
    return std::max(nxy, padded);
}

int resolve_image_backend(int nx, int ny, int nd, int backend) {
    if (backend == galario::BACKEND_AUTO) {
        // Empirical cost proxies, not exact operation counts. Threshold changes
        // affect public AUTO behavior and should be benchmark-backed.
        double const pixels = static_cast<double>(nx) * ny;
        double const fft_cost =
            pixels * std::log2(std::max(2.0, pixels)) + nd;
        double const nufft_pixels = 4.0 * pixels;
        double const nufft_cost =
            0.80 * nufft_pixels * std::log2(std::max(2.0, nufft_pixels))
            + 0.95 * nd;

        if (nx <= 64 && ny <= 64 && nd <= 4096) {
            return galario::BACKEND_DFT;
        }
        if (
            nd >= 250000
            && nx <= 192
            && ny <= 192
            && nufft_cost <= fft_cost * 1.10
        ) {
            return galario::BACKEND_NUFFT;
        }
        return galario::BACKEND_FFT;
    }
    return backend;
}

int resolve_batched_image_backend(
    int nx,
    int ny,
    int nd,
    int batch_size,
    int backend
) {
    if (backend != galario::BACKEND_AUTO) {
        return backend;
    }

    double const pixels = static_cast<double>(nx) * ny;
    double const nufft_pixels = 4.0 * pixels;
    double const fft_cost = static_cast<double>(batch_size)
        * (
            pixels * std::log2(std::max(2.0, pixels))
            + static_cast<double>(nd)
        );
    double const dft_cost =
        static_cast<double>(batch_size) * 0.50 * pixels * nd;
    double const nufft_cost = static_cast<double>(batch_size)
        * (
            0.45 * nufft_pixels
                * std::log2(std::max(2.0, nufft_pixels))
            + 0.35 * nd
        );

    if (
        nx <= 64
        && ny <= 64
        && nd <= 4096
        && dft_cost <= fft_cost * 2.0
    ) {
        return galario::BACKEND_DFT;
    }
    if (
        nd >= 250000
        && batch_size >= 16
        && nx <= 256
        && ny <= 256
        && nufft_cost <= fft_cost * 1.15
    ) {
        return galario::BACKEND_NUFFT;
    }
    return galario::BACKEND_FFT;
}

int resolve_profile_backend(int nr, int nd, int backend) {
    (void)nr;
    (void)nd;
    if (backend == galario::BACKEND_AUTO) {
        return galario::BACKEND_DFT;
    }
    return backend;
}

std::vector<dreal> pad_real_image(
    int nx_in,
    int ny_in,
    const dreal* realdata,
    int nx_out,
    int ny_out
) {
    std::vector<dreal> padded(
        static_cast<size_t>(nx_out) * ny_out,
        0.0
    );
    int const row_offset = (nx_out - nx_in) / 2;
    int const col_offset = (ny_out - ny_in) / 2;

    for (int i = 0; i < nx_in; ++i) {
        std::memcpy(
            &padded[
                static_cast<size_t>(i + row_offset) * ny_out + col_offset
            ],
            &realdata[static_cast<size_t>(i) * ny_in],
            sizeof(dreal) * ny_in
        );
    }

    return padded;
}

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
) {
    // The model dimensions describe the physical image; the output may be
    // larger when an oversampled backend needs centered zero padding.
    std::fill_n(out, static_cast<size_t>(nx_out) * ny_out, 0.0);

    int const row_offset = (nx_out - nx_model) / 2;
    int const col_offset = (ny_out - ny_model) / 2;
    dreal const x_min = -0.5 * ny_model * dxy;
    dreal const y_min = -0.5 * nx_model * dxy;
    dreal const x_step =
        ny_model > 1 ? (ny_model * dxy) / (ny_model - 1) : 0.0;
    dreal const y_step =
        nx_model > 1 ? (nx_model * dxy) / (nx_model - 1) : 0.0;
    dreal const cos_inc = std::cos(inc);

#pragma omp parallel for
    for (int i = 0; i < nx_model; ++i) {
        dreal const y = y_min + i * y_step;
        size_t const out_row =
            static_cast<size_t>(i + row_offset) * ny_out + col_offset;

        for (int j = 0; j < ny_model; ++j) {
            dreal const x = x_min + j * x_step;
            dreal const x_deproj = x / cos_inc;
            dreal const radius = std::sqrt(x_deproj * x_deproj + y * y);
            dreal const phi = std::atan2(y, x_deproj);
            dreal value = 0.0;

            for (int g = 0; g < ngauss; ++g) {
                int const base = 2 * g;
                dreal const flux = gauss_params[base];
                dreal const sigma = gauss_params[base + 1];
                value += flux * std::exp(
                    -(radius * radius) / (2.0 * sigma * sigma)
                );
            }
            for (int r = 0; r < nrings; ++r) {
                int const base = 3 * r;
                value += gaussian_ring_value(
                    radius,
                    ring_params[base],
                    ring_params[base + 1],
                    ring_params[base + 2]
                );
            }
            for (int a = 0; a < narcs; ++a) {
                int const base = 5 * a;
                value += gaussian_arc_value(
                    radius,
                    phi,
                    arc_params[base],
                    arc_params[base + 1],
                    arc_params[base + 2],
                    arc_params[base + 3],
                    arc_params[base + 4]
                );
            }

            out[out_row + j] = value;
        }
    }
}

}
