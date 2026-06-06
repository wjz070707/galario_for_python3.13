#pragma once

#include "galario_defs.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __CUDACC__
#define GALARIO_PROFILE_HD __host__ __device__
#else
#define GALARIO_PROFILE_HD
#endif

namespace galario_profile_detail {

constexpr double pi = 3.141592653589793238462643383279502884;

GALARIO_PROFILE_HD inline dreal bessel_j0(dreal x) {
#ifdef __CUDACC__
    return ::j0(x);
#else
    return ::j0(x);
#endif
}

GALARIO_PROFILE_HD inline void rotate(
    dreal cos_pa,
    dreal sin_pa,
    dreal x,
    dreal y,
    dreal& x_rot,
    dreal& y_rot
) {
    x_rot = x * cos_pa - y * sin_pa;
    y_rot = x * sin_pa + y * cos_pa;
}

GALARIO_PROFILE_HD inline dcomplex sample_visibility(
    int nr,
    const dreal* intensity,
    dreal r_min,
    dreal dr,
    dreal abs_cos_inc,
    dreal cos_pa,
    dreal sin_pa,
    dreal dra_rot,
    dreal ddec_rot,
    dreal u,
    dreal v
) {
    dreal u_rot;
    dreal v_rot;
    rotate(cos_pa, sin_pa, u, v, u_rot, v_rot);

    dreal const rho = hypot(abs_cos_inc * u_rot, v_rot);
    dreal const k = 2.0 * static_cast<dreal>(pi) * rho;
    dreal amplitude = 0.0;

    if (nr >= 2 && abs_cos_inc != 0.0) {
        for (int i = 0; i < nr; ++i) {
            dreal const radius = r_min + i * dr;
            dreal const weight = (i == 0 || i == nr - 1) ? 0.5 : 1.0;
            amplitude +=
                weight * radius * intensity[i] * bessel_j0(k * radius);
        }
        amplitude *=
            2.0 * static_cast<dreal>(pi) * abs_cos_inc * dr;
    }

    dreal const angle = 2.0 * static_cast<dreal>(pi)
        * (u_rot * dra_rot + v_rot * ddec_rot);
    return dcomplex{
        amplitude * static_cast<dreal>(cos(angle)),
        amplitude * static_cast<dreal>(sin(angle))
    };
}

inline void sample_direct(
    int nr,
    const dreal* intensity,
    dreal r_min,
    dreal dr,
    dreal inc,
    dreal dra,
    dreal ddec,
    dreal pa,
    int nd,
    const dreal* u,
    const dreal* v,
    dcomplex* vis
) {
    dreal const abs_cos_inc = std::fabs(std::cos(inc));
    dreal const cos_pa = std::cos(pa);
    dreal const sin_pa = std::sin(pa);
    dreal dra_rot;
    dreal ddec_rot;
    rotate(cos_pa, sin_pa, dra, ddec, dra_rot, ddec_rot);

    for (int i = 0; i < nd; ++i) {
        vis[i] = sample_visibility(
            nr,
            intensity,
            r_min,
            dr,
            abs_cos_inc,
            cos_pa,
            sin_pa,
            dra_rot,
            ddec_rot,
            u[i],
            v[i]
        );
    }
}

inline void check_sweep_inputs(
    int nr,
    dreal r_min,
    dreal dr,
    int nxy,
    dreal dxy
) {
    if (nxy < 2) {
        throw std::invalid_argument(
            "x dimension = " + std::to_string(nxy) + " is less than 2"
        );
    }
    if (nxy % 2 != 0) {
        throw std::invalid_argument(
            "x dimension = " + std::to_string(nxy) + " is odd"
        );
    }
    if (nr < 2) {
        throw std::invalid_argument(
            "profile intensity must contain at least two samples"
        );
    }
    dreal const ratio = (dxy / 2.0 - r_min) / dr;
    if (ratio < 5.0) {
        throw std::invalid_argument(
            "Expect (dxy/2-Rmin)/dR > 5; got "
            + std::to_string(ratio)
        );
    }
}

inline void sweep_image(
    int nr,
    const dreal* intensity,
    dreal r_min,
    dreal dr,
    int nxy,
    dreal dxy,
    dreal inc,
    dcomplex* image
) {
    check_sweep_inputs(nr, r_min, dr, nxy, dxy);

    int const ncol = nxy / 2 + 1;
    int const row_size = 2 * ncol;
    std::fill_n(
        image,
        static_cast<size_t>(nxy) * ncol,
        dcomplex{}
    );

    dreal* real_image = reinterpret_cast<dreal*>(image);
    dreal const cos_inc = std::cos(inc);
    int const rmax = std::min(
        static_cast<int>(std::ceil((r_min + nr * dr) / dxy)),
        nxy / 2
    );
    dreal const sr_to_px = dxy * dxy;
    int const offset = nxy / 2 - rmax;

    for (int i = 0; i < 2 * rmax; ++i) {
        for (int j = 0; j < 2 * rmax; ++j) {
            dreal const x = (rmax - j) * dxy;
            dreal const y = (rmax - i) * dxy;
            dreal const radius =
                std::sqrt((x / cos_inc) * (x / cos_inc) + y * y);
            int const radial_index = std::max(
                static_cast<int>(std::floor((radius - r_min) / dr)),
                0
            );
            size_t const index =
                static_cast<size_t>(i + offset) * row_size + j + offset;

            if (radial_index > nr - 2) {
                real_image[index] = 0.0;
            } else {
                real_image[index] = sr_to_px * (
                    intensity[radial_index]
                    + (
                        radius - radial_index * dr - r_min
                    ) * (
                        intensity[radial_index + 1]
                        - intensity[radial_index]
                    ) / dr
                );
            }
        }
    }

    int const inner_index =
        static_cast<int>(std::floor((dxy / 2.0 - r_min) / dr));
    dreal flux = 0.0;
    for (int i = 1; i < inner_index; ++i) {
        flux += (r_min + dr * i) * intensity[i];
    }
    flux *= 2.0;
    flux +=
        r_min * intensity[0]
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
    real_image[
        static_cast<size_t>(nxy / 2) * row_size + nxy / 2
    ] = sr_to_px * flux / area;
}

inline std::vector<dreal> unpack_image(
    int nxy,
    const dcomplex* image
) {
    std::vector<dreal> unpacked(
        static_cast<size_t>(nxy) * nxy
    );
    int const row_size = 2 * (nxy / 2 + 1);
    const dreal* real_image = reinterpret_cast<const dreal*>(image);

    for (int i = 0; i < nxy; ++i) {
        std::memcpy(
            &unpacked[static_cast<size_t>(i) * nxy],
            &real_image[i * row_size],
            sizeof(dreal) * nxy
        );
    }
    return unpacked;
}

}

#undef GALARIO_PROFILE_HD
