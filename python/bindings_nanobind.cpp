/*
 * GALARIO
 * Copyright (C) 2017-2020, Marco Tazzari, Frederik Beaujean, Leonardo Testi.
 * Copyright (C) 2026, wjz070707.
 * SPDX-License-Identifier: LGPL-3.0-or-later
 */

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "galario.h"
#include "galario_py.h"

namespace nb = nanobind;
using namespace nb::literals;

#ifndef GALARIO_NB_MODULE_NAME
#define GALARIO_NB_MODULE_NAME _nanobind_double_cuda
#endif

#ifndef GALARIO_NB_WITH_CONTEXT
#define GALARIO_NB_WITH_CONTEXT 1
#endif

#ifndef GALARIO_NB_CONTEXT_TAG
#define GALARIO_NB_CONTEXT_TAG 0
#endif

namespace {

using Real = dreal;
using Complex = std::complex<dreal>;
using RealArray1D = nb::ndarray<nb::numpy, const Real, nb::ndim<1>, nb::c_contig>;
using RealArray2D = nb::ndarray<nb::numpy, const Real, nb::ndim<2>, nb::c_contig>;

nb::list vector_to_list(const std::vector<Real> &values) {
    nb::list out;
    for (Real value : values) {
        out.append(value);
    }
    return out;
}

nb::list complex_vector_to_list(const std::vector<Complex> &values) {
    nb::list out;
    for (const auto &value : values) {
        out.append(value);
    }
    return out;
}

int parse_backend(const std::string &backend) {
    if (backend == "auto") {
        return galario::BACKEND_AUTO;
    }
    if (backend == "fft") {
        return galario::BACKEND_FFT;
    }
    if (backend == "dft") {
        return galario::BACKEND_DFT;
    }
    if (backend == "nufft") {
        return galario::BACKEND_NUFFT;
    }
    throw std::invalid_argument("Unknown backend: " + backend);
}

std::string backend_name(int backend) {
    switch (backend) {
        case galario::BACKEND_AUTO: return "auto";
        case galario::BACKEND_FFT: return "fft";
        case galario::BACKEND_DFT: return "dft";
        case galario::BACKEND_NUFFT: return "nufft";
        default: throw std::invalid_argument("Unknown backend code");
    }
}

Real set_v_origin(const std::string &origin) {
    if (origin == "upper") {
        return static_cast<Real>(1.0);
    }
    if (origin == "lower") {
        return static_cast<Real>(-1.0);
    }
    throw std::invalid_argument("Expect origin to be 'upper' or 'lower'");
}

void validate_obs(const RealArray1D &u,
                  const RealArray1D &v,
                  const RealArray1D &vis_obs_re,
                  const RealArray1D &vis_obs_im,
                  const RealArray1D &weights) {
    size_t nd = u.shape(0);
    if (v.shape(0) != nd || vis_obs_re.shape(0) != nd || vis_obs_im.shape(0) != nd || weights.shape(0) != nd) {
        throw std::invalid_argument("Observation arrays must all have the same length");
    }
}

void validate_component_shape(const RealArray2D &params, size_t width, const char *name) {
    if (params.shape(1) != width) {
        throw std::invalid_argument(std::string("Expect ") + name + " to have width " + std::to_string(width));
    }
}

template <int Tag>
struct NBChi2ImageContextT {
    explicit NBChi2ImageContextT(galario::Chi2ImageContext *ctx_, int nx_, int ny_, int nd_)
        : ctx(ctx_), nx_value(nx_), ny_value(ny_), nd_value(nd_) {}
    ~NBChi2ImageContextT() {
        if (ctx) {
            galario::destroy_image_context(ctx);
            ctx = nullptr;
        }
    }

    NBChi2ImageContextT(const NBChi2ImageContextT &) = delete;
    NBChi2ImageContextT &operator=(const NBChi2ImageContextT &) = delete;
    NBChi2ImageContextT(NBChi2ImageContextT &&other) noexcept : ctx(other.ctx) {
        other.ctx = nullptr;
    }
    NBChi2ImageContextT &operator=(NBChi2ImageContextT &&other) noexcept {
        if (this != &other) {
            if (ctx) {
                galario::destroy_image_context(ctx);
            }
            ctx = other.ctx;
            other.ctx = nullptr;
        }
        return *this;
    }

    int nx() const { return nx_value; }
    int ny() const { return ny_value; }
    int nd() const { return nd_value; }
    std::string requested_backend() const { return backend_name(galario::image_context_requested_backend(ctx)); }
    std::string resolved_backend() const { return backend_name(galario::image_context_backend(ctx)); }
    std::string batch_backend(int batch_size) const { return backend_name(galario::image_context_batch_backend(ctx, batch_size)); }

    galario::Chi2ImageContext *ctx;
    int nx_value;
    int ny_value;
    int nd_value;
};

using NBChi2ImageContext = NBChi2ImageContextT<GALARIO_NB_CONTEXT_TAG>;

}  // namespace

NB_MODULE(GALARIO_NB_MODULE_NAME, m) {
    m.doc() = "Experimental nanobind wrapper for GALARIO APIs.";
    m.attr("BACKEND_AUTO") = "auto";
    m.attr("BACKEND_FFT") = "fft";
    m.attr("BACKEND_DFT") = "dft";
    m.attr("BACKEND_NUFFT") = "nufft";
    m.def("_init", &galario::init);
    m.def("_cleanup", &galario::cleanup);
    m.def("threads", &galario::threads, "num"_a = 0);
    m.def("ngpus", &galario::ngpus);
    m.def("use_gpu", &galario::use_gpu, "device_id"_a);

#if GALARIO_NB_WITH_CONTEXT
    nb::class_<NBChi2ImageContext>(m, "Chi2ImageContext")
        .def_prop_ro("nx", &NBChi2ImageContext::nx)
        .def_prop_ro("ny", &NBChi2ImageContext::ny)
        .def_prop_ro("nd", &NBChi2ImageContext::nd)
        .def_prop_ro("shape", [](const NBChi2ImageContext &ctx) {
            return nb::make_tuple(ctx.nx(), ctx.ny());
        })
        .def_prop_ro("requested_backend", &NBChi2ImageContext::requested_backend)
        .def_prop_ro("resolved_backend", &NBChi2ImageContext::resolved_backend)
        .def("batch_backend", &NBChi2ImageContext::batch_backend, "batch_size"_a);

    m.def("_create_image_context",
          [](int nx, int ny,
             const RealArray1D &u,
             const RealArray1D &v,
             const RealArray1D &vis_obs_re,
             const RealArray1D &vis_obs_im,
             const RealArray1D &weights,
             const std::string &backend,
             double nufft_oversample) {
              validate_obs(u, v, vis_obs_re, vis_obs_im, weights);
              auto *ctx = galario::create_image_context(
                  nx, ny, static_cast<int>(u.shape(0)),
                  u.data(), v.data(),
                  vis_obs_re.data(), vis_obs_im.data(), weights.data(),
                  parse_backend(backend), nufft_oversample);
              return std::make_unique<NBChi2ImageContext>(ctx, nx, ny, static_cast<int>(u.shape(0)));
          },
          "nx"_a, "ny"_a, "u"_a, "v"_a, "vis_obs_re"_a, "vis_obs_im"_a, "weights"_a,
          "backend"_a = std::string("auto"), "nufft_oversample"_a = 2.0);
#endif

    m.def("sampleImage",
          [](const RealArray2D &image,
             double dxy,
             const RealArray1D &u,
             const RealArray1D &v,
             double dRA,
             double dDec,
             double PA,
             const std::string &origin,
             const std::string &backend,
             double nufft_oversample) {
              if (u.shape(0) != v.shape(0)) {
                  throw std::invalid_argument("u and v must have the same length");
              }
              if (image.shape(0) != image.shape(1)) {
                  throw std::invalid_argument("Expect square image");
              }
              std::vector<Complex> vis_out(u.shape(0));
              Real duv = static_cast<Real>(1.0) / (static_cast<Real>(dxy) * static_cast<Real>(image.shape(0)));
              galario::_sample_image(
                  static_cast<int>(image.shape(0)),
                  static_cast<int>(image.shape(1)),
                  const_cast<Real*>(image.data()),
                  set_v_origin(origin),
                  dRA,
                  dDec,
                  duv,
                  PA,
                  static_cast<int>(u.shape(0)),
                  const_cast<Real*>(u.data()),
                  const_cast<Real*>(v.data()),
                  vis_out.data(),
                  parse_backend(backend),
                  nufft_oversample);
              return complex_vector_to_list(vis_out);
          },
          "image"_a, "dxy"_a, "u"_a, "v"_a,
          "dRA"_a = 0.0, "dDec"_a = 0.0, "PA"_a = 0.0,
          "origin"_a = std::string("upper"),
          "backend"_a = std::string("auto"),
          "nufft_oversample"_a = 2.0);

    m.def("sampleImageComponents",
          [](int nx,
             int ny,
             double dxy,
             const RealArray1D &u,
             const RealArray1D &v,
             const RealArray2D &gauss_params,
             const RealArray2D &ring_params,
             const RealArray2D &arc_params,
             double inc,
             double dRA,
             double dDec,
             double PA,
             const std::string &origin,
             const std::string &backend,
             double nufft_oversample) {
              if (u.shape(0) != v.shape(0)) {
                  throw std::invalid_argument("u and v must have the same length");
              }
              validate_component_shape(gauss_params, 2, "gauss_params");
              validate_component_shape(ring_params, 3, "ring_params");
              validate_component_shape(arc_params, 5, "arc_params");
              std::vector<Complex> vis_out(u.shape(0));
              Real duv = static_cast<Real>(1.0) / (static_cast<Real>(dxy) * static_cast<Real>(nx));
              galario::_sample_image_components(
                  nx, ny, dxy,
                  static_cast<int>(gauss_params.shape(0)), gauss_params.shape(0) ? const_cast<Real*>(gauss_params.data()) : nullptr,
                  static_cast<int>(ring_params.shape(0)), ring_params.shape(0) ? const_cast<Real*>(ring_params.data()) : nullptr,
                  static_cast<int>(arc_params.shape(0)), arc_params.shape(0) ? const_cast<Real*>(arc_params.data()) : nullptr,
                  inc, set_v_origin(origin), dRA, dDec, duv, PA,
                  static_cast<int>(u.shape(0)), const_cast<Real*>(u.data()), const_cast<Real*>(v.data()), vis_out.data(),
                  parse_backend(backend), nufft_oversample);
              return complex_vector_to_list(vis_out);
          },
          "nx"_a, "ny"_a, "dxy"_a, "u"_a, "v"_a,
          "gauss_params"_a, "ring_params"_a, "arc_params"_a,
          "inc"_a = 0.0, "dRA"_a = 0.0, "dDec"_a = 0.0, "PA"_a = 0.0,
          "origin"_a = std::string("upper"),
          "backend"_a = std::string("auto"),
          "nufft_oversample"_a = 2.0);

    m.def("sampleProfile",
          [](const RealArray1D &intensity,
             double Rmin,
             double dR,
             int nxy,
             double dxy,
             const RealArray1D &u,
             const RealArray1D &v,
             double dRA,
             double dDec,
             double PA,
             double inc,
             const std::string &backend,
             double nufft_oversample) {
              if (u.shape(0) != v.shape(0)) {
                  throw std::invalid_argument("u and v must have the same length");
              }
              std::vector<Complex> vis_out(u.shape(0));
              Real duv = static_cast<Real>(1.0) / (static_cast<Real>(dxy) * static_cast<Real>(nxy));
              galario::_sample_profile(
                  static_cast<int>(intensity.shape(0)), const_cast<Real*>(intensity.data()), Rmin, dR, dxy, nxy, inc, dRA, dDec, duv, PA,
                  static_cast<int>(u.shape(0)), const_cast<Real*>(u.data()), const_cast<Real*>(v.data()), vis_out.data(),
                  parse_backend(backend), nufft_oversample);
              return complex_vector_to_list(vis_out);
          },
          "intensity"_a, "Rmin"_a, "dR"_a, "nxy"_a, "dxy"_a, "u"_a, "v"_a,
          "dRA"_a = 0.0, "dDec"_a = 0.0, "PA"_a = 0.0, "inc"_a = 0.0,
          "backend"_a = std::string("auto"),
          "nufft_oversample"_a = 2.0);

#if GALARIO_NB_WITH_CONTEXT
    m.def("_chi2_image_from_context",
          [](NBChi2ImageContext &ctx,
             const RealArray2D &image,
             double dxy,
             double dRA,
             double dDec,
             double PA,
             const std::string &origin) {
              if (image.shape(0) != static_cast<size_t>(ctx.nx()) || image.shape(1) != static_cast<size_t>(ctx.ny())) {
                  throw std::invalid_argument("Image shape does not match context");
              }
              Real duv = static_cast<Real>(1.0) / (static_cast<Real>(dxy) * static_cast<Real>(ctx.nx()));
              return galario::chi2_image_from_context(ctx.ctx, image.data(), set_v_origin(origin), dRA, dDec, duv, PA);
          },
          "ctx"_a, "image"_a, "dxy"_a, "dRA"_a = 0.0, "dDec"_a = 0.0, "PA"_a = 0.0, "origin"_a = std::string("upper"));

    m.def("_chi2_image_from_context_components",
          [](NBChi2ImageContext &ctx,
             double dxy,
             const RealArray2D &gauss_params,
             const RealArray2D &ring_params,
             const RealArray2D &arc_params,
             double inc,
             double dRA,
             double dDec,
             double PA,
             const std::string &origin) {
              validate_component_shape(gauss_params, 2, "gauss_params");
              validate_component_shape(ring_params, 3, "ring_params");
              validate_component_shape(arc_params, 5, "arc_params");
              Real duv = static_cast<Real>(1.0) / (static_cast<Real>(dxy) * static_cast<Real>(ctx.nx()));
              return galario::chi2_image_from_context_components(
                  ctx.ctx, dxy,
                  static_cast<int>(gauss_params.shape(0)), gauss_params.shape(0) ? gauss_params.data() : nullptr,
                  static_cast<int>(ring_params.shape(0)), ring_params.shape(0) ? ring_params.data() : nullptr,
                  static_cast<int>(arc_params.shape(0)), arc_params.shape(0) ? arc_params.data() : nullptr,
                  inc, set_v_origin(origin), dRA, dDec, duv, PA);
          },
          "ctx"_a, "dxy"_a,
          "gauss_params"_a, "ring_params"_a, "arc_params"_a,
          "inc"_a = 0.0, "dRA"_a = 0.0, "dDec"_a = 0.0, "PA"_a = 0.0,
          "origin"_a = std::string("upper"));

    m.def("_chi2_profile_from_context",
          [](NBChi2ImageContext &ctx,
             const RealArray1D &intensity,
             double Rmin,
             double dR,
             int nxy,
             double dxy,
             double dRA,
             double dDec,
             double PA,
             double inc) {
              if (nxy != ctx.nx() || nxy != ctx.ny()) {
                  throw std::invalid_argument(
                      "Profile image size does not match context"
                  );
              }
              Real duv = static_cast<Real>(1.0)
                  / (static_cast<Real>(dxy) * static_cast<Real>(nxy));
              return galario::chi2_profile_from_context(
                  ctx.ctx,
                  static_cast<int>(intensity.shape(0)),
                  intensity.data(),
                  Rmin, dR, nxy, dxy,
                  inc, dRA, dDec, duv, PA
              );
          },
          "ctx"_a, "intensity"_a, "Rmin"_a, "dR"_a,
          "nxy"_a, "dxy"_a,
          "dRA"_a = 0.0, "dDec"_a = 0.0,
          "PA"_a = 0.0, "inc"_a = 0.0);

    m.def("_chi2_profile_from_context_batch",
          [](NBChi2ImageContext &ctx,
             const RealArray2D &intensity_batch,
             double Rmin,
             double dR,
             int nxy,
             double dxy,
             const RealArray1D &inc_batch,
             const RealArray1D &dRA_batch,
             const RealArray1D &dDec_batch,
             const RealArray1D &PA_batch) {
              size_t const batch_size = intensity_batch.shape(0);
              if (
                  inc_batch.shape(0) != batch_size
                  || dRA_batch.shape(0) != batch_size
                  || dDec_batch.shape(0) != batch_size
                  || PA_batch.shape(0) != batch_size
              ) {
                  throw std::invalid_argument(
                      "Profile batch parameter arrays must match batch size"
                  );
              }
              if (nxy != ctx.nx() || nxy != ctx.ny()) {
                  throw std::invalid_argument(
                      "Profile image size does not match context"
                  );
              }
              std::vector<Real> chi2(batch_size);
              Real duv = static_cast<Real>(1.0)
                  / (static_cast<Real>(dxy) * static_cast<Real>(nxy));
              galario::chi2_profile_from_context_batch(
                  ctx.ctx,
                  static_cast<int>(intensity_batch.shape(1)),
                  intensity_batch.data(),
                  static_cast<int>(batch_size),
                  Rmin, dR, nxy, dxy,
                  inc_batch.data(),
                  dRA_batch.data(),
                  dDec_batch.data(),
                  duv,
                  PA_batch.data(),
                  chi2.data()
              );
              return vector_to_list(chi2);
          },
          "ctx"_a, "intensity_batch"_a,
          "Rmin"_a, "dR"_a, "nxy"_a, "dxy"_a,
          "inc_batch"_a, "dRA_batch"_a,
          "dDec_batch"_a, "PA_batch"_a);
#endif

    m.def("chi2Image",
          [](const RealArray2D &image,
             double dxy,
             const RealArray1D &u,
             const RealArray1D &v,
             const RealArray1D &vis_obs_re,
             const RealArray1D &vis_obs_im,
             const RealArray1D &weights,
             double dRA,
             double dDec,
             double PA,
             const std::string &origin,
             const std::string &backend,
             double nufft_oversample) {
              validate_obs(u, v, vis_obs_re, vis_obs_im, weights);
              if (image.shape(0) != image.shape(1)) {
                  throw std::invalid_argument("Expect square image");
              }
              Real duv = static_cast<Real>(1.0) / (static_cast<Real>(dxy) * static_cast<Real>(image.shape(0)));
              return galario::chi2_image(
                  static_cast<int>(image.shape(0)),
                  static_cast<int>(image.shape(1)),
                  image.data(),
                  set_v_origin(origin),
                  dRA,
                  dDec,
                  duv,
                  PA,
                  static_cast<int>(u.shape(0)),
                  u.data(),
                  v.data(),
                  vis_obs_re.data(),
                  vis_obs_im.data(),
                  weights.data(),
                  parse_backend(backend),
                  nufft_oversample);
          },
          "image"_a, "dxy"_a, "u"_a, "v"_a, "vis_obs_re"_a, "vis_obs_im"_a, "weights"_a,
          "dRA"_a = 0.0, "dDec"_a = 0.0, "PA"_a = 0.0,
          "origin"_a = std::string("upper"),
          "backend"_a = std::string("auto"),
          "nufft_oversample"_a = 2.0);

    m.def("chi2ImageComponents",
          [](int nx, int ny,
             double dxy,
             const RealArray1D &u,
             const RealArray1D &v,
             const RealArray1D &vis_obs_re,
             const RealArray1D &vis_obs_im,
             const RealArray1D &weights,
             const RealArray2D &gauss_params,
             const RealArray2D &ring_params,
             const RealArray2D &arc_params,
             double inc,
             double dRA,
             double dDec,
             double PA,
             const std::string &origin,
             const std::string &backend,
             double nufft_oversample) {
              validate_obs(u, v, vis_obs_re, vis_obs_im, weights);
              validate_component_shape(gauss_params, 2, "gauss_params");
              validate_component_shape(ring_params, 3, "ring_params");
              validate_component_shape(arc_params, 5, "arc_params");
              Real duv = static_cast<Real>(1.0) / (static_cast<Real>(dxy) * static_cast<Real>(nx));
              return galario::chi2_image_components(
                  nx, ny, dxy,
                  static_cast<int>(gauss_params.shape(0)), gauss_params.shape(0) ? gauss_params.data() : nullptr,
                  static_cast<int>(ring_params.shape(0)), ring_params.shape(0) ? ring_params.data() : nullptr,
                  static_cast<int>(arc_params.shape(0)), arc_params.shape(0) ? arc_params.data() : nullptr,
                  inc, set_v_origin(origin), dRA, dDec, duv, PA,
                  static_cast<int>(u.shape(0)),
                  u.data(), v.data(),
                  vis_obs_re.data(), vis_obs_im.data(), weights.data(),
                  parse_backend(backend), nufft_oversample);
          },
          "nx"_a, "ny"_a, "dxy"_a,
          "u"_a, "v"_a, "vis_obs_re"_a, "vis_obs_im"_a, "weights"_a,
          "gauss_params"_a, "ring_params"_a, "arc_params"_a,
          "inc"_a = 0.0, "dRA"_a = 0.0, "dDec"_a = 0.0, "PA"_a = 0.0,
          "origin"_a = std::string("upper"),
          "backend"_a = std::string("auto"),
          "nufft_oversample"_a = 2.0);

    m.def("chi2Profile",
          [](const RealArray1D &intensity,
             double Rmin,
             double dR,
             int nxy,
             double dxy,
             const RealArray1D &u,
             const RealArray1D &v,
             const RealArray1D &vis_obs_re,
             const RealArray1D &vis_obs_im,
             const RealArray1D &weights,
             double dRA,
             double dDec,
             double PA,
             double inc,
             const std::string &backend,
             double nufft_oversample) {
              validate_obs(u, v, vis_obs_re, vis_obs_im, weights);
              Real duv = static_cast<Real>(1.0) / (static_cast<Real>(dxy) * static_cast<Real>(nxy));
              return galario::chi2_profile(
                  static_cast<int>(intensity.shape(0)), intensity.data(), Rmin, dR, dxy, nxy, inc, dRA, dDec, duv, PA,
                  static_cast<int>(u.shape(0)), u.data(), v.data(), vis_obs_re.data(), vis_obs_im.data(), weights.data(),
                  parse_backend(backend), nufft_oversample);
          },
          "intensity"_a, "Rmin"_a, "dR"_a, "nxy"_a, "dxy"_a,
          "u"_a, "v"_a, "vis_obs_re"_a, "vis_obs_im"_a, "weights"_a,
          "dRA"_a = 0.0, "dDec"_a = 0.0, "PA"_a = 0.0, "inc"_a = 0.0,
          "backend"_a = std::string("auto"),
          "nufft_oversample"_a = 2.0);

#if GALARIO_NB_WITH_CONTEXT
    m.def("_chi2_image_from_context_components_batch",
          [](NBChi2ImageContext &ctx,
             double dxy,
             const RealArray2D &gauss_params_batch,
             const RealArray2D &ring_params_batch,
             const RealArray2D &arc_params_batch,
             const RealArray1D &inc_batch,
             const RealArray1D &dRA_batch,
             const RealArray1D &dDec_batch,
             const RealArray1D &PA_batch,
             const std::string &origin) {
              size_t batch_size = inc_batch.shape(0);
              if (dRA_batch.shape(0) != batch_size || dDec_batch.shape(0) != batch_size || PA_batch.shape(0) != batch_size) {
                  throw std::invalid_argument("Batch parameter vectors must have the same length");
              }
              if (gauss_params_batch.shape(0) != batch_size || ring_params_batch.shape(0) != batch_size || arc_params_batch.shape(0) != batch_size) {
                  throw std::invalid_argument("Component batch matrices must match batch size");
              }
              if (gauss_params_batch.shape(1) % 2 != 0 || ring_params_batch.shape(1) % 3 != 0 || arc_params_batch.shape(1) % 5 != 0) {
                  throw std::invalid_argument("Component batch widths must be divisible by 2, 3, and 5 respectively");
              }
              Real duv = static_cast<Real>(1.0) / (static_cast<Real>(dxy) * static_cast<Real>(ctx.nx()));
              std::vector<Real> chi2_out(batch_size);
              galario::chi2_image_from_context_components_batch(
                  ctx.ctx, dxy, static_cast<int>(batch_size),
                  static_cast<int>(gauss_params_batch.shape(1) / 2), gauss_params_batch.shape(1) ? gauss_params_batch.data() : nullptr,
                  static_cast<int>(ring_params_batch.shape(1) / 3), ring_params_batch.shape(1) ? ring_params_batch.data() : nullptr,
                  static_cast<int>(arc_params_batch.shape(1) / 5), arc_params_batch.shape(1) ? arc_params_batch.data() : nullptr,
                  inc_batch.data(), set_v_origin(origin),
                  dRA_batch.data(), dDec_batch.data(), duv, PA_batch.data(), chi2_out.data());
              return vector_to_list(chi2_out);
          },
          "ctx"_a, "dxy"_a,
          "gauss_params_batch"_a, "ring_params_batch"_a, "arc_params_batch"_a,
          "inc_batch"_a, "dRA_batch"_a, "dDec_batch"_a, "PA_batch"_a,
          "origin"_a = std::string("upper"));
#endif
}

