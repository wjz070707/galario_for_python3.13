###############################################################################
# This file is part of GALARIO:                                               #
# Gpu Accelerated Library for Analysing Radio Interferometer Observations     #
#                                                                             #
# Copyright (C) 2017-2020, Marco Tazzari, Frederik Beaujean, Leonardo Testi.  #
# Copyright (C) 2026, wjz070707.                                             #
#                                                                             #
# This program is free software: you can redistribute it and/or modify        #
# it under the terms of the Lesser GNU General Public License as published by #
# the Free Software Foundation, either version 3 of the License, or           #
# (at your option) any later version.                                         #
#                                                                             #
# This program is distributed in the hope that it will be useful,             #
# but WITHOUT ANY WARRANTY; without even the implied warranty of              #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                        #
#                                                                             #
# For more details see the LICENSE file.                                      #
# Maintained at https://github.com/wjz070707/galario_for_python3.13           #
###############################################################################

#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import (division, print_function, absolute_import, unicode_literals)

# Public Python API integration tests and numerical backend comparisons.
import numpy as np
import pytest
from os import environ

from utils import *

import galario
from galario import deg, arcsec

if galario.HAVE_CUDA and int(environ.get("GALARIO_TEST_GPU", 0)):
    from galario import double_cuda as g_double
else:
    from galario import double as g_double

from galario import double as g_double_cpu

CUDA_DIRECT_IMAGE_BACKEND = galario.HAVE_CUDA and int(environ.get("GALARIO_TEST_GPU", 0))
IMAGE_BACKENDS = [galario.BACKEND_FFT, galario.BACKEND_DFT, galario.BACKEND_NUFFT]


def test_estimate_fov_from_source_uses_extent_offset_and_padding():
    fov = g_double_cpu.estimate_fov_from_source(
        4.0 * arcsec,
        offset=(2.0 * arcsec, -1.0 * arcsec),
        padding=4.0 / 3.0,
    )

    assert fov / arcsec == pytest.approx(16.0)


def test_estimate_fov_from_source_can_require_primary_beam_level():
    fov = g_double_cpu.estimate_fov_from_source(
        1.0 * arcsec,
        primary_beam=10.0 * arcsec,
        primary_beam_level=0.25,
    )

    assert fov / arcsec == pytest.approx(10.0 * np.sqrt(2.0))


def test_estimate_fov_from_source_rejects_unsafe_max_fov():
    with pytest.raises(RuntimeError, match="exceeds max_fov"):
        g_double_cpu.estimate_fov_from_source(
            4.0 * arcsec,
            offset=2.0 * arcsec,
            padding=2.0,
            max_fov=10.0 * arcsec,
        )


def test_get_image_size_from_fov_uses_explicit_fov():
    u = np.array([1.0e5, 3.8e6])
    v = np.array([0.0, 0.0])
    fov = 15.0 * arcsec

    nxy, dxy = g_double_cpu.get_image_size_from_fov(
        u, v, fov, pixels_per_fringe=3.0
    )

    assert nxy == 1024
    assert dxy == pytest.approx(fov / nxy)
    assert 1.0 / (3.8e6 * dxy) >= 3.0


def test_get_image_size_from_fov_rejects_invalid_uv_data():
    with pytest.raises(ValueError, match="No finite, positive uv distances"):
        g_double_cpu.get_image_size_from_fov(
            np.array([0.0, np.nan]), np.array([0.0, 0.0]), 1.0 * arcsec
        )

# PARAMETERS FOR MULTIPLE TEST EXECUTIONS
par1 = {'dRA': 0., 'dDec': 0.4, 'PA': 2., 'nxy': 1024}
par2 = {'dRA': -3.5, 'dDec': 7.2, 'PA': -23., 'nxy': 2048}
par3 = {'dRA': 2.3, 'dDec': 3.2, 'PA': 88., 'nxy': 4096}
par4 = {'dRA': 0., 'dDec': 0., 'PA': 145., 'nxy': 1024}


# use last gpu if available. Check `watch -n 0.1 nvidia-smi` to see which gpu is
# used during test execution.
ngpus = g_double.ngpus()
g_double.use_gpu(0) #max(0, ngpus-1))

g_double.threads()


def _make_image_backend_case(nxy=128, nsamples=96, real_type='float64'):
    udat, vdat = create_sampling_points(nsamples, 1500., dtype=real_type)
    _, _, maxuv = matrix_size(udat, vdat)
    dxy = 1. / maxuv
    image = create_reference_image(nxy, x0=nxy / 10., y0=-nxy / 14.,
                                   sigma_x=0.22 * nxy, sigma_y=0.15 * nxy,
                                   dtype=real_type)
    x_obs, _, weights = generate_random_vis(nsamples, real_type)

    return {
        'image': image,
        'dxy': dxy,
        'u': udat,
        'v': vdat,
        'vis_obs_re': x_obs.real.copy(),
        'vis_obs_im': x_obs.imag.copy(),
        'weights': weights,
        'dRA': 0.35 * arcsec,
        'dDec': -0.2 * arcsec,
        'PA': 18. * deg
    }


def _make_profile_backend_case(nxy=256, nsamples=96, real_type='float64'):
    udat, vdat = create_sampling_points(nsamples, 1500., dtype=real_type)
    _, _, maxuv = matrix_size(udat, vdat)
    dxy = 1. / maxuv
    Rmin = dxy / 120.
    dR = dxy / 18.
    nrad = 2048
    inc = 27. * deg
    intensity = radial_profile(Rmin, dR, nrad, 'Cos-Gauss', dtype=real_type, gauss_width=dxy * 14.)
    image = sweep_ref(intensity, Rmin, dR, nxy, nxy, dxy, inc, dtype_image=real_type)
    x_obs, _, weights = generate_random_vis(nsamples, real_type)

    return {
        'intensity': intensity,
        'Rmin': Rmin,
        'dR': dR,
        'nxy': nxy,
        'dxy': dxy,
        'u': udat,
        'v': vdat,
        'image': image,
        'inc': inc,
        'vis_obs_re': x_obs.real.copy(),
        'vis_obs_im': x_obs.imag.copy(),
        'weights': weights,
        'dRA': -0.15 * arcsec,
        'dDec': 0.1 * arcsec,
        'PA': 31. * deg
    }


def _make_component_case(nxy=96, nsamples=64, real_type='float64'):
    udat, vdat = create_sampling_points(nsamples, 1500., dtype=real_type)
    _, _, maxuv = matrix_size(udat, vdat)
    dxy = 1. / maxuv
    inc = 29. * deg
    gauss_params = np.array([[0.25, 0.05 * arcsec]], dtype=real_type)
    ring_params = np.array([
        [0.80, 0.21 * arcsec, 0.032 * arcsec],
        [0.42, 0.46 * arcsec, 0.041 * arcsec],
    ], dtype=real_type)
    arc_params = np.array([[0.30, 0.37 * arcsec, 0.025 * arcsec, 118. * deg, 16. * deg]], dtype=real_type)
    xarr = np.linspace(-nxy / 2 * dxy, nxy / 2 * dxy, nxy)
    yarr = np.linspace(-nxy / 2 * dxy, nxy / 2 * dxy, nxy)
    x, y = np.meshgrid(xarr, yarr)
    x_deproj = x / np.cos(inc)
    radius = np.sqrt(x_deproj ** 2. + y ** 2.)
    phi = np.arctan2(y, x_deproj) - arc_params[0, 3]
    phi = (phi + np.pi) % (2 * np.pi) - np.pi
    image = (
        gauss_params[0, 0] * np.exp(-(radius ** 2.) / (2. * gauss_params[0, 1] ** 2.))
        + ring_params[0, 0] * np.exp(-((radius - ring_params[0, 1]) ** 2.) / (2. * ring_params[0, 2] ** 2.))
        + ring_params[1, 0] * np.exp(-((radius - ring_params[1, 1]) ** 2.) / (2. * ring_params[1, 2] ** 2.))
        + arc_params[0, 0] * np.exp(-((radius - arc_params[0, 1]) ** 2.) / (2. * arc_params[0, 2] ** 2.))
        * np.exp(-(phi ** 2.) / (2. * arc_params[0, 4] ** 2.))
    )
    x_obs, _, weights = generate_random_vis(nsamples, real_type)

    return {
        'image': image.astype(real_type),
        'gauss_params': gauss_params,
        'ring_params': ring_params,
        'arc_params': arc_params,
        'inc': inc,
        'dxy': dxy,
        'u': udat,
        'v': vdat,
        'vis_obs_re': x_obs.real.copy(),
        'vis_obs_im': x_obs.imag.copy(),
        'weights': weights,
        'dRA': -0.09 * arcsec,
        'dDec': 0.07 * arcsec,
        'PA': 13. * deg,
    }


def test_double_cuda_uses_nanobind_compat_layer_when_available():
    if not (galario.HAVE_CUDA and int(environ.get("GALARIO_TEST_GPU", 0)) and getattr(galario, "HAVE_NANOBIND", False)):
        pytest.skip("nanobind compat layer only applies to CUDA builds with the experimental binding available")

    assert g_double.create_image_context.__module__ == 'galario.nanobind_double_cuda'
    assert g_double.sampleImage.__module__ == 'galario.nanobind_double_cuda'
    assert g_double.sampleProfile.__module__ == 'galario.nanobind_double_cuda'
    assert g_double.chi2Image.__module__ in {'galario.nanobind_double_cuda', 'galario._nanobind_double_cuda'}
    assert g_double.chi2Profile.__module__ in {'galario.nanobind_double_cuda', 'galario._nanobind_double_cuda'}


def test_double_uses_nanobind_compat_layer_when_available():
    if not getattr(galario, "HAVE_NANOBIND", False):
        pytest.skip("nanobind compat layer not available")

    assert g_double_cpu.sampleImage.__module__ == 'galario.nanobind_double'
    assert g_double_cpu.sampleProfile.__module__ == 'galario.nanobind_double'
    assert g_double_cpu.chi2Image.__module__ in {'galario.nanobind_double', 'galario._nanobind_double'}
    assert g_double_cpu.chi2Profile.__module__ in {'galario.nanobind_double', 'galario._nanobind_double'}


def _complex_imag_atol(vis, rtol):
    return max(np.abs(np.mean(vis.real)) * rtol, 1e-12)


def _relative_complex_error(reference, candidate):
    denom = np.linalg.norm(reference)
    if denom == 0:
        return 0.
    return np.linalg.norm(candidate - reference) / denom


########################################################
#                                                      #
#                      TESTS                           #
#                                                      #
########################################################

@pytest.mark.parametrize("Rmin, dR, nrad, nxy, dxy, inc, profile_mode, real_type",
                          [(1e-6, 0.001, 2000, 1024, 0.2, 20., 'Gauss', 'float64'),
                           (1e-6, 0.001, 2000, 2048, 0.2, 44.23, 'Cos-Gauss', 'float64'),
                           (1e-6, 0.001, 2000, 2048, 0.5, 20., 'Gauss', 'float64'),
                           (1e-6, 0.001, 2000, 1024, 0.3, 20., 'Gauss', 'float64')],
                          ids=["{}".format(i) for i in range(4)])
def test_intensity_sweep(Rmin, dR, nrad, nxy, dxy, inc, profile_mode, real_type):
    """
    Test the image creation algorithm, `sweep`.

    """
    Rmin *= arcsec
    dR *= arcsec
    dxy *= arcsec
    inc = np.radians(inc)

    # compute radial profile
    intensity = radial_profile(Rmin, dR, nrad, profile_mode, dtype=real_type, gauss_width=dxy*6)

    nrow, ncol = nxy, nxy

    image_ref = sweep_ref(intensity, Rmin, dR, nrow, ncol, dxy, inc, dtype_image=real_type)

    image_sweep_galario = g_double.sweep(intensity, Rmin, dR, nxy, dxy, inc)

    image_prototype = g_sweep_prototype(intensity, Rmin, dR, nrow, ncol, dxy, inc, dtype_image=real_type)

    # uncomment for debugging
    # plot images
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.matshow(image_sweep_galario)
    # plt.savefig("./test_intensity_sweep_galario.pdf")
    # plt.clf()
    # plt.matshow(image_ref)
    # plt.savefig("./test_intensity_sweep_ref.pdf")

    # plot cuts - benchmark
    # for line_no in [0, nx//2-1, nx//2, nx//2+1, nx-1]:
    #     imin, imax = nx//2 - 10, nx//2 + 10
    #     # plt.plot(image_ref[line_no, imin:imax], '.-', label=line_no)
    #     # plt.plot(image_g_sweep_prototype[line_no, imin:imax], '.--', ms=3, lw=0.3, label=line_no)
    #     plt.plot(image_ref[line_no, imin:imax]-image_g_sweep_prototype[line_no, imin:imax], '.--', ms=3, lw=0.3, label=line_no)
    # plt.legend()
    # plt.savefig("./profile_intensity_ref.pdf")
    # plt.clf()

    assert_allclose(image_ref, image_prototype, rtol=1.e-12, atol=0)
    assert_allclose(image_prototype, image_sweep_galario, rtol=1.e-12, atol=0)


@pytest.mark.parametrize("nsamples, real_type, rtol, atol, acc_lib, pars",
                          [(1000, 'float64', 1e-6, 0, g_double, par1),
                          (1000, 'float64',  1e-6, 0, g_double, par2),
                          (1000, 'float64',  1e-6, 0, g_double, par3),
                          (1000, 'float64',  1e-6, 0, g_double, par4)],
                         ids=["{}".format(i) for i in range(4)])
def test_R2C_vs_C2C(nsamples, real_type, rtol, atol, acc_lib, pars):
    """
    Test the (current) R2C implementation against the (old) C2C one.
    # it is possible that 0.0002% of points differ at rtol>1e-5

    """
    if CUDA_DIRECT_IMAGE_BACKEND:
        pytest.skip("CUDA sampleImage now uses direct Fourier summation instead of the legacy R2C interpolation path.")

    dRA = pars['dRA']
    dDec = pars['dDec']
    PA = pars['PA']
    nxy = pars['nxy']

    # generate the samples
    maxuv_generator = 3.e3
    udat, vdat = create_sampling_points(nsamples, maxuv_generator, dtype=real_type)

    # compute the matrix nxy and maxuv
    _, minuv, maxuv = matrix_size(udat, vdat)
    du = maxuv/nxy

    # create model image (it happens to have 0 imaginary part)
    reference_image = create_reference_image(nxy, -5., 2., dtype=real_type)
    ref_real = reference_image.copy()

    # CPU version
    PA *= deg
    dRA *= arcsec
    dDec *= arcsec
    dRArot, dDecrot, urot, vrot = apply_rotation(PA, dRA, dDec, udat, vdat)
    dRArot_g, dDecrot_g, urot_g, vrot_g = acc_lib.uv_rotate(PA, dRA, dDec, udat, vdat)

    np.testing.assert_allclose(dRArot, dRArot_g)
    np.testing.assert_allclose(dDecrot, dDecrot_g)
    np.testing.assert_allclose(urot, urot_g)
    np.testing.assert_allclose(vrot, vrot_g)

    #  1) C2C (numpy)
    fft_c2c_shifted = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(reference_image.copy())))
    uroti_c2c, vroti_c2c = uv_idx(urot, vrot, du, nxy/2.)
    ReInt_c2c = int_bilin_MT(fft_c2c_shifted.real, uroti_c2c, vroti_c2c)
    ImInt_c2c = int_bilin_MT(fft_c2c_shifted.imag, uroti_c2c, vroti_c2c)
    AmpInt_c2c = int_bilin_MT(np.abs(fft_c2c_shifted), uroti_c2c, vroti_c2c)
    PhaseInt_c2c = np.angle(ReInt_c2c + 1j*ImInt_c2c)
    vis_c2c = AmpInt_c2c * (np.cos(PhaseInt_c2c) + 1j*np.sin(PhaseInt_c2c))
    vis_c2c_shifted = apply_phase_array(urot, vrot, vis_c2c, dRArot, dDecrot)

    # CPU/GPU version (galario)
    dxy = 1./nxy/du
    vis_galario = acc_lib.sampleImage(ref_real, dxy, udat, vdat, dRA=dRA, dDec=dDec, PA=PA)

    # check python c2c vs galario
    assert_allclose(vis_galario.real, vis_c2c_shifted.real, rtol, atol)
    assert_allclose(vis_galario.imag, vis_c2c_shifted.imag, rtol, np.abs(np.mean(vis_galario.real))*rtol)


@pytest.mark.parametrize("size, real_type, complex_type, rtol, atol, acc_lib",
                         [(1024, 'float64', 'complex128', 1e-16, 1e-8, g_double)],
                         ids=["DP"])
def test_interpolate(size, real_type, complex_type, rtol, atol, acc_lib):
    """
    Test the interpolation of the output FT.

    """
    nsamples = 10000
    maxuv = 1000.

    reference_image = create_reference_image(size=size, dtype=real_type)
    udat, vdat = create_sampling_points(nsamples, maxuv/2.2)
    # this factor has to be > than 2 because the matrix cover between -maxuv/2 to +maxuv/2,
    # therefore the sampling points have to be contained inside.

    udat = udat.astype(real_type)
    vdat = vdat.astype(real_type)

    # no rotation
    du = maxuv/size
    uroti, vroti = uv_idx_r2c(udat, vdat, du, size/2.)

    uroti = uroti.astype(real_type)
    vroti = vroti.astype(real_type)

    ft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(reference_image))).astype(complex_type, order='C')

    ReInt = int_bilin_MT(ft.real, uroti, vroti)
    ImInt = int_bilin_MT(ft.imag, uroti, vroti)
    AmpInt = int_bilin_MT(np.abs(ft), uroti, vroti)
    uneg = udat < 0.
    ImInt[uneg] *= -1.
    PhaseInt = np.angle(ReInt + 1j*ImInt)
    
    ReInt = AmpInt * np.cos(PhaseInt)
    ImInt = AmpInt * np.sin(PhaseInt)

    complexInt = acc_lib.interpolate(ft, du,
                                     udat.astype(real_type),
                                     vdat.astype(real_type))

    assert_allclose(ReInt, complexInt.real, rtol, atol)
    assert_allclose(ImInt, complexInt.imag, rtol, atol)


@pytest.mark.parametrize("size, real_type, rtol, atol, acc_lib",
                         [(1024, 'float64', 1.e-16, 1e-8, g_double)],
                         ids=["DP"])
def test_FFT(size, real_type, rtol, atol, acc_lib):
    """
    Test the Real to Complex FFTW/cuFFT against numpy Complex to Complex.

    """
    reference_image = create_reference_image(size=size, dtype=real_type)

    ft = np.fft.fft2(reference_image)

    acc_res = acc_lib._fft2d(reference_image)

    # outputs of different shape because np doesn't use the redundancy y[i] == y[n-i] for i>0
    np.testing.assert_equal(ft.shape[0], acc_res.shape[0])
    np.testing.assert_equal(acc_res.shape[1], int(acc_res.shape[0]/2)+1)

    # some real parts can be very close to zero, so we need atol > 0!
    # only get the 0-th and the first half of columns to compare to compact FFTW output
    assert_allclose(unique_part(ft).real, acc_res.real, rtol, atol)
    assert_allclose(unique_part(ft).imag, acc_res.imag, rtol, atol)


@pytest.mark.parametrize("size, real_type, tol, acc_lib",
                         [(1024, 'float64', 1.e-16, g_double)],
                         ids=["DP"])
def test_shift_axes01(size, real_type, tol, acc_lib):
    """
    Test the 1st shift to be applied to the input image before the FFT.

    """
    # just a create a runtime-typical image with a big offset disk
    reference_image = create_reference_image(size=size, x0=size/10., y0=-size/10.,
                                            sigma_x=3.*size, sigma_y=2.*size, dtype=real_type)

    npshifted = np.fft.fftshift(reference_image)

    ref_complex = reference_image.copy()
    acc_shift_real = acc_lib._fftshift(ref_complex)

    # interpret complex array as real and skip last two columns
    real_view = acc_shift_real.view(dtype=real_type)[:, :-2]

    assert_allclose(npshifted, real_view, rtol=tol)


@pytest.mark.parametrize("size, complex_type, tol, acc_lib",
                         [(1024, 'complex128', 1.e-16, g_double)],
                         ids=["DP"])
def test_shift_axis0(size, complex_type, tol, acc_lib):
    """
    Test the 2nd shift to be applied to the output of the FFT.

    """
    #  the reference image has the shape of the typical output of FFTW R2C,
    #  but acc_lib.fftshift_axis0() works for every matrix size.
    reference_image = np.random.random((size, int(size/2)+1)).astype(complex_type)

    # numpy reference
    npshifted = np.fft.fftshift(reference_image, axes=0)

    ref_complex = reference_image.copy()
    acc_lib._fftshift_axis0(ref_complex)
    assert_allclose(npshifted, ref_complex, rtol=tol)


@pytest.mark.parametrize("real_type, complex_type, rtol, atol, acc_lib, pars",
                         [('float64', 'complex128', 1.e-16, 1e-13, g_double, par1),
                          ('float64', 'complex128', 1.e-16, 1e-13, g_double, par2),
                          ('float64', 'complex128', 1.e-16, 1e-13, g_double, par3)],
                         ids=["DP_par1", "DP_par2", "DP_par3"])
def test_apply_phase_vis(real_type, complex_type, rtol, atol, acc_lib, pars):
    """
    Test apply phase to visibilities

    """
    dRA = pars.get('dRA', 0.4)
    dDec = pars.get('dDec', 10.)

    dRA *= arcsec
    dDec *= arcsec

    # generate the samples
    nsamples = 10000
    maxuv_generator = 3.e3
    udat, vdat = create_sampling_points(nsamples, maxuv_generator, dtype=real_type)

    # generate mock visibility values
    vis_int = np.zeros(nsamples, dtype=complex_type)
    vis_int.real = np.random.random(nsamples) * 10.
    vis_int.imag = np.random.random(nsamples) * 30.

    vis_int_numpy = apply_phase_array(udat, vdat, vis_int.copy(), dRA, dDec)

    vis_int_shifted = acc_lib.apply_phase_vis(dRA, dDec, udat, vdat, vis_int)

    assert_allclose(vis_int_numpy.real, vis_int_shifted.real, rtol, atol)
    assert_allclose(vis_int_numpy.imag, vis_int_shifted.imag, rtol, atol)


@pytest.mark.parametrize("nsamples, real_type, tol, acc_lib",
                         [(1000, 'float64', 1.e-15, g_double)],
                         ids=["DP"])
def test_reduce_chi2(nsamples, real_type, tol, acc_lib):
    """
    Test chi2 reduction

    """
    x, y, w = generate_random_vis(nsamples, real_type)
    chi2_ref = np.sum(((x.real - y.real) ** 2. + (x.imag - y.imag)**2.) * w)

    chi2_loc = acc_lib.reduce_chi2(x.real.copy(order='C'), x.imag.copy(order='C'), w, y.copy())

    assert_allclose(chi2_ref, chi2_loc, rtol=tol)


@pytest.mark.parametrize("nsamples, real_type, rtol, acc_lib",
                         [(1000, 'float64', 1.e-10, g_double)],
                         ids=["DP"])
def test_image_origin(nsamples, real_type, rtol, acc_lib):
    def model1(R):
        y = (np.exp(-(R / (0.2 * arcsec)) ** 2) + 0.3 * np.exp(
            -((R - 0.4 * arcsec) / ((0.15 * arcsec))) ** 2))
        return 1e12 * y

    def model2(R):
        y = 1 * np.exp(
            -((R - 1 * arcsec) / (0.5 * arcsec)) ** 2) + 0.7 * np.exp(
            -((R - 2.5 * arcsec) / (0.25 * arcsec)) ** 2) + 0.2 * np.exp(
            -((R - 3.5 * arcsec) / (0.15 * arcsec)) ** 2)

        return 1e12 * y

    def model3(R):
        y = 1 * np.exp(-((R - 0.5 * arcsec) / ((0.1 * arcsec))) ** 2)
        return 1e12 * y

    def model4(R):
        y = 1 * (R / 2. / arcsec) ** -0.05 * np.exp(-(R / 2. / arcsec) ** 4)
        return 1e12 * y

    if CUDA_DIRECT_IMAGE_BACKEND:
        nsamples = min(nsamples, 256)

    # u, v points
    maxuv_generator = 3e3
    udat, vdat = create_sampling_points(nsamples, maxuv_generator,
                                        dtype=real_type)
    full_fov = 4096 * 6.42956326721e-08
    if CUDA_DIRECT_IMAGE_BACKEND:
        nxy = 512
        dxy = full_fov / nxy
        sample_image_ref = py_sampleImage_direct
        sample_image_backend = galario.BACKEND_DFT
    else:
        nxy, dxy = 4096, 6.42956326721e-08
        sample_image_ref = py_sampleImage
        sample_image_backend = galario.BACKEND_AUTO

    # radial grid
    Rmin = 0.00001 * arcsec
    dR = 0.0001 * arcsec
    nrad = 2000
    gridrad = np.linspace(Rmin, Rmin + dR * (nrad - 1), nrad)

    # create sample image with origin='upper'
    image_asym = sweep_ref(model1(gridrad), Rmin, dR, nxy, nxy, dxy,  0  * deg, Dx=-50.*dxy,  Dy=66.*dxy,   dtype_image=real_type) + \
                 sweep_ref(model2(gridrad), Rmin, dR, nxy, nxy, dxy, 20. * deg, Dx=+150.*dxy, Dy=+250.*dxy, dtype_image=real_type) + \
                 sweep_ref(model3(gridrad), Rmin, dR, nxy, nxy, dxy, 35. * deg, Dx=-110.*dxy, Dy=-100.*dxy, dtype_image=real_type) + \
                 sweep_ref(model4(gridrad), Rmin, dR, nxy, nxy, dxy, 44. * deg, Dx=-110.*dxy, Dy=-100.*dxy, dtype_image=real_type)

    # create sample image with origin='lower'
    image_asym2 = sweep_ref(model1(gridrad), Rmin, dR, nxy, nxy, dxy,   0 * deg, Dx=-50.*dxy,  Dy=66.*dxy,   dtype_image=real_type, origin='lower') + \
                  sweep_ref(model2(gridrad), Rmin, dR, nxy, nxy, dxy, 20. * deg, Dx=+150.*dxy, Dy=+250.*dxy, dtype_image=real_type, origin='lower') + \
                  sweep_ref(model3(gridrad), Rmin, dR, nxy, nxy, dxy, 35. * deg, Dx=-110.*dxy, Dy=-100.*dxy, dtype_image=real_type, origin='lower') + \
                  sweep_ref(model4(gridrad), Rmin, dR, nxy, nxy, dxy, 44. * deg, Dx=-110.*dxy, Dy=-100.*dxy, dtype_image=real_type, origin='lower')

    # check that the images are flipped and rolled when diffent origin option is used
    assert_allclose(image_asym2, np.roll(np.flipud(image_asym), 1, 0), atol=0, rtol=rtol)

    # remove spurious values
    image_asym[0, :] = 0.
    image_asym[:, 0] = 0.
    image_asym[np.where(image_asym < 1e-10)] = 0.

    # remove spurious values
    image_asym2[0, :] = 0.
    image_asym2[:, 0] = 0.
    image_asym2[np.where(image_asym2 < 1e-10)] = 0.

    # Compute visibilities of ORIGINAL image with CURRENT GALARIO algorithm (only: origin='upper')
    vis_C_upper_image_upper = acc_lib.sampleImage(
        image_asym, dxy, udat, vdat, dRA=0.5, dDec=-3., PA=10.,
        backend=sample_image_backend)

    # Compute visibilities of ORIGINAL image with NEW algorithm, origin='upper'
    vis_py_upper_image_upper = sample_image_ref(image_asym, dxy, udat, vdat, dRA=0.5, dDec=-3., PA=10., origin='upper')

    # Compute visibilities of LOWER ORIGIN image with NEW algorithm, origin='lower'
    vis_py_lower_image_lower = sample_image_ref(image_asym2, dxy, udat, vdat, dRA=0.5, dDec=-3., PA=10., origin='lower')

    # Compute with C implementation
    vis_C_lower_image_lower = acc_lib.sampleImage(
        image_asym2, dxy, udat, vdat, dRA=0.5, dDec=-3., PA=10.,
        origin='lower', backend=sample_image_backend)

    # check that they produce all the same visibilities
    assert_allclose(vis_py_upper_image_upper, vis_C_lower_image_lower, atol=0., rtol=rtol)
    assert_allclose(vis_py_upper_image_upper, vis_C_upper_image_upper, atol=0., rtol=rtol)
    assert_allclose(vis_py_lower_image_lower, vis_C_upper_image_upper, atol=0., rtol=rtol)
    assert_allclose(vis_C_lower_image_lower, vis_C_upper_image_upper, atol=0., rtol=rtol)



@pytest.mark.parametrize("nsamples, real_type, rtol, atol, acc_lib, pars",
                          [(int(1e3), 'float64', 1e-6, 0, g_double, par1),
                          (int(1e3), 'float64', 1e-6, 0, g_double, par2),
                          (int(1e3), 'float64', 1e-6, 0, g_double, par3),
                          (int(1e3), 'float64', 1e-6, 0, g_double, par4)],
                         ids=["{}".format(i) for i in range(4)])
def test_all(nsamples, real_type, rtol, atol, acc_lib, pars):
    """
    Main test function: tests Python vs galario implementation of sampleImage,
    sampleProfile, chi2Image, chi2Profile.

    For the imaginary part the test has atol=np.abs(np.mean(vis_g_sampleImage.real))*rtol.
    The reason is that for symmetric images the imaginary part of FFT can fluctuate quite a lot
    this manual absolute tolerance checks that such fluctuations are small compared to the real part.

    """
    dRA = pars['dRA']
    dDec = pars['dDec']
    PA = pars['PA']
    nxy = pars['nxy']

    # generate the samples
    maxuv_generator = 3.e3
    udat, vdat = create_sampling_points(nsamples, maxuv_generator, dtype=real_type)

    _, minuv, maxuv = matrix_size(udat, vdat)

    dxy = 1. / maxuv # pixel size (rad)
    # create intensity profile and model image
    Rmin, dR, nrad, inc, profile_mode, real_type = dxy/100., dxy/10.5, 10000, 20., 'Gauss', 'float64',
    dRA *= arcsec
    dDec *= arcsec
    PA *= deg
    inc *= deg

    intensity = radial_profile(Rmin, dR, nrad, profile_mode, dtype=real_type, gauss_width=dxy*10)
    reference_image = sweep_ref(intensity, Rmin, dR, nxy, nxy, dxy, inc, dtype_image=real_type)

    # test sampleImage
    if CUDA_DIRECT_IMAGE_BACKEND:
        image_nsamples = min(nsamples, 128)
        image_nxy = min(nxy, 256)
        image_udat = udat[:image_nsamples]
        image_vdat = vdat[:image_nsamples]
        image_reference = create_reference_image(image_nxy, -5., 2., dtype=real_type)

        vis_py_sampleImage = py_sampleImage_direct(image_reference, dxy, image_udat, image_vdat, PA=PA, dRA=dRA, dDec=dDec)
        vis_g_sampleImage = acc_lib.sampleImage(
            image_reference, dxy, image_udat, image_vdat,
            PA=PA, dRA=dRA, dDec=dDec, backend=galario.BACKEND_DFT)
    else:
        image_udat = udat
        image_vdat = vdat
        image_reference = reference_image
        vis_py_sampleImage = py_sampleImage(reference_image, dxy, udat, vdat, PA=PA, dRA=dRA, dDec=dDec)
        vis_g_sampleImage = acc_lib.sampleImage(reference_image, dxy, udat, vdat, PA=PA, dRA=dRA, dDec=dDec)

    assert_allclose(vis_py_sampleImage.real, vis_g_sampleImage.real, rtol=rtol, atol=atol)
    assert_allclose(vis_py_sampleImage.imag, vis_g_sampleImage.imag, rtol=rtol, atol=np.abs(np.mean(vis_g_sampleImage.real))*rtol)

    # test sampleProfile
    vis_py_sampleProfile = py_sampleProfile(intensity.copy(), Rmin, dR, nxy, dxy, udat, vdat, inc=inc, dRA=dRA, dDec=dDec, PA=PA)
    vis_g_sampleProfile = acc_lib.sampleProfile(intensity, Rmin, dR, nxy, dxy, udat, vdat, inc=inc, dRA=dRA, dDec=dDec, PA=PA)

    # check galario vs python implementation
    assert_allclose(vis_g_sampleProfile.real, vis_py_sampleProfile.real, rtol=rtol, atol=atol)
    assert_allclose(vis_g_sampleProfile.imag, vis_py_sampleProfile.imag, rtol=rtol, atol=np.abs(np.mean(vis_g_sampleProfile.real))*rtol)

    # test chi2Image
    x, _, w = generate_random_vis(nsamples, real_type)

    if CUDA_DIRECT_IMAGE_BACKEND:
        x_image, _, w_image = generate_random_vis(len(image_udat), real_type)
        chi2_pychi2Image = py_chi2Image_direct(image_reference, dxy, image_udat, image_vdat,
                                               x_image.real.copy(), x_image.imag.copy(), w_image,
                                               dRA=dRA, dDec=dDec, PA=PA)
        chi2_g_chi2Image = acc_lib.chi2Image(image_reference, dxy, image_udat, image_vdat,
                                             x_image.real.copy(), x_image.imag.copy(), w_image,
                                             dRA=dRA, dDec=dDec, PA=PA,
                                             backend=galario.BACKEND_DFT)
    else:
        chi2_pychi2Image = py_chi2Image(reference_image, dxy, udat, vdat, x.real.copy(), x.imag.copy(), w, dRA=dRA, dDec=dDec)
        chi2_g_chi2Image = acc_lib.chi2Image(reference_image, dxy, udat, vdat, x.real.copy(), x.imag.copy(), w, dRA=dRA, dDec=dDec)

    # test chi2Profile
    chi2_pychi2Profile = py_chi2Profile(intensity, Rmin, dR, nxy, dxy, udat, vdat, x.real.copy(), x.imag.copy(), w, inc=inc, dRA=dRA, dDec=dDec)
    chi2_g_chi2Profile = acc_lib.chi2Profile(intensity, Rmin, dR, nxy, dxy, udat, vdat, x.real.copy(), x.imag.copy(), w, inc=inc, dRA=dRA, dDec=dDec)

    # check galario vs python implementation
    assert_allclose(chi2_pychi2Profile, chi2_g_chi2Profile, rtol=rtol, atol=atol)
    assert_allclose(chi2_pychi2Image, chi2_g_chi2Image, rtol=rtol, atol=atol)

def test_backend_constants_exposed():
    assert galario.BACKEND_AUTO == 'auto'
    assert galario.BACKEND_FFT == 'fft'
    assert galario.BACKEND_DFT == 'dft'
    assert galario.BACKEND_NUFFT == 'nufft'


@pytest.mark.parametrize("backend", IMAGE_BACKENDS, ids=["fft", "dft", "nufft"])
def test_sample_image_backends(backend):
    case = _make_image_backend_case()
    image = case['image']
    dxy = case['dxy']
    udat = case['u']
    vdat = case['v']
    dRA = case['dRA']
    dDec = case['dDec']
    PA = case['PA']

    vis = g_double.sampleImage(image, dxy, udat, vdat,
                               dRA=dRA, dDec=dDec, PA=PA,
                               backend=backend, nufft_oversample=2.0)

    if backend == galario.BACKEND_FFT:
        vis_ref = py_sampleImage(image, dxy, udat, vdat, dRA=dRA, dDec=dDec, PA=PA)
        rtol = 1e-6
    elif backend == galario.BACKEND_DFT:
        vis_ref = py_sampleImage_direct(image, dxy, udat, vdat, dRA=dRA, dDec=dDec, PA=PA)
        rtol = 5e-7
    else:
        vis_ref = g_double.sampleImage(image, dxy, udat, vdat,
                                       dRA=dRA, dDec=dDec, PA=PA,
                                       backend=galario.BACKEND_DFT)
        assert _relative_complex_error(vis_ref, vis) < 2e-2
        return

    assert_allclose(vis.real, vis_ref.real, rtol=rtol, atol=1e-10)
    assert_allclose(vis.imag, vis_ref.imag, rtol=rtol, atol=_complex_imag_atol(vis_ref, rtol))


@pytest.mark.parametrize("backend", IMAGE_BACKENDS, ids=["fft", "dft", "nufft"])
def test_chi2_image_cached_backends(backend):
    case = _make_image_backend_case()
    image = case['image']
    dxy = case['dxy']
    udat = case['u']
    vdat = case['v']
    vis_obs_re = case['vis_obs_re']
    vis_obs_im = case['vis_obs_im']
    weights = case['weights']
    dRA = case['dRA']
    dDec = case['dDec']
    PA = case['PA']

    chi2_uncached = g_double.chi2Image(image, dxy, udat, vdat, vis_obs_re, vis_obs_im, weights,
                                       dRA=dRA, dDec=dDec, PA=PA,
                                       backend=backend, nufft_oversample=2.0)
    ctx = g_double.create_image_context(image.shape[0], image.shape[1], udat, vdat,
                                          vis_obs_re, vis_obs_im, weights,
                                          backend=backend, nufft_oversample=2.0)
    chi2_cached = g_double.chi2_image(ctx=ctx, image=image, dxy=dxy, dRA=dRA, dDec=dDec, PA=PA)

    assert ctx.shape == image.shape
    assert_allclose(chi2_cached, chi2_uncached, rtol=1e-12, atol=1e-12)


def test_cpu_cached_fft_context_survives_thread_count_change():
    case = _make_image_backend_case()
    image = case['image']
    ctx = g_double_cpu.create_image_context(
        image.shape[0], image.shape[1], case['u'], case['v'],
        case['vis_obs_re'], case['vis_obs_im'], case['weights'],
        backend=galario.BACKEND_FFT,
    )
    original_threads = g_double_cpu.threads()
    try:
        g_double_cpu.threads(1)
        chi2_one_thread = g_double_cpu.chi2_image(
            ctx=ctx, image=image, dxy=case['dxy'],
            dRA=case['dRA'], dDec=case['dDec'], PA=case['PA'],
        )
        g_double_cpu.threads(2)
        chi2_two_threads = g_double_cpu.chi2_image(
            ctx=ctx, image=image, dxy=case['dxy'],
            dRA=case['dRA'], dDec=case['dDec'], PA=case['PA'],
        )
    finally:
        g_double_cpu.threads(original_threads)

    assert_allclose(chi2_two_threads, chi2_one_thread, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("backend", IMAGE_BACKENDS, ids=["fft", "dft", "nufft"])
def test_component_model_apis_match_python_image(backend):
    case = _make_component_case()
    image = case['image']
    gauss_params = case['gauss_params']
    ring_params = case['ring_params']
    arc_params = case['arc_params']
    inc = case['inc']
    dxy = case['dxy']
    udat = case['u']
    vdat = case['v']
    vis_obs_re = case['vis_obs_re']
    vis_obs_im = case['vis_obs_im']
    weights = case['weights']
    dRA = case['dRA']
    dDec = case['dDec']
    PA = case['PA']

    vis_ref = g_double.sampleImage(image, dxy, udat, vdat, dRA=dRA, dDec=dDec, PA=PA,
                                   backend=backend, nufft_oversample=2.0)
    vis_components = g_double.sampleImageComponents(image.shape[0], image.shape[1], dxy, udat, vdat,
                                                    gauss_params=gauss_params, ring_params=ring_params,
                                                    arc_params=arc_params, inc=inc,
                                                    dRA=dRA, dDec=dDec, PA=PA,
                                                    backend=backend, nufft_oversample=2.0)
    assert_allclose(vis_components.real, vis_ref.real, rtol=1e-10, atol=1e-10)
    assert_allclose(vis_components.imag, vis_ref.imag, rtol=1e-10, atol=1e-10)

    chi2_ref = g_double.chi2Image(image, dxy, udat, vdat, vis_obs_re, vis_obs_im, weights,
                                  dRA=dRA, dDec=dDec, PA=PA,
                                  backend=backend, nufft_oversample=2.0)
    chi2_components = g_double.chi2ImageComponents(image.shape[0], image.shape[1], dxy, udat, vdat,
                                                   vis_obs_re, vis_obs_im, weights,
                                                   gauss_params=gauss_params, ring_params=ring_params,
                                                   arc_params=arc_params, inc=inc,
                                                   dRA=dRA, dDec=dDec, PA=PA,
                                                   backend=backend, nufft_oversample=2.0)
    ctx = g_double.create_image_context(image.shape[0], image.shape[1], udat, vdat,
                                          vis_obs_re, vis_obs_im, weights,
                                          backend=backend, nufft_oversample=2.0)
    chi2_cached_components = g_double.chi2_image(ctx=ctx, dxy=dxy,
                                                                gauss_params=gauss_params,
                                                                ring_params=ring_params,
                                                                arc_params=arc_params,
                                                                inc=inc,
                                                                dRA=dRA, dDec=dDec, PA=PA,
                                                                origin='lower')

    assert_allclose(chi2_components, chi2_ref, rtol=1e-10, atol=1e-10)
    assert_allclose(chi2_cached_components, chi2_ref, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("backend", IMAGE_BACKENDS, ids=["fft", "dft", "nufft"])
def test_chi2_image_cached_components_batch_matches_scalar(backend):
    case = _make_component_case()
    dxy = case['dxy']
    udat = case['u']
    vdat = case['v']
    vis_obs_re = case['vis_obs_re']
    vis_obs_im = case['vis_obs_im']
    weights = case['weights']
    ctx = g_double.create_image_context(case['image'].shape[0], case['image'].shape[1], udat, vdat,
                                          vis_obs_re, vis_obs_im, weights,
                                          backend=backend, nufft_oversample=2.0)

    gauss_batch = np.vstack([case['gauss_params'].reshape(-1), (case['gauss_params'] * np.array([[1.05, 0.95]])).reshape(-1)])
    ring_batch = np.vstack([case['ring_params'].reshape(-1), (case['ring_params'] * np.array([[0.97, 1.02, 1.03], [1.01, 0.98, 1.02]])).reshape(-1)])
    arc_batch = np.vstack([case['arc_params'].reshape(-1), (case['arc_params'] * np.array([[1.04, 1.01, 0.96, 1.0, 1.02]])).reshape(-1)])
    inc_batch = np.array([case['inc'], case['inc'] * 1.03])
    dRA_batch = np.array([case['dRA'], case['dRA'] * 0.8])
    dDec_batch = np.array([case['dDec'], case['dDec'] * 1.2])
    PA_batch = np.array([case['PA'], case['PA'] * 0.9])

    chi2_scalar = np.array([
        g_double.chi2_image(ctx=ctx, dxy=dxy,
                                           gauss_params=gauss_batch[i].reshape(case['gauss_params'].shape),
                                           ring_params=ring_batch[i].reshape(case['ring_params'].shape),
                                           arc_params=arc_batch[i].reshape(case['arc_params'].shape),
                                           inc=inc_batch[i], dRA=dRA_batch[i], dDec=dDec_batch[i], PA=PA_batch[i],
                                           origin='lower')
        for i in range(2)
    ])
    chi2_batch = g_double.chi2_image(ctx=ctx, dxy=dxy,
                                                         gauss_params_batch=gauss_batch,
                                                         ring_params_batch=ring_batch,
                                                         arc_params_batch=arc_batch,
                                                         inc_batch=inc_batch,
                                                         dRA_batch=dRA_batch,
                                                         dDec_batch=dDec_batch,
                                                         PA_batch=PA_batch,
                                                         origin='lower')
    assert_allclose(chi2_batch, chi2_scalar, rtol=1e-10, atol=1e-10)


def test_chi2_image_cached_components_batch_auto_matches_scalar():
    case = _make_component_case()
    dxy = case['dxy']
    udat = case['u'][:256]
    vdat = case['v'][:256]
    vis_obs_re = case['vis_obs_re'][:256]
    vis_obs_im = case['vis_obs_im'][:256]
    weights = case['weights'][:256]
    ctx = g_double.create_image_context(case['image'].shape[0], case['image'].shape[1], udat, vdat,
                                          vis_obs_re, vis_obs_im, weights,
                                          backend=galario.BACKEND_AUTO, nufft_oversample=2.0)

    gauss_batch = np.vstack([case['gauss_params'].reshape(-1), case['gauss_params'].reshape(-1)])
    ring_batch = np.vstack([case['ring_params'].reshape(-1), (case['ring_params'] * np.array([[0.98, 1.01, 1.02], [1.0, 0.99, 1.01]])).reshape(-1)])
    arc_batch = np.vstack([case['arc_params'].reshape(-1), (case['arc_params'] * np.array([[1.01, 1.02, 0.99, 1.0, 1.01]])).reshape(-1)])
    inc_batch = np.array([case['inc'], case['inc'] * 1.01])
    dRA_batch = np.array([case['dRA'], case['dRA'] * 0.9])
    dDec_batch = np.array([case['dDec'], case['dDec'] * 1.1])
    PA_batch = np.array([case['PA'], case['PA'] * 0.95])

    chi2_scalar = np.array([
        g_double.chi2_image(ctx=ctx, dxy=dxy,
                                           gauss_params=gauss_batch[i].reshape(case['gauss_params'].shape),
                                           ring_params=ring_batch[i].reshape(case['ring_params'].shape),
                                           arc_params=arc_batch[i].reshape(case['arc_params'].shape),
                                           inc=inc_batch[i], dRA=dRA_batch[i], dDec=dDec_batch[i], PA=PA_batch[i],
                                           origin='lower')
        for i in range(2)
    ])
    chi2_batch = g_double.chi2_image(ctx=ctx, dxy=dxy,
                                                         gauss_params_batch=gauss_batch,
                                                         ring_params_batch=ring_batch,
                                                         arc_params_batch=arc_batch,
                                                         inc_batch=inc_batch,
                                                         dRA_batch=dRA_batch,
                                                         dDec_batch=dDec_batch,
                                                         PA_batch=PA_batch,
                                                         origin='lower')
    assert_allclose(chi2_batch, chi2_scalar, rtol=1e-10, atol=1e-10)


def test_chi2_image_context_backend_properties():
    case = _make_component_case()
    ctx = g_double.create_image_context(case['image'].shape[0], case['image'].shape[1],
                                          case['u'][:256], case['v'][:256],
                                          case['vis_obs_re'][:256], case['vis_obs_im'][:256], case['weights'][:256],
                                          backend=galario.BACKEND_AUTO, nufft_oversample=2.0)

    assert ctx.requested_backend == galario.BACKEND_AUTO
    assert ctx.resolved_backend in {galario.BACKEND_FFT, galario.BACKEND_DFT, galario.BACKEND_NUFFT}
    assert ctx.batch_backend(32) in {galario.BACKEND_FFT, galario.BACKEND_DFT, galario.BACKEND_NUFFT}


def test_sample_profile_matches_analytic_circular_gaussian():
    """Guard the radial transform against a known closed-form solution."""
    sigma = 0.25 * arcsec
    central_intensity = 3.7e10
    dR = sigma / 100.0
    radius = dR * np.arange(801)
    intensity = central_intensity * np.exp(-0.5 * (radius / sigma) ** 2)
    udat = np.linspace(0.0, 4.0e5, 41)
    vdat = np.zeros_like(udat)

    actual = g_double_cpu.sampleProfile(
        intensity, 0.0, dR, 128, 0.02 * arcsec, udat, vdat,
        backend=g_double_cpu.BACKEND_DFT,
    )
    rho = np.hypot(udat, vdat)
    expected = (
        2.0 * np.pi * central_intensity * sigma**2
        * np.exp(-2.0 * np.pi**2 * sigma**2 * rho**2)
    )

    peak_normalized_error = (
        np.max(np.abs(actual.real - expected)) / np.max(expected)
    )
    assert peak_normalized_error < 1e-5
    assert_allclose(actual.imag, np.zeros_like(actual.imag), atol=1e-12)


def test_chi2_profile_context_matches_uncached_fft():
    case = _make_profile_backend_case()
    ctx = g_double_cpu.create_image_context(
        case['nxy'],
        case['nxy'],
        case['u'],
        case['v'],
        case['vis_obs_re'],
        case['vis_obs_im'],
        case['weights'],
        backend=g_double_cpu.BACKEND_FFT,
    )

    expected = g_double_cpu.chi2_profile(
        case['intensity'],
        case['Rmin'],
        case['dR'],
        case['nxy'],
        case['dxy'],
        case['u'],
        case['v'],
        case['vis_obs_re'],
        case['vis_obs_im'],
        case['weights'],
        inc=case['inc'],
        dRA=case['dRA'],
        dDec=case['dDec'],
        PA=case['PA'],
        backend=g_double_cpu.BACKEND_FFT,
    )
    actual = g_double_cpu.chi2_profile(
        case['intensity'],
        case['Rmin'],
        case['dR'],
        case['nxy'],
        case['dxy'],
        ctx=ctx,
        inc=case['inc'],
        dRA=case['dRA'],
        dDec=case['dDec'],
        PA=case['PA'],
    )

    assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("backend", IMAGE_BACKENDS, ids=["fft", "dft", "nufft"])
def test_chi2_profile_context_batch_matches_scalar(backend):
    case = _make_profile_backend_case()
    ctx = g_double.create_image_context(
        case['nxy'],
        case['nxy'],
        case['u'],
        case['v'],
        case['vis_obs_re'],
        case['vis_obs_im'],
        case['weights'],
        backend=backend,
    )
    intensity_batch = np.vstack([
        case['intensity'],
        case['intensity'] * 1.03,
    ])
    inc_batch = np.array([case['inc'], case['inc'] * 0.97])
    dRA_batch = np.array([case['dRA'], case['dRA'] * 0.8])
    dDec_batch = np.array([case['dDec'], case['dDec'] * 1.2])
    PA_batch = np.array([case['PA'], case['PA'] * 0.9])

    scalar = np.array([
        g_double.chi2_profile(
            intensity_batch[idx],
            case['Rmin'],
            case['dR'],
            case['nxy'],
            case['dxy'],
            ctx=ctx,
            inc=inc_batch[idx],
            dRA=dRA_batch[idx],
            dDec=dDec_batch[idx],
            PA=PA_batch[idx],
        )
        for idx in range(len(intensity_batch))
    ])
    batch = g_double.chi2_profile(
        intensity_batch,
        case['Rmin'],
        case['dR'],
        case['nxy'],
        case['dxy'],
        ctx=ctx,
        inc_batch=inc_batch,
        dRA_batch=dRA_batch,
        dDec_batch=dDec_batch,
        PA_batch=PA_batch,
    )

    assert_allclose(batch, scalar, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("backend", IMAGE_BACKENDS, ids=["fft", "dft", "nufft"])
def test_sample_profile_backends(backend):
    case = _make_profile_backend_case()
    intensity = case['intensity']
    Rmin = case['Rmin']
    dR = case['dR']
    nxy = case['nxy']
    dxy = case['dxy']
    udat = case['u']
    vdat = case['v']
    image = case['image']
    inc = case['inc']
    dRA = case['dRA']
    dDec = case['dDec']
    PA = case['PA']

    vis = g_double.sampleProfile(intensity, Rmin, dR, nxy, dxy, udat, vdat,
                                 inc=inc, dRA=dRA, dDec=dDec, PA=PA,
                                 backend=backend, nufft_oversample=2.0)

    if backend == galario.BACKEND_FFT:
        vis_ref = py_sampleImage(image, dxy, udat, vdat, dRA=dRA, dDec=dDec, PA=PA)
        rtol = 1e-6
    elif backend == galario.BACKEND_DFT:
        vis_ref = py_sampleProfile(intensity, Rmin, dR, nxy, dxy, udat, vdat,
                                   inc=inc, dRA=dRA, dDec=dDec, PA=PA)
        rtol = 5e-7
    else:
        vis_ref = g_double.sampleProfile(intensity, Rmin, dR, nxy, dxy, udat, vdat,
                                         inc=inc, dRA=dRA, dDec=dDec, PA=PA,
                                         backend=galario.BACKEND_DFT)
        assert _relative_complex_error(vis_ref, vis) < 2e-2
        return

    assert_allclose(vis.real, vis_ref.real, rtol=rtol, atol=1e-10)
    assert_allclose(vis.imag, vis_ref.imag, rtol=rtol, atol=_complex_imag_atol(vis_ref, rtol))


@pytest.mark.parametrize("backend", IMAGE_BACKENDS, ids=["fft", "dft", "nufft"])
def test_chi2_profile_backends(backend):
    case = _make_profile_backend_case()
    intensity = case['intensity']
    Rmin = case['Rmin']
    dR = case['dR']
    nxy = case['nxy']
    dxy = case['dxy']
    udat = case['u']
    vdat = case['v']
    vis_obs_re = case['vis_obs_re']
    vis_obs_im = case['vis_obs_im']
    weights = case['weights']
    inc = case['inc']
    dRA = case['dRA']
    dDec = case['dDec']
    PA = case['PA']

    vis = g_double.sampleProfile(intensity, Rmin, dR, nxy, dxy, udat, vdat,
                                 inc=inc, dRA=dRA, dDec=dDec, PA=PA,
                                 backend=backend, nufft_oversample=2.0)
    chi2_ref = np.sum(((vis.real - vis_obs_re) ** 2. + (vis.imag - vis_obs_im) ** 2.) * weights)
    chi2 = g_double.chi2Profile(intensity, Rmin, dR, nxy, dxy, udat, vdat,
                                vis_obs_re, vis_obs_im, weights,
                                inc=inc, dRA=dRA, dDec=dDec, PA=PA,
                                backend=backend, nufft_oversample=2.0)

    assert_allclose(chi2, chi2_ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("nsamples, real_type, complex_type, rtol, atol, acc_lib, pars",
                         [(1000, 'float64', 'complex128', 1e-14, 1e-10, g_double, par1)],
                         ids=["DP_par1"])
def test_loss(nsamples, real_type, complex_type, rtol, atol, acc_lib, pars):
    # try to find out where precision is lost

    dRA = pars.get('dRA', 0.4)
    dDec = pars.get('dDec', 10.)

    # generate the samples
    maxuv_generator = 3.e3
    udat, vdat = create_sampling_points(nsamples, maxuv_generator, dtype=real_type)

    # compute the matrix size and maxuv
    size, minuv, maxuv = matrix_size(udat, vdat)

    # create model complex image (it happens to have 0 imaginary part)
    reference_image = create_reference_image(size=size, dtype=real_type)

    ###
    # shift real
    ###
    py_shift_real = np.fft.fftshift(reference_image)
    acc_shift_real = acc_lib._fftshift(reference_image)

    # interpret complex array as real and skip last two columns
    real_view = acc_shift_real.view(dtype=real_type)[:, :-2]

    # shifting the values should make no difference, so ask for high precision
    assert_allclose(py_shift_real, real_view, rtol=1e-15, atol=1e-15)

    ###
    # FFT
    ###
    py_fft = np.fft.fft2(py_shift_real)
    # use the real input!
    acc_fft = acc_lib._fft2d(py_shift_real)

    assert_allclose(unique_part(py_fft).real, acc_fft.real, rtol, atol)
    assert_allclose(unique_part(py_fft).imag, acc_fft.imag, rtol, atol)

    ###
    # shift complex
    ###
    py_shift_cmplx = np.fft.fftshift(py_fft, axes=0)
    acc_lib._fftshift_axis0(acc_fft)
    assert_allclose(unique_part(py_shift_cmplx).real, acc_fft.real, rtol, atol)
    assert_allclose(unique_part(py_shift_cmplx).imag, acc_fft.imag, rtol, atol)

    ###
    # phase
    ###
    du = maxuv/size
    uroti, vroti = uv_idx(udat, vdat, du, size/2.)
    ReInt = int_bilin_MT(py_shift_cmplx.real, uroti, vroti).astype(real_type)
    ImInt = int_bilin_MT(py_shift_cmplx.imag, uroti, vroti).astype(real_type)
    AmpInt = int_bilin_MT(np.abs(py_shift_cmplx), uroti, vroti).astype(real_type)
    PhaseInt = np.angle(ReInt + 1j*ImInt)

    vis_int = AmpInt * (np.cos(PhaseInt) + 1j*np.sin(PhaseInt))
    vis_int_acc = vis_int.copy()
    vis_int_shifted = apply_phase_array(udat, vdat, vis_int, dRA, dDec)
    vis_int_acc_shifted = acc_lib.apply_phase_vis(dRA, dDec, udat, vdat, vis_int_acc)


    # lose some absolute precision here  --> not anymore. Really? check by decreasing rtol, atol
    # atol *= 2
    assert_allclose(vis_int_shifted.real, vis_int_acc_shifted.real, rtol, atol)
    assert_allclose(vis_int_shifted.imag, vis_int_acc_shifted.imag, rtol, atol)
    # but continue with previous tolerance
    # atol /= 2

    ###
    # interpolation
    ###
    uroti, vroti = uv_idx_r2c(udat, vdat, du, size/2.)
    ReInt = int_bilin_MT(py_shift_cmplx.real, uroti, vroti).astype(real_type)
    ImInt = int_bilin_MT(py_shift_cmplx.imag, uroti, vroti).astype(real_type)
    AmpInt = int_bilin_MT(np.abs(py_shift_cmplx), uroti, vroti).astype(real_type)

    uneg = udat < 0.
    ImInt[uneg] *= -1.
    PhaseInt = np.angle(ReInt + 1j*ImInt)

    ReInt = AmpInt * np.cos(PhaseInt)
    ImInt = AmpInt * np.sin(PhaseInt)

    complexInt = acc_lib.interpolate(py_shift_cmplx.astype(complex_type, order='C'),
                                     du,
                                     udat.astype(real_type),
                                     vdat.astype(real_type))

    assert_allclose(ReInt, complexInt.real, rtol, atol)
    assert_allclose(ImInt, complexInt.imag, rtol, atol)

    ###
    # now all steps in one function
    # -> MT removed this because there is already a test for sample and here it is not clear what is the reference.
    ###
    # sampled = acc_lib.sampleImage(ref_real, dRA, dDec, du, udat, vdat)
    #
    # # a lot of precision lost. Why? --> not anymore
    # # rtol = 1
    # # atol = 0.5
    # assert_allclose(vis_int_shifted.real, sampled.real, rtol, atol)
    # assert_allclose(vis_int_shifted.imag, sampled.imag, rtol, atol)

def test_exception():
    """
    Make sure exceptions propagate from C++ to python
    """
    with pytest.raises(ValueError, match="dimension.*is less than 2"):
        g_double._fft2d(np.ones((1, 1), dtype=np.float64))

    with pytest.raises(ValueError, match="Expect a square image"):
        g_double._fft2d(np.ones((10, 12), dtype=np.float64))

    with pytest.raises(ValueError, match="dimension.*is odd"):
        g_double._fft2d(np.ones((9, 9), dtype=np.float64))


@pytest.mark.parametrize("nxy, inc, dxy, Dx, Dy, real_type, tol, acc_lib",
                         [(1000, 33.4, 1e-8, 0.23, -1.23, 'float64', 1.e-15, g_double)],
                         ids=["DP"])
def test_get_coords_meshgrid(nxy, inc, dxy, Dx, Dy, real_type, tol, acc_lib):

    ncol, nrow = nxy, nxy

    # create the referencemesh grid
    inc_cos = np.cos(inc)
    x = (np.linspace(0.5, -0.5 + 1./float(ncol), ncol, dtype=real_type)) * dxy * ncol
    y = (np.linspace(0.5, -0.5 + 1./float(nrow), nrow, dtype=real_type)) * dxy * nrow

    # we shrink the x axis, since PA is the angle East of North of the
    # the plane of the disk (orthogonal to the angular momentum axis)
    # PA=0 is a disk with vertical orbital node (aligned along North-South)
    x_m, y_m = np.meshgrid((x - Dx)/ inc_cos, y - Dy)
    R_m = np.sqrt(x_m ** 2. + y_m ** 2.)

    x_test, y_test, x_m_test, y_m_test, R_m_test = acc_lib.get_coords_meshgrid(nrow, ncol, dxy, inc, Dx=Dx, Dy=Dy, origin='upper')

    assert_allclose(x, x_test, atol=0, rtol=tol)
    assert_allclose(y, y_test, atol=0, rtol=tol)
    assert_allclose(x_m, x_m_test, atol=0, rtol=tol)
    assert_allclose(y_m, y_m_test, atol=0, rtol=tol)
    assert_allclose(R_m, R_m_test, atol=0, rtol=tol)
