from __future__ import annotations

# Focused smoke tests for direct nanobind CPU/CUDA module behavior.
import numpy as np
import pytest
from os import environ

import galario
from galario import arcsec, deg

if not (galario.HAVE_CUDA and int(environ.get("GALARIO_TEST_GPU", 0))):
    pytest.skip("nanobind smoke test requires CUDA test mode", allow_module_level=True)

from galario import double as g_double_cpu
from galario import double_cuda as g_double
from galario import nanobind_double as g_nb_cpu
from galario import nanobind_double_cuda as g_nb
from utils import create_sampling_points, generate_random_vis, matrix_size


def _make_case(nxy=96, nsamples=64):
    udat, vdat = create_sampling_points(nsamples, 1500.0, dtype="float64")
    _, _, maxuv = matrix_size(udat, vdat)
    dxy = 1.0 / maxuv
    x_obs, _, weights = generate_random_vis(nsamples, "float64")
    params_batch = np.array(
        [
            [
                9.395, 0.451, 0.392,
                9.905, 0.363, 0.031,
                10.318, 0.473, 0.128,
                10.291, 0.450, 0.049,
                20.429, 26.992,
                59.52, 117.18, 0.01372, 0.01487,
            ],
            [
                9.401, 0.447, 0.401,
                9.887, 0.369, 0.033,
                10.301, 0.478, 0.126,
                10.274, 0.446, 0.051,
                20.112, 27.205,
                58.90, 116.40, 0.01290, 0.01510,
            ],
        ],
        dtype=np.float64,
    )
    ctx_compat = g_double.create_image_context(
        nxy, nxy, udat, vdat, x_obs.real.copy(), x_obs.imag.copy(), weights,
        backend=galario.BACKEND_AUTO, nufft_oversample=2.0,
    )
    ctx_nb = g_nb.create_image_context(
        nxy, nxy, udat, vdat, x_obs.real.copy(), x_obs.imag.copy(), weights,
        backend=galario.BACKEND_AUTO, nufft_oversample=2.0,
    )
    return dxy, params_batch, ctx_compat, ctx_nb


def _make_image(nxy):
    axis = np.linspace(-1.0, 1.0, nxy, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    return np.exp(-4.0 * (xx ** 2 + 0.7 * yy ** 2))


def test_nanobind_cached_components_batch_matches_double_cuda():
    dxy, _, ctx_compat, ctx_nb = _make_case()
    gauss_batch = np.empty((2, 0), dtype=np.float64)
    ring_batch = np.array(
        [
            [0.2476394121246608, 2.1869092010030152e-06, 1.899669629949381e-06,
             0.8011905157849979, 1.7592730022273154e-06, 1.5029224118365612e-07,
             2.081559480312316, 2.2935602108471133e-06, 6.205615118201921e-07],
            [0.2769508484660254, 2.167515494029088e-06, 1.9433031859577785e-06,
             0.7671444228991417, 1.7883612536885054e-06, 1.599890506148916e-07,
             2.000475654451667, 2.317801807064217e-06, 6.108647023889566e-07],
        ],
        dtype=np.float64,
    )
    arc_batch = np.array(
        [
            [1.954916036714793, 2.181586781168863e-06, 2.3767270374367267e-07, 1.9273843811168246, 0.47109933669298774],
            [1.8791635431120732, 2.162193074194936e-06, 2.473695131749082e-07, 1.9466030420897752, 0.44754436985833834],
        ],
        dtype=np.float64,
    )
    inc_batch = np.array([1.038817442169167, 1.0492056165908587], dtype=np.float64)
    pa_batch = np.array([2.04517700598558, 1.9997772449112885], dtype=np.float64)
    dra_batch = np.array([6.650042905637843e-08, 6.253040331299572e-08], dtype=np.float64)
    ddec_batch = np.array([7.209878791947352e-08, 8.006965459061561e-08], dtype=np.float64)

    expected = g_double.chi2_image(
        ctx=ctx_compat,
        dxy=dxy,
        gauss_params_batch=gauss_batch,
        ring_params_batch=ring_batch,
        arc_params_batch=arc_batch,
        inc_batch=inc_batch,
        dRA_batch=dra_batch,
        dDec_batch=ddec_batch,
        PA_batch=pa_batch,
        origin="lower",
    )
    actual = np.asarray(
        g_nb.chi2_image(
            ctx=ctx_nb,
            dxy=dxy,
            gauss_params_batch=gauss_batch,
            ring_params_batch=ring_batch,
            arc_params_batch=arc_batch,
            inc_batch=inc_batch,
            dRA_batch=dra_batch,
            dDec_batch=ddec_batch,
            PA_batch=pa_batch,
            origin="lower",
        )
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_nanobind_direct_image_matches_double_cuda():
    nxy = 96
    dxy, _, _, _ = _make_case(nxy=nxy)
    udat, vdat = create_sampling_points(64, 1500.0, dtype="float64")
    x_obs, _, weights = generate_random_vis(64, "float64")
    image = _make_image(nxy)

    expected = g_double.chi2_image(
        image=image,
        dxy=dxy,
        u=udat,
        v=vdat,
        vis_obs_re=x_obs.real.copy(),
        vis_obs_im=x_obs.imag.copy(),
        weights=weights,
        dRA=0.03 * arcsec,
        dDec=-0.02 * arcsec,
        PA=11.0 * deg,
        origin="lower",
        backend=galario.BACKEND_AUTO,
        nufft_oversample=2.0,
    )
    actual = g_nb.chi2_image(
        image=image,
        dxy=dxy,
        u=udat,
        v=vdat,
        vis_obs_re=x_obs.real.copy(),
        vis_obs_im=x_obs.imag.copy(),
        weights=weights,
        dRA=0.03 * arcsec,
        dDec=-0.02 * arcsec,
        PA=11.0 * deg,
        origin="lower",
        backend=galario.BACKEND_AUTO,
        nufft_oversample=2.0,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_nanobind_sample_image_matches_double_cuda():
    nxy = 96
    dxy, _, _, _ = _make_case(nxy=nxy)
    udat, vdat = create_sampling_points(64, 1500.0, dtype="float64")
    image = _make_image(nxy)

    expected = g_double.sample_image(
        image=image,
        dxy=dxy,
        u=udat,
        v=vdat,
        dRA=0.02 * arcsec,
        dDec=-0.01 * arcsec,
        PA=9.0 * deg,
        origin="lower",
        backend=galario.BACKEND_AUTO,
        nufft_oversample=2.0,
    )
    actual = np.asarray(
        g_nb.sample_image(
            image=image,
            dxy=dxy,
            u=udat,
            v=vdat,
            dRA=0.02 * arcsec,
            dDec=-0.01 * arcsec,
            PA=9.0 * deg,
            origin="lower",
            backend=galario.BACKEND_AUTO,
            nufft_oversample=2.0,
        )
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_nanobind_cached_components_matches_double_cuda():
    dxy, _, ctx_compat, ctx_nb = _make_case()
    gauss = np.empty((0, 2), dtype=np.float64)
    ring = np.array(
        [
            [0.2476394121246608, 2.1869092010030152e-06, 1.899669629949381e-06],
            [0.8011905157849979, 1.7592730022273154e-06, 1.5029224118365612e-07],
            [2.081559480312316, 2.2935602108471133e-06, 6.205615118201921e-07],
        ],
        dtype=np.float64,
    )
    arc = np.array(
        [[1.954916036714793, 2.181586781168863e-06, 2.3767270374367267e-07, 1.9273843811168246, 0.47109933669298774]],
        dtype=np.float64,
    )
    inc = 1.038817442169167
    pa = 2.04517700598558
    dra = 6.650042905637843e-08
    ddec = 7.209878791947352e-08

    expected = g_double.chi2_image(
        ctx=ctx_compat,
        dxy=dxy,
        gauss_params=gauss,
        ring_params=ring,
        arc_params=arc,
        inc=inc,
        dRA=dra,
        dDec=ddec,
        PA=pa,
        origin="lower",
    )
    actual = g_nb.chi2_image(
        ctx=ctx_nb,
        dxy=dxy,
        gauss_params=gauss,
        ring_params=ring,
        arc_params=arc,
        inc=inc,
        dRA=dra,
        dDec=ddec,
        PA=pa,
        origin="lower",
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_nanobind_profile_matches_double_cuda():
    udat, vdat = create_sampling_points(64, 1500.0, dtype="float64")
    x_obs, _, weights = generate_random_vis(64, "float64")
    intensity = np.exp(-np.linspace(0.0, 3.0, 80, dtype=np.float64))
    Rmin = 0.0
    dR = 0.002 * arcsec
    nxy = 128
    dxy = 0.003 * arcsec

    expected = g_double.chi2_profile(
        intensity, Rmin, dR, nxy, dxy,
        udat, vdat,
        x_obs.real.copy(), x_obs.imag.copy(), weights,
        dRA=0.01 * arcsec, dDec=-0.015 * arcsec, PA=7.0 * deg, inc=35.0 * deg,
        backend=galario.BACKEND_AUTO, nufft_oversample=2.0,
    )
    actual = g_nb.chi2_profile(
        intensity, Rmin, dR, nxy, dxy,
        udat, vdat,
        x_obs.real.copy(), x_obs.imag.copy(), weights,
        dRA=0.01 * arcsec, dDec=-0.015 * arcsec, PA=7.0 * deg, inc=35.0 * deg,
        backend=galario.BACKEND_AUTO, nufft_oversample=2.0,
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)


def test_nanobind_sample_profile_matches_double_cuda():
    udat, vdat = create_sampling_points(64, 1500.0, dtype="float64")
    intensity = np.exp(-np.linspace(0.0, 3.0, 80, dtype=np.float64))
    Rmin = 0.0
    dR = 0.002 * arcsec
    nxy = 128
    dxy = 0.003 * arcsec

    expected = g_double.sample_profile(
        intensity, Rmin, dR, nxy, dxy,
        udat, vdat,
        dRA=0.01 * arcsec, dDec=-0.015 * arcsec, PA=7.0 * deg, inc=35.0 * deg,
        backend=galario.BACKEND_AUTO, nufft_oversample=2.0,
    )
    actual = np.asarray(
        g_nb.sample_profile(
            intensity, Rmin, dR, nxy, dxy,
            udat, vdat,
            dRA=0.01 * arcsec, dDec=-0.015 * arcsec, PA=7.0 * deg, inc=35.0 * deg,
            backend=galario.BACKEND_AUTO, nufft_oversample=2.0,
        )
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
