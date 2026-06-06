Basic usage
===========

GALARIO accepts double-precision NumPy arrays. The ``u`` and ``v`` coordinates
must be expressed in observing wavelengths, image pixel size ``dxy`` in
radians, and visibilities in Jy.

Portable top-level API
----------------------

The top-level package selects the CUDA implementation when it was built with
CUDA and otherwise selects CPU:

.. code-block:: python

    import galario

    vis = galario.sample_image(
        image=image,
        dxy=dxy,
        u=u,
        v=v,
        backend=galario.BACKEND_AUTO,
    )

Explicit backend module
-----------------------

Use the compatibility modules when code must explicitly select hardware:

.. code-block:: python

    from galario import double as cpu

    cpu.threads(4)
    vis_cpu = cpu.sampleImage(image, dxy, u, v)

    if galario.HAVE_CUDA:
        from galario import double_cuda as cuda
        cuda.use_gpu(0)
        vis_gpu = cuda.sampleImage(image, dxy, u, v)

Image and profile sampling
--------------------------

Use ``sample_image`` for a two-dimensional model image:

.. code-block:: python

    vis = galario.sample_image(
        image=image,
        dxy=dxy,
        u=u,
        v=v,
        dRA=dRA,
        dDec=dDec,
        PA=PA,
    )

Use ``sample_profile`` for an axisymmetric radial brightness profile:

.. code-block:: python

    vis = galario.sample_profile(
        intensity,
        Rmin,
        dR,
        nxy,
        dxy,
        u,
        v,
        inc=inc,
        PA=PA,
    )

Repeated chi-squared evaluation
-------------------------------

For optimizers and MCMC, create a context once when observations remain fixed:

.. code-block:: python

    ctx = galario.create_image_context(
        image.shape[0],
        image.shape[1],
        u,
        v,
        vis_obs.real,
        vis_obs.imag,
        weights,
        backend=galario.BACKEND_AUTO,
    )

    chi2 = galario.chi2_image(ctx=ctx, image=image, dxy=dxy)

The context reuses observation arrays, work buffers, and transform plans. It is
a mutable workspace and must not be shared by simultaneous calls.

Backend selection
-----------------

The available values are ``BACKEND_AUTO``, ``BACKEND_FFT``, ``BACKEND_DFT``,
and ``BACKEND_NUFFT``. Start with ``AUTO``. Explicit selection is useful for
validation and benchmarking.
