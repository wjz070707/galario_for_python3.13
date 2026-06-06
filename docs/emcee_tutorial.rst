Fitting a Gaussian profile with emcee 3
=======================================

The executable example ``examples/emcee_gaussian_profile.py`` fits the
documented ``docs/uvtable.txt`` data with a six-parameter Gaussian radial
profile. It is the maintained emcee 3 version of the historical quickstart.

Install the optional tutorial dependencies with::

    python -m pip install ".[demo]"

Open ``examples/emcee_gaussian_profile.py`` in VSCode and edit the
``USER CONFIGURATION`` block near the top of the file. Then run the file
directly. No command-line arguments are required::

    PYTHONPATH=build/python python examples/emcee_gaussian_profile.py

The default configuration uses all rows in the uv table, an image of at least
128 pixels per side, 1280 radial cells, 24 walkers, and a 500-step burn-in.
Set ``UV_SAMPLES`` to a positive integer only when a faster, subsampled
experiment is desired. Values larger than the table automatically use all
available rows. The script always writes a corner plot to ``OUTPUT``; by
default this is
``triangle_example.png`` in the repository root.

The default is ``USE_GPU = True``. Select the CUDA device with ``GPU_DEVICE``.
Set ``USE_GPU = False`` to use the CPU backend and configure ``CPU_THREADS``.
The profile likelihood uses an FFT image context. The radial profile is swept
onto an image for each model, while fixed observations, the transform plan,
and backend work buffers are reused across MCMC evaluations.

Profile functions demonstrated
------------------------------

The MCMC example uses:

* ``get_image_size`` derives ``nxy`` and ``dxy`` from the uv coverage.
* ``create_image_context`` caches observations and FFT workspaces.
* ``sample_profile`` evaluates initial model visibilities.
* ``reduce_chi2`` computes chi-squared from those sampled visibilities.
* ``chi2_profile(..., ctx=context)`` evaluates the repeated likelihood.

The script checks both profile chi-squared paths before starting emcee.

Image API example
-----------------

Run ``examples/chi2_image_gaussian.py`` for the equivalent complete image
workflow. It uses ``get_image_size``, ``sample_image``,
``create_image_context``, and contextual ``chi2_image`` before fitting the
Gaussian with emcee and writing ``triangle_image_example.png``. Its
configuration block is independent from the profile example.

Model and sampled parameters
----------------------------

The radial brightness model is

.. math::

    I(R) = f_0 \exp\left[-\frac{1}{2}\left(\frac{R}{\sigma}\right)^2\right].

The sampler explores ``log10(f0)``, ``sigma``, inclination, position angle, and
the right-ascension and declination offsets. The prior is uniform in those
sampled coordinates. This means it is uniform in ``log10(f0)``, not in the
linear brightness ``f0``.

emcee 3 differences
-------------------

Modern emcee no longer accepts the old ``threads=`` argument. CPU parallelism
inside GALARIO is selected with::

    from galario import double as g
    g.threads(4)

The modern sampling and chain APIs are::

    sampler.run_mcmc(initial_positions, nsteps, progress=True)
    samples = sampler.get_chain(discard=burn_in, flat=True)

The deprecated three-value return from ``run_mcmc`` and ``sampler.chain`` are
not used.

CPU and process parallelism
---------------------------

GALARIO uses OpenMP threads inside each likelihood evaluation. Start with one
emcee process and a modest GALARIO thread count. If an external multiprocessing
pool is added later, reduce GALARIO to one thread per worker to avoid CPU
oversubscription.
