Python API reference
====================

Recommended API
---------------

New code should import ``galario`` and use:

* ``create_image_context``
* ``sample_image``
* ``chi2_image``
* ``sample_profile``
* ``chi2_profile``

The public backend constants are ``BACKEND_AUTO``, ``BACKEND_FFT``,
``BACKEND_DFT``, and ``BACKEND_NUFFT``.

Compatibility API
-----------------

The double-precision CPU module is ``galario.double``. CUDA builds additionally
provide ``galario.double_cuda``. Their camelCase functions remain available:

* ``sampleImage`` and ``chi2Image``
* ``sampleProfile`` and ``chi2Profile``
* ``sampleImageComponents`` and ``chi2ImageComponents``

Runtime information
-------------------

``galario.HAVE_CUDA`` reports whether this package build contains the CUDA
module. ``galario.HAVE_NANOBIND`` reports whether the native nanobind extension
is available.

CPU execution is configured with ``galario.double.threads(count)``. CUDA device
selection uses ``galario.double_cuda.ngpus()`` and
``galario.double_cuda.use_gpu(device_id)``.

Array requirements
------------------

Inputs are converted to contiguous ``numpy.float64`` or ``numpy.complex128``
arrays by the Python policy layer. Observation arrays must have matching
one-dimensional lengths. Images must be square and have an even side length.

Exceptions
----------

===============================================  ======================
Event                                            Python exception
===============================================  ======================
Invalid dimensions, shapes, or backend name      ``ValueError``
Allocation failure                               ``MemoryError``
Backend or transform failure                     ``RuntimeError``
===============================================  ======================

For the complete callable surface, use Python introspection:

.. code-block:: python

    import galario
    help(galario.sample_image)
    help(galario.chi2_image)
