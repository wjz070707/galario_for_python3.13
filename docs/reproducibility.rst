Reproducible research
=====================

Do not maintain a permanent second source tree only for a paper. A copied tree
quickly loses its relationship to tests, fixes, and dependency metadata.
Instead, freeze each scientific analysis with an immutable release:

#. Commit the tested source and record ``git rev-parse HEAD``.
#. Create an annotated tag, for example ``v1.3.1-paper-name``.
#. Publish that tag as a GitHub Release.
#. Connect the repository to Zenodo and archive the release to obtain a DOI.
#. Store the environment lock file, run configuration, random seed, input-data
   checksums, and benchmark output with the analysis.
#. Cite both the original GALARIO methods paper and the DOI for the exact
   maintained release.

The tagged release preserves the Context acceleration exactly as tested while
the main branch remains free to receive later optimizations. If a paper needs
a correction, branch from the tag, make and test the correction, and publish a
new patch tag rather than silently changing the old tag.

Performance provenance
----------------------

For optimizer or MCMC results, record:

* GALARIO version, Git commit, and release DOI.
* CPU or CUDA module and explicit or automatic Fourier backend.
* CPU model and thread count, or GPU model, driver, CUDA, and cuFFT versions.
* Image dimensions, visibility count, walker batch size, and MCMC steps.
* Whether one reusable Context was created outside the likelihood loop.
* The benchmark command and wall-clock result on the reported hardware.

Context reuse changes throughput, not the likelihood definition. Numerical
equivalence tests between scalar and batched Context paths should remain part
of every archived release.
