.. image:: https://github.com/RRZE-HPC/kerncraft/blob/master/doc/logo/logo-lightbg.svg

kerncraft
=========

Loop Kernel Analysis and Performance Modeling Toolkit

This tool allows automatic analysis of loop kernels using the Execution Cache Memory (ECM) model,
the Roofline model and actual benchmarks. kerncraft provides a framework to investigate the
data reuse and cache requirements by static code analysis. In combination with the Intel IACA tool
kerncraft can give a good overview of both in-core and memory bottlenecks and use that data to
apply performance models.

For a detailed documentation see publications in `<doc/>`_.

.. image:: https://travis-ci.org/RRZE-HPC/kerncraft.svg?branch=master
    :target: https://travis-ci.org/RRZE-HPC/kerncraft?branch=master

.. image:: https://codecov.io/github/RRZE-HPC/kerncraft/coverage.svg?branch=master
    :target: https://codecov.io/github/RRZE-HPC/kerncraft?branch=master

.. image:: https://landscape.io/github/RRZE-HPC/kerncraft/master/landscape.svg?style=flat
   :target: https://landscape.io/github/RRZE-HPC/kerncraft/master
   :alt: Code Health

Installation
============

On most systems with python pip and setuputils installed, just run:

``pip install --user kerncraft``

for the latest release. In order to get the `Intel Achitecture Code Analyzer (IACA) <https://software.intel.com/en-us/articles/intel-architecture-code-analyzer>`_, required by the `ECM`, `ECMCPU` and `RooflineIACA` performance models, read `this <https://software.intel.com/protected-download/267266/157552>`_ and run:

``iaca_get --I-accept-the-Intel-What-If-Pre-Release-License-Agreement-and-please-take-my-soul``

Additional requirements are:
 * `likwid <https://github.com/RRZE-HPC/likwid>`_ (used in Benchmark model and by ``likwid_bench_auto.py``)

Usage
=====

1. Get an example kernel and machine file from the examples directory

``wget https://raw.githubusercontent.com/RRZE-HPC/kerncraft/master/examples/machine-files/SandyBridgeEP_E5-2680.yml``

``wget https://raw.githubusercontent.com/RRZE-HPC/kerncraft/master/examples/kernels/2d-5pt.c``

2. Have a look at the machine file and change it to match your targeted machine (above we downloaded a file for a Sandy Bridge EP machine)

3. Run kerncraft

``kerncraft -p ECM -m SandyBridgeEP_E5-2680.yml 2d-5pt.c -D N 10000 -D M 10000``
add `-vv` for more information on the kernel and ECM model analysis.

Citations
=========

When using Kerncraft for your work, please consider citing the following publication:

`Kerncraft: A Tool for Analytic Performance Modeling of Loop Kernels <https://dx.doi.org/10.1007/978-3-319-56702-0_1>`_ (`preprint <https://arxiv.org/abs/1702.04653>`_)

::

    J. Hammer, J. Eitzinger, G. Hager, and G. Wellein: Kerncraft: A Tool for Analytic Performance Modeling of Loop Kernels. In: Tools for High Performance Computing 2016, ISBN 978-3-319-56702-0, 1-22 (2017). Proceedings of IPTW 2016, the 10th International Parallel Tools Workshop, October 4-5, 2016, Stuttgart, Germany. Springer, Cham. DOI: 10.1007/978-3-319-56702-0_1, Preprint: arXiv:1702.04653``


Credits
=======

| Implementation: Julian Hammer;
| ECM Model (theory): Georg Hager, Holger Stengel, Jan Treibig;
| LC generalization: Julian Hammer

License
=======
AGPLv3
