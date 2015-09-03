kerncraft
=========

Loop Kernel Analysis and Performance Modeling Toolkit

This tool allows automatic analysis of loop kernels using the Execution Cache Memory (ECM) model, 
the Roofline model and actual benchmarks. kerncraft provides a framework to investigate the
data reuse and cache requirements by static code analysis. In combination with the Intel IACA tool
kerncraft can give a good overview of both in-core and memory bottlenecks and use that data to 
apply performance models.

For a detailed documentation see this master's thesis: `pdf <doc/masterthesis-2015.pdf>`_

Installation
============

Run:
``pip install kerncraft``

Additional requirements are:
 * Intel IACA tool, with (working) ``iaca.sh`` in PATH environment variable (used by ECM, ECMCPU and Roofline models)
 * likwid (used in Benchmark model and by ``likwid_bench_auto.py``)

Usage
=====

1. Get an example kernel and machine file from the examples directory

``wget https://raw.githubusercontent.com/cod3monk/kerncraft/master/examples/machine-files/phinally.yaml``

``wget https://raw.githubusercontent.com/cod3monk/kerncraft/master/examples/kernels/2d-5pt.c``

2. Have a look at the machine file and change it to match your targeted machine (above we downloaded a file for a sandy bridge EP machine)

3. Run kerncraft

``kerncraft -p ECM -m phinally.yaml 2d-5pt.c -D N 10000 -D M 10000``
add `-vv` for more information on the kernel and ECM model analysis.

Credits
=======
Implementation: Julian Hammer
ECM Model (theory): Georg Hager, Holger Stengel, Jan Treibig

License
=======
AGPLv3
