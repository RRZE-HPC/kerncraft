#!/usr/bin/env python3
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter

from kerncraft import machinemodel

kernel_colors = 'bgrcmyk'

def main():
    mm = machinemodel.MachineModel(sys.argv[1])
    kernels = sorted(mm['benchmarks']['kernels'])
    cache_levels = sorted(mm['benchmarks']['measurements'])
    fig, axs = plt.subplots(len(cache_levels), 1, figsize=(7, 14), tight_layout=True)
    lines = {}
    for i, cache_level in enumerate(cache_levels):
        max_bw = 0
        max_bw_core = 0

        axs[i].set_title(cache_level)
        formatter1 = EngFormatter(places=0)  # , sep="\N{THIN SPACE}")  # U+2009
        axs[i].yaxis.set_major_formatter(formatter1)
        if cache_level == 'L1':
            axs[i].set_ylabel("Bandwidth [B/s]")
        else:
            axs[i].set_ylabel("Bandwidth [B/s]\n(incl. write-allocate)")
        axs[i].set_xlabel('cores')
        # axs[i].set_xscale('log')

        for ki, kernel in enumerate(kernels):
            if cache_level == 'L1':
                # L1 does not have write-allocate, so everything is measured correctly
                factor = 1.0
            else:
                measurement_kernel_info = mm['benchmarks']['kernels'][kernel]
                factor = (float(measurement_kernel_info['read streams']['bytes']) +
                          2.0 * float(measurement_kernel_info['write streams']['bytes']) -
                          float(measurement_kernel_info['read+write streams']['bytes'])) / \
                         (float(measurement_kernel_info['read streams']['bytes']) +
                          float(measurement_kernel_info['write streams']['bytes']))

            for SMT in mm['benchmarks']['measurements'][cache_level]:
                measurements = [
                    bw*factor
                    for bw in mm['benchmarks']['measurements'][cache_level][SMT]['results'][kernel]]
                max_bw = max(measurements+[max_bw])
                max_bw_core = max(max_bw_core, measurements[0])
                lines[kernel], = axs[i].plot(
                    range(1, 1 + len(measurements)),
                    measurements,
                    linestyle=['-', '--', '..', '-.'][SMT-1],
                    color=kernel_colors[ki])
        axs[i].set_xlim(1)
        axs[i].axhline(max_bw, color='black')
        axs[i].axhline(max_bw_core, color='black')
        axs[i].set_yticks(np.append(axs[i].get_yticks(), [float(max_bw), float(max_bw_core)]))
        axs[i].set_xticks(range(1, 1+len(measurements)))
    fig.legend(lines.values(), lines.keys(), 'lower center', ncol=10)
    fig.savefig(sys.argv[1]+'.pdf')
    #plt.show()


if __name__ == '__main__':
    main()
