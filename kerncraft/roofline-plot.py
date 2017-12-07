#!/usr/bin/env python3
from pprint import pprint

import matplotlib.pyplot as plt
from ruamel import yaml

from .prefixedunit import PrefixedUnit


def frange(start, stop, step=1.0):
    f = start
    while f < stop:
        f += step
        yield f


# Input (usually from ECM model)
result = {
    'min performance': 11175000000.0, 'bottleneck level': 2,
    'mem bottlenecks': [{'performance': PrefixedUnit(24474545454.545452, '', 'FLOP/s'),
                         'bandwidth': PrefixedUnit(89.74, u'G', u'B/s'),
                         'arithmetic intensity': 0.2727272727272727,
                         'bw kernel': 'triad', 'level': 'L1-L2'},
                        {'performance': PrefixedUnit(12957000000.0, '',
                         'FLOP/s'), 'bandwidth': PrefixedUnit(43.19, u'G', u'B/s'),
                         'arithmetic intensity': 0.3, 'bw kernel': 'triad', 'level': 'L2-L3'},
                        {'performance': PrefixedUnit(11175000000.0, '', 'FLOP/s'),
                         'bandwidth': PrefixedUnit(22.35, u'G', u'B/s'),
                         'arithmetic intensity': 0.5, 'bw kernel': 'triad', 'level': 'L3-MEM'}]}
machine = yaml.load(open('machine-files/emmy.yaml'))
max_flops = machine['clock']*sum(machine['FLOPs per cycle']['DP'].values())
max_flops.unit = "FLOP/s"

pprint(result)
pprint(max_flops)

# Plot configuration
height = 0.8

fig = plt.figure(frameon=False)
ax = fig.add_subplot(1, 1, 1)

yticks_labels = []
yticks = []
xticks_labels = []
xticks = [2.**i for i in range(-4, 4)]

ax.set_xlabel('arithmetic intensity [FLOP/byte]')
ax.set_ylabel('performance [FLOP/s]')

# Upper bound
x = list(frange(min(xticks), max(xticks), 0.01))
bw = float(result['mem bottlenecks'][result['bottleneck level']]['bandwidth'])
ax.plot(x, [min(bw*x, float(max_flops)) for x in x])

# Code location
perf = min(
    float(max_flops),
    float(result['mem bottlenecks'][result['bottleneck level']]['performance']))
arith_intensity = result['mem bottlenecks'][result['bottleneck level']]['arithmetic intensity']
ax.plot(arith_intensity, perf, 'r+', markersize=12, markeredgewidth=4)

# ax.tick_params(axis='y', which='both', left='off', right='off')
# ax.tick_params(axis='x', which='both', top='off')
ax.set_xscale('log', basex=2)
ax.set_yscale('log')
ax.set_xlim(min(xticks), max(xticks))
# ax.set_yticks([perf, float(max_flops)])
ax.set_xticks(xticks+[arith_intensity])
ax.grid(axis='x', alpha=0.7, linestyle='--')
# fig.savefig('out.pdf')
plt.show()
