#!/usr/bin/env python3
"""Roofline model and helper functions."""
import sys
from pprint import pformat  # Do not use pprint, breaks in combination with --store and StringIO

from kerncraft.prefixedunit import PrefixedUnit
from kerncraft.cacheprediction import LayerConditionPredictor, CacheSimulationPredictor
from .base import PerformanceModel


class RooflineFLOP(PerformanceModel):
    """
    Representation of the RooflineFLOP model based on simplistic FLOP analysis.

    more info to follow...
    """

    name = "RooflineFLOP"

    @classmethod
    def configure_arggroup(cls, parser):
        """Configure argument parser."""
        pass

    def __init__(self, kernel, machine, args=None, parser=None, cores=1,
                 cache_predictor=LayerConditionPredictor, verbose=0):
        """
        Create roofline model from kernel and machine objects.

        *kernel* is a Kernel object
        *machine* describes the machine (cpu, cache and memory) characteristics
        *args* (optional) are the parsed arguments from the comand line


        If *args* is None, *asm_block*, *pointer_increment* and *verbose* will be used, otherwise
        *args* takes precedence.
        """
        self.kernel = kernel
        self.machine = machine
        self._args = args
        self._parser = parser
        self.results = None

        if args:
            self.verbose = self._args.verbose
            self.cores = self._args.cores
            if self._args.cache_predictor == 'SIM':
                self.predictor = CacheSimulationPredictor(self.kernel, self.machine, self.cores)
            elif self._args.cache_predictor == 'LC':
                self.predictor = LayerConditionPredictor(self.kernel, self.machine, self.cores)
            else:
                raise NotImplementedError("Unknown cache predictor, only LC (layer condition) and "
                                          "SIM (cache simulation with pycachesim) is supported.")
        else:
            self.cores = cores
            self.predictor = cache_predictor(self.kernel, self.machine, self.cores)
            self.verbose = verbose

    def calculate_cache_access(self):
        """Apply cache prediction to generate cache access behaviour."""
        self.results = {'loads': self.predictor.get_loads(),
                        'stores': self.predictor.get_stores(),
                        'verbose infos': self.predictor.get_infos(),  # only for verbose outputs
                        'bottleneck level': 0,
                        'mem bottlenecks': []}

        element_size = self.kernel.datatypes_size[self.kernel.datatype]
        cacheline_size = float(self.machine['cacheline size'])
        elements_per_cacheline = int(cacheline_size // element_size)

        total_flops = sum(self.kernel._flops.values())*elements_per_cacheline

        # TODO let user choose threads_per_core:
        threads_per_core = 1

        # Compile relevant information

        # CPU-L1 stats (in bytes!)
        # We compile CPU-L1 stats on our own, because cacheprediction only works on cache lines
        read_offsets, write_offsets = zip(*list(self.kernel.compile_global_offsets(
            iteration=range(0, elements_per_cacheline))))
        read_offsets = set([item for sublist in read_offsets if sublist is not None
                            for item in sublist])
        write_offsets = set([item for sublist in write_offsets if sublist is not None
                             for item in sublist])

        write_streams = len(write_offsets)
        read_streams = len(read_offsets) + write_streams  # write-allocate
        total_loads = read_streams * element_size
        total_evicts = write_streams * element_size
        bw, measurement_kernel = self.machine.get_bandwidth(
            0,
            read_streams,
            0,  # we do not consider stores to L1 
            threads_per_core,
            cores=self.cores)

        # Calculate performance (arithmetic intensity * bandwidth with
        # arithmetic intensity = Iterations / bytes loaded
        if total_loads == 0:
            # This happens in case of full-caching
            arith_intens = None
            it_s = None
        else:
            arith_intens = 1.0/(total_loads/elements_per_cacheline)
            it_s = PrefixedUnit(float(bw)*arith_intens, 'It/s')

        self.results['mem bottlenecks'].append({
            'performance': self.conv_perf(it_s),
            'level': self.machine['memory hierarchy'][0]['level'],
            'arithmetic intensity': arith_intens,
            'bw kernel': measurement_kernel,
            'bandwidth': bw,
            'bytes transfered': total_loads})
        self.results['bottleneck level'] = len(self.results['mem bottlenecks'])-1
        self.results['min performance'] = self.conv_perf(it_s)

        # for other cache and memory levels:
        for cache_level, cache_info in list(enumerate(self.machine['memory hierarchy']))[:-1]:
            # Compiling stats (in bytes!)
            total_loads = self.results['loads'][cache_level+1]*cacheline_size
            total_stores = self.results['stores'][cache_level+1]*cacheline_size

            # choose bw according to cache level and problem
            # first, compile stream counts at current cache level
            # write-allocate is allready resolved above
            read_streams = self.results['loads'][cache_level+1]
            write_streams = self.results['stores'][cache_level+1]
            # second, try to find best fitting kernel (closest to stream seen stream counts):
            bw, measurement_kernel = self.machine.get_bandwidth(
                cache_level+1, read_streams, write_streams, threads_per_core,
                cores=self.cores)

            # Calculate performance (arithmetic intensity * bandwidth with
            # arithmetic intensity = flops / bytes transfered)
            bytes_transfered = total_loads + total_stores

            if bytes_transfered == 0:
                # This happens in case of full-caching
                arith_intens = float('inf')
                it_s = PrefixedUnit(float('inf'), 'It/s')
            else:
                arith_intens = 1/(bytes_transfered/elements_per_cacheline)
                it_s = PrefixedUnit(float(bw)*arith_intens, 'It/s')

            self.results['mem bottlenecks'].append({
                'performance': self.conv_perf(it_s),
                'level': (self.machine['memory hierarchy'][cache_level + 1]['level']),
                'arithmetic intensity': arith_intens,
                'bw kernel': measurement_kernel,
                'bandwidth': bw,
                'bytes transfered': bytes_transfered})
            if it_s < self.results.get('min performance', {'It/s': it_s})['It/s']:
                self.results['bottleneck level'] = len(self.results['mem bottlenecks'])-1
                self.results['min performance'] = self.conv_perf(it_s)

        return self.results

    def analyze(self):
        """Run analysis."""
        precision = 'DP' if self.kernel.datatype == 'double' else 'SP'
        self.calculate_cache_access()

        self.results['max_perf'] = self.conv_perf(self.machine['clock'] * self.cores * \
            self.machine['FLOPs per cycle'][precision]['total'])

    def conv_perf(self, it_s):
        """Convert performance (It/s) to other units, such as FLOP/s or cy/CL."""
        clock = self.machine['clock']
        flops_per_it = sum(self.kernel._flops.values())
        performance = it_s*flops_per_it
        performance.unit = 'FLOP/s'
        element_size = self.kernel.datatypes_size[self.kernel.datatype]
        elements_per_cacheline = int(float(self.machine['cacheline size'])) / element_size
        cy_cl = clock/it_s*elements_per_cacheline
        cy_cl.unit = 'cy/CL'
        cy_it = clock/it_s
        cy_it.unit = 'cy/It'

        return {'It/s': it_s,
                'cy/CL': cy_cl,
                'cy/It': cy_it,
                'FLOP/s': performance}

    def report(self, output_file=sys.stdout):
        """Report analysis outcome in human readable form."""
        max_perf = self.results['max_perf']

        if self._args and self._args.verbose >= 3:
            print('{}'.format(pformat(self.results)), file=output_file)

        if self._args and self._args.verbose >= 1:
            print('{}'.format(pformat(self.results['verbose infos'])), file=output_file)
            print('Bottlenecks:', file=output_file)
            print('  level | a. intensity |   performance   |   peak bandwidth  | peak bandwidth kernel',
                  file=output_file)
            print('--------+--------------+-----------------+-------------------+----------------------',
                  file=output_file)
            print('    CPU |              | {!s:>15} |                   |'.format(
                max_perf[self._args.unit]),
                  file=output_file)
            for b in self.results['mem bottlenecks']:
                print('{level:>7} | {arithmetic intensity:>7.2} It/B | {0!s:>15} |'
                      ' {bandwidth!s:>17} | {bw kernel:<8}'.format(
                          b['performance'][self._args.unit], **b),
                      file=output_file)
            print('', file=output_file)

        if self.results['min performance']['FLOP/s'] > max_perf['FLOP/s']:
            # CPU bound
            print('CPU bound. {!s} due to CPU max. FLOP/s'.format(max_perf), file=output_file)
        else:
            # Cache or mem bound
            print('Cache or mem bound.', file=output_file)

            bottleneck = self.results['mem bottlenecks'][self.results['bottleneck level']]
            print('{!s} due to {} transfer bottleneck (with bw from {} benchmark)'.format(
                    bottleneck['performance'][self._args.unit],
                    bottleneck['level'],
                    bottleneck['bw kernel']),
                  file=output_file)
            print('Arithmetic Intensity: {:.2f} It/B'.format(bottleneck['arithmetic intensity']),
                  file=output_file)

        if any(['_Complex' in var_info[0] for var_info in self.kernel.variables.values()]):
            print("WARNING: FLOP counts are probably wrong, because complex flops are counted\n"
                  "         as single flops. All other units should not be affected.\n",
                  file=sys.stderr)


class RooflineASM(RooflineFLOP):
    """
    Representation of the Roofline model based on ASM throughput analysis.

    more info to follow...
    """

    name = "Roofline (with instruction execution model)"

    @classmethod
    def configure_arggroup(cls, parser):
        """Configure argument parser."""
        pass

    def __init__(self, kernel, machine, args=None, parser=None, asm_block='auto',
                 pointer_increment='auto', cores=1, predictor=LayerConditionPredictor, verbose=0):
        """
        Create Roofline model with ASM analysis from kernel and machine objects.

        *kernel* is a Kernel object
        *machine* describes the machine (cpu, cache and memory) characteristics
        *args* (optional) are the parsed arguments from the comand line
        if *args* is given also *parser* has to be provided

        If *args* is None, *asm_block*, *pointer_increment* and *verbose* will be used, otherwise
        *args* takes precedence.
        """
        RooflineFLOP.__init__(self, kernel, machine, args, parser, cores, predictor, verbose)
        self.results = None

        if args:
            # handle CLI info
            self.asm_block = self._args.asm_block
            self.pointer_increment = self._args.pointer_increment
        else:
            self.asm_block = asm_block
            self.pointer_increment = pointer_increment

        # Validate arguments
        if self.asm_block not in ['auto', 'manual']:
            try:
                self.asm_block = int(args.asm_block)
            except ValueError:
                parser.error('asm_block can only be "auto", "manual" or an integer')
        if self.pointer_increment not in ['auto', 'auto_with_manual_fallback', 'manual']:
            try:
                self.pointer_increment = int(args.pointer_increment)
            except ValueError:
                parser.error('pointer_increment can only be "auto", '
                             '"auto_with_manual_fallback", "manual" or an integer')

    def analyze(self):
        """Run complete analysis."""
        self.results = self.calculate_cache_access()
        try:
            incore_analysis, pointer_increment = self.kernel.incore_analysis(
                asm_block=self.asm_block,
                pointer_increment=self.pointer_increment,
                model=self._args.incore_model,
                verbose=self.verbose > 2)
        except RuntimeError as e:
            print("In-core analysis failed: " + str(e))
            sys.exit(1)

        block_throughput = incore_analysis['throughput']
        uops = incore_analysis['uops']
        incore_output = incore_analysis['output']
        port_cycles = incore_analysis['port cycles']

        # Normalize to cycles per cacheline
        elements_per_block = abs(
            pointer_increment / self.kernel.datatypes_size[self.kernel.datatype])
        block_size = elements_per_block*self.kernel.datatypes_size[self.kernel.datatype]
        try:
            block_to_cl_ratio = float(self.machine['cacheline size'])/block_size
        except ZeroDivisionError as e:
            print("Too small block_size / pointer_increment:", e, file=sys.stderr)
            sys.exit(1)

        port_cycles = dict([(i[0], i[1]*block_to_cl_ratio) for i in list(port_cycles.items())])
        if uops is not None:
            uops = uops*block_to_cl_ratio
        cl_throughput = block_throughput*block_to_cl_ratio
        flops_per_element = sum(self.kernel._flops.values())

        # Overwrite CPU-L1 stats, because they are covered by In-Core Model
        self.results['mem bottlenecks'][0] = None

        # Reevaluate mem bottleneck
        self.results['min performance'] = self.conv_perf(PrefixedUnit(float('inf'), 'It/s'))
        self.results['bottleneck level'] = None
        for level, bottleneck in enumerate(self.results['mem bottlenecks']):
            if level == 0:
                # ignoring CPU-L1
                continue
            if bottleneck['performance']['It/s'] < self.results['min performance']['It/s']:
                self.results['bottleneck level'] = level
                self.results['min performance'] = bottleneck['performance']

        # Create result dictionary
        self.results.update({
            'cpu bottleneck': {
                'port cycles': port_cycles,
                'cl throughput': cl_throughput,
                'uops': uops,
                'performance throughput': self.conv_perf(PrefixedUnit(
                    self.machine['clock']/block_throughput*elements_per_block
                    * self.cores, "It/s")),
                'in-core model output': incore_output}})

    def report(self, output_file=sys.stdout):
        """Print human readable report of model."""
        cpu_perf = self.results['cpu bottleneck']['performance throughput']

        if self.verbose >= 3:
            print('{}'.format(pformat(self.results)), file=output_file)

        if self.verbose >= 1:
            print('Bottlenecks:', file=output_file)
            print('  level | a. intensity |   performance   |   peak bandwidth  | peak bandwidth kernel',
                  file=output_file)
            print('--------+--------------+-----------------+-------------------+----------------------',
                  file=output_file)
            print('    CPU |              | {!s:>15} |                   |'.format(
                cpu_perf[self._args.unit]),
                  file=output_file)
            for b in self.results['mem bottlenecks']:
                # Skip CPU-L1 from Roofline model
                if b is None:
                    continue
                print('{level:>7} | {arithmetic intensity:>7.2} It/B | {0!s:>15} |'
                      ' {bandwidth!s:>17} | {bw kernel:<8}'.format(
                          b['performance'][self._args.unit], **b),
                      file=output_file)
            print('', file=output_file)
            print('In-Core Model analisys:', file=output_file)
            print('{!s}'.format(
                {k: v
                 for k, v in list(self.results['cpu bottleneck'].items())
                 if k not in['in-core model output']}),
                file=output_file)

        if self.results['min performance']['It/s'] > cpu_perf['It/s']:
            # CPU bound
            print('CPU bound. {!s} due to CPU bottleneck'.format(cpu_perf[self._args.unit]),
                  file=output_file)
        else:
            # Cache or mem bound
            print('Cache or mem bound.', file=output_file)

            bottleneck = self.results['mem bottlenecks'][self.results['bottleneck level']]
            print('{!s} due to {} transfer bottleneck (with bw from {} benchmark)'.format(
                      bottleneck['performance'][self._args.unit],
                      bottleneck['level'],
                      bottleneck['bw kernel']),
                  file=output_file)
            print('Arithmetic Intensity: {:.2f} It/B'.format(bottleneck['arithmetic intensity']),
                  file=output_file)

        if any(['_Complex' in var_info[0] for var_info in self.kernel.variables.values()]) and \
                self._args.unit == 'FLOP/s':
            print("WARNING: FLOP counts are probably wrong, because complex flops are counted\n"
                  "         as single flops. All other units should not be affected.\n",
                  file=sys.stderr)
