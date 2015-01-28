#!/usr/bin/env python

class MachineModel:
    def __init__(
            self, name, arch, clock, cores, cl_size, mem_bw, cache_stack, port_match, icc_flags):
        '''
        *name* is the official name of the CPU
        *arch* is the archetecture name, this must be SNB, IVB or HSW
        *clock* is the number of cycles per second the CPU can perform
        *cores* is the number of cores
        *cl_size* is the number of bytes in one cache line
        *mem_bw* is the number of bytes per second that are read from memory to the lowest cache lvl
        *cache_stack* is a list of cache levels (tuple):
            (level, size, type, bw)
            *level* is the numerical id of the cache level
            *size* is the size of the cache
            *type* can be 'per core' or 'per socket'
            *cycles* is is the numbe of cycles to transfer one cache line from/to lower level
        *port_match* is a dict matching port names to LOAD and STORE
        '''
        self.name = name
        self.arch = arch
        self.clock = clock
        self.cores = cores
        self.cl_size = cl_size
        self.mem_bw = mem_bw
        self.cache_stack = cache_stack
        self.port_match = port_match
        self.icc_flags = icc_flags
    
    @classmethod
    def parse_dict(cls, input):
        # TODO
        input = {
        'name': 'Intel Xeon 2660v2',
        'clock': '2.2 GHz',
        'IACA architecture': 'IVB',
        'caheline': '64 B',
        'memory bandwidth': '60 GB/s',
        'cores': '10',
        'cache stack': 
            [{'size': '32 KB', 'type': 'per core', 'bw': '1 CL/cy'},
             {'size': '256 KB', 'type': 'per core', 'bw': '1 CL/cy'},
             {'size': '25 MB', 'type': 'per socket'}]
        }
        
        obj = cls(input['name'], arch=input['IACA architecture'], clock=input['clock'],
                  cores=input['cores'], cl_size=input['cacheline'],
                  mem_bw=input['memory bandwidth'], cache_stack=input['cache stack'])