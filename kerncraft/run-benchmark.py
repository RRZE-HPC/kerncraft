#!/usr/bin/env python
from __future__ import print_function, division, absolute_import, unicode_literals

import sys
import os
import subprocess
import re

# For compatability with py2.6
if "check_output" not in dir( subprocess ): # duck punch it in!
    def f(*popenargs, **kwargs):
        if 'stdout' in kwargs:
            raise ValueError('stdout argument not allowed, it will be overridden.')
        process = subprocess.Popen(stdout=subprocess.PIPE, *popenargs, **kwargs)
        output, unused_err = process.communicate()
        retcode = process.poll()
        if retcode:
            cmd = kwargs.get("args")
            if cmd is None:
                cmd = popenargs[0]
            raise subprocess.CalledProcessError(retcode, cmd)
        return output
    subprocess.check_output = f

def main():
    if len(sys.argv) != 2 and '-h' not in sys.argv and '--help' not in sys.argv:
        print('Usage:', sys.argv[0], 'BENCH')
        print()
        print('this will execute BENCH with likwid-perf and do a resonable argument sweep')
        print('retruns a space seperated lists of runtimes and data volumes')
    benchmark = sys.argv[1]

if __name__ == '__main__':
    main()