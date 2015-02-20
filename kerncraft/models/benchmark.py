class Benchmark:
    """
    this will produce a benchmarkable binary to be used with likwid
    """

    name = "benchmark"

    @classmethod
    def configure_arggroup(cls, parser):
        pass

    def __init__(self, kernel, machine, args=None, parser=None):
        """
        *kernel* is a Kernel object
        *machine* describes the machine (cpu, cache and memory) characteristics
        *args* (optional) are the parsed arguments from the comand line
        """
        self.kernel = kernel
        self.machine = machine
        self._args = args

        if args:
            # handle CLI info
            pass

    def analyze(self):
        print(self.kernel.build())

    def report(self):
        pass