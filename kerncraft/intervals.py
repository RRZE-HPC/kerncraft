#!/usr/bin/env python
'''
A simple interval implementation
'''


class Intervals(object):
    '''Very simple interval implementation for integers (might also work on floats)'''

    def __init__(self, *args, **kwargs):
        '''If keywords *sane* is True (default: False), checks will not be done on given data.'''
        self.data = list(args)
        if not kwargs.get('sane', False):
            self.data = [d for d in self.data if d[1] > d[0]]
            self._enforce_order()
            self._enforce_no_overlap()

    def _enforce_order(self):
        '''Enforces the order of all entries in internal storage'''
        self.data.sort(key=lambda d: d[0])

    def _enforce_no_overlap(self, start_at=0):
        '''Enforces that no ranges overlap in internal storage'''
        i = start_at
        while i+1 < len(self.data):
            if self.data[i][1] >= self.data[i+1][0]:
                # beginning of i+1-th range is contained in i-th range
                if self.data[i][1] < self.data[i+1][1]:
                    # i+1-th range is longer, thus enlarge i-th range
                    self.data[i][1] = self.data[i+1][1]

                # removed contained range
                del self.data[i+1]
            i += 1

    def __and__(self, other):
        '''Combines two intervals, under the assumption that they are sane'''
        return Intervals(*(self.data+other.data))

    def __len__(self):
        '''Returns sum of range lengths'''
        return int(sum(upper-lower for (lower, upper) in self.data))

    def __contains__(self, needle):
        return any(lower <= needle < upper for (lower, upper) in self.data)

    def __repr__(self):
        return str(self.__class__) + '(' + ', '.join([list.__repr__(d) for d in self.data]) + ')'

    def __eq__(self, other):
        return self.data == other.data
