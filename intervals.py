#!/usr/bin/env python

class Intervals:
    '''Very simple interval implementation for integers (might also work on floats)'''
    
    def __init__(self, *args):
        self._data = list(args)
        self._data = filter(lambda (lower, upper): upper > lower, self._data)
        self._enforce_order()
        self._enforce_no_overlap()
    
    def _enforce_order(self):
        self._data.sort(key=lambda d: d[0])
    
    def _enforce_no_overlap(self):
        i = 0
        while i+1 < len(self._data):
            if self._data[i][1] >= self._data[i+1][0]:
                # beginning of i+1-th range is contained in i-th range
                if self._data[i][1] < self._data[i+1][1]:
                    # i+1-th range is longer, thus enlarge i-th range
                    self._data[i][1] = self._data[i+1][1]
                
                # removed contained range
                del self._data[i+1]
            i += 1
    
    def __and__(self, other):
        if len(self._data) == 0:
            self._data = list(other._data)
        
        for od in other._data:
            for sd in self._data:
                if od[1] <= sd[0]:
                    # od is before sd
                    self._data.insert(0, list(od))
                    break
                elif od[0] <= sd[1]:
                    # od and sd overlap
                    sd[0] = min(sd[0], od[0])
                    sd[1] = max(sd[1], od[1])
                    break
                else:
                    # od is after sd
                    continue
            if od[0] > self._data[-1][1]:
                self._data.append(list(od))
        return self
    
    def __len__(self):
        '''returns sum of range lengths'''
        return sum(upper-lower for (lower, upper) in self._data)
    
    def __contains__(self, needle):
        return any(lower <= needle < upper for (lower, upper) in self._data)
            
    def __repr__(self):
         return str(self.__class__) + '(' + ', '.join(map(list.__repr__, self._data)) + ')'
    
    def __cmp__(self, other):
        if self._data == other._data:
            return 0
        else:
            return 1

if __name__ == '__main__':
    assert Intervals([0,10])._data == [[0,10]]
    assert Intervals([0,10], [1,9])._data == [[0,10]]
    assert Intervals([0,10], [5,15])._data == [[0,15]]
    assert Intervals([-5,5], [0,10])._data == [[-5,10]]
    assert Intervals([0,9], [10,11])._data == [[0,9], [10,11]]
    assert Intervals([0,10], [10,11])._data == [[0,11]]
    assert Intervals([0,5]) & Intervals([1,9]) == Intervals([0,9])
    assert Intervals([0,5]) & Intervals([5,9]) == Intervals([0,9])
    assert Intervals([2,4]) & Intervals([0,9]) == Intervals([0,9])
    assert len(Intervals([1,2])) == 1
    assert 10 not in Intervals([0,10]) and 0 in Intervals([0,10])