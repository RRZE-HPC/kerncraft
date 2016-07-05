#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from __future__ import division

from ruamel import yaml
import re
import six


class PrefixedUnit(yaml.YAMLObject):
    PREFIXES = {'k': 1e3, 'M': 1e6, 'G': 1e9, 'T': 1e13, 'P': 1e16, 'E': 1e19, 'Z': 1e21, 'Y': 1e24,
                '': 1}

    yaml_loader = yaml.Loader
    yaml_dumper = yaml.Dumper

    yaml_tag = u'!prefixed'
    yaml_implicit_pattern = re.compile(re.compile(
        r'^(?P<value>[0-9]+(?:\.[0-9]+)?) (?P<prefix>[kMGTP])?(?P<unit>.*)$'))

    @classmethod
    def from_yaml(cls, loader, node):
        return PrefixedUnit(node.value)

    @classmethod
    def to_yaml(cls, dumper, data):
        return dumper.represent_scalar(cls.yaml_tag, six.text_type(data))

    def __init__(self, *args):
        if len(args) == 1:
            if isinstance(args[0], six.string_types):
                m = re.match(
                    r'^(?P<value>(?:[0-9]+(?:\.[0-9]+)?|inf)) (?P<prefix>[kMGTP])?(?P<unit>.*)$', args[0])
                assert m, "Could not parse unit parameter "+repr(args[0])
                g = m.groups()
                args = [float(g[0]), g[1], g[2]]
            else:
                args = [float(args[0]), '', '']

        if len(args) == 2:
            m = re.match(r'^(?P<prefix>[kMGTP])?(?P<unit>.+)$', args[1])
            assert m, "Could not parse unit parameter"+repr(args)
            gd = m.groupdict()
            args = [float(args[0]), gd['prefix'], gd['unit']]

        if args[1] is None:
            args[1] = ''

        assert args[1] in self.PREFIXES, "Unknown prefix: "+repr(args[1])
        self.value, self.prefix, self.unit = args

    def base_value(self):
        '''gives value without prefix'''
        return self.value*self.PREFIXES[self.prefix]

    __float__ = base_value

    def __int__(self):
        return int(self.base_value())

    def good_prefix(self, max_error=0.01, round_length=2, min_prefix='', max_prefix=None):
        '''
        returns the largest prefix where the relative error is bellow *max_error* although rounded
        by *round_length*

        if *max_prefix* is found in PrefixedUnit.PREFIXES, returned value will not exceed this
        prefix.
        if *min_prefix* is given, returned value will atleast be of that prefix (no matter the
        error)
        '''
        good_prefix = min_prefix
        base_value = self.base_value()

        for k, v in list(self.PREFIXES.items()):
            # Ignoring to large prefixes
            if max_prefix is not None and v > self.PREFIXES[max_prefix]:
                continue

            # Check that differences is < relative error *max_error*
            if abs(round(base_value/v, round_length)*v - base_value) > base_value*max_error:
                continue

            # Check that resulting number is >= 0.9
            if abs(round(base_value/v, round_length)) < 0.9:
                continue

            # Check if prefix is larger then already chosen
            if v < self.PREFIXES[good_prefix]:
                continue

            # seems to be okay
            good_prefix = k

        return good_prefix

    def with_prefix(self, prefix):
        return self.__class__(
            self.base_value()/self.PREFIXES[prefix], prefix, self.unit)

    def reduced(self):
        return self.with_prefix(self.good_prefix(max_error=0.0))

    def __str__(self):
        good_prefix = self.good_prefix()
        if self.prefix == good_prefix:
            return '{0:.2f} {1}{2}'.format(self.value, self.prefix, self.unit).strip()
        else:
            return self.with_prefix(good_prefix).__str__()

    def __repr__(self):
        return '{0}({1!r}, {2!r}, {3!r})'.format(
            self.__class__.__name__, self.value, self.prefix, self.unit)

    def __mul__(self, other):
        if isinstance(other, self.__class__):
            unit = self.unit+other.unit
        else:
            unit = self.unit

        v = self.__class__(float(self)*float(other), unit)
        return v.reduced()

    def __truediv__(self, other):
        if isinstance(other, self.__class__):
            unit = self.unit+'/'+other.unit
        else:
            unit = self.unit

        v = self.__class__(float(self)/float(other), unit)
        return v.reduced()

    def __floordiv__(self, other):
        if isinstance(other, self.__class__):
            unit = self.unit+'/'+other.unit
        else:
            unit = self.unit

        v = self.__class__(float(self)//float(other), unit)
        return v.reduced()

    def __lt__(self, other):
        return float(self) < float(other)

    def __gt__(self, other):
        return float(self) > float(other)

    def __eq__(self, other):
        try:
            return float(self) == float(other)
        except TypeError:
            return False

    def __le__(self, other):
        return float(self) <= float(other)

    def __ge__(self, other):
        return float(self) >= float(other)

    def __ne__(self, other):
        try:
            return float(self) != float(other)
        except TypeError:
            return True

# Make this tag automatic
yaml.add_implicit_resolver(PrefixedUnit.yaml_tag, PrefixedUnit.yaml_implicit_pattern)
