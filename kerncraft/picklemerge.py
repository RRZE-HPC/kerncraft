#!/usr/bin/env python3
"""Merge two pickle files containing dictionarys recursively."""
import argparse
import pickle
import collections


def update(d, u):
    """
    Update dictionary recursivly.

    Origin: http://stackoverflow.com/a/3233356/2754040
    """
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            r = update(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]
    return d


def main():
    """Comand line interface of picklemerge."""
    parser = argparse.ArgumentParser(
        description='Recursively merges two or more pickle files. Only supports pickles consisting '
        'of a single dictionary object.')
    parser.add_argument('destination', type=argparse.FileType('r+b'),
                        help='File to write to and include in resulting pickle. (WILL BE CHANGED)')
    parser.add_argument('source', type=argparse.FileType('rb'), nargs='+',
                        help='File to include in resulting pickle.')

    args = parser.parse_args()

    result = pickle.load(args.destination)
    assert isinstance(result, collections.Mapping), "only Mapping types can be handled."

    for s in args.source:
        data = pickle.load(s)
        assert isinstance(data, collections.Mapping), "only Mapping types can be handled."

        update(result, data)

    args.destination.seek(0)
    args.destination.truncate()
    pickle.dump(result, args.destination)


if __name__ == '__main__':
    main()
