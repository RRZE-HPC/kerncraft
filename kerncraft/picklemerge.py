#!/usr/bin/env python

from __future__ import print_function

import argparse
import ast
import sys
import os.path
import pickle


def main():
    parser = argparse.ArgumentParser(
        description='Merges two or more pickle files. Only supports pickles consisting of a single '
        'dictionary object.')
    parser.add_argument('destination', type=argparse.FileType('r+'),
                        help='File to write to and include in resulting pickle. (WILL BE CHANGED)')
    parser.add_argument('source', type=argparse.FileType('r'), nargs='+',
                        help='File to include in resulting pickle.')

    args = parser.parse_args()

    result = pickle.load(args.destination)
    assert type(result) is dict, "only dictionaries can be handled."
    
    for s in args.source:
        data = pickle.load(s)
        assert type(data) is dict, "only dictionaries can be handled."
        result.update(data)
    
    args.destination.seek(0)
    args.destination.truncate()
    pickle.dump(result, args.destination)

if __name__ == '__main__':
    main()