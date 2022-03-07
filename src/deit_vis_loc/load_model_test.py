#!/usr/bin/env python3

import argparse
import pickle
import sys


def parse_args(args_it):
    parser = argparse.ArgumentParser(description='Allows to load test results')
    parser.add_argument('--test-results', help='The path to the test results to load',
            required=True, metavar='FILE')
    return vars(parser.parse_args(args_it))


def read_test_results(fpath):
    with open(fpath, 'rb') as f:
        for _ in range(pickle.load(f)):
            yield pickle.load(f)


if __name__ == "__main__":
    args       = parse_args(sys.argv[1:])
    results_it = read_test_results(args['test_results'])

