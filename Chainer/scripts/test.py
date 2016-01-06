#!/usr/bin/env python

from ctagger import test


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train NN tagger.')

    parser.add_argument('data', help='path to test data')
    parser.add_argument('model', help='path to model')

    test.test(parser.parse_args())

