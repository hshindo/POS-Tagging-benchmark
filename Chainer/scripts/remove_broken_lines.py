#!/usr/bin/env python

import sys


def main(args):
    broken_ids = set()
    dim = args.dim

    # get broken lines and remove from embedding file
    with open(args.emb) as f_orig, open(args.emb_out, 'w') as f_dest:
        for i, line in enumerate(f_orig):
            line_strip = line.strip()
            es = line_strip.split()
            if dim is None:
                # get dimension of first line
                dim = len(es)

            if dim != len(es):
                # line has different number of dimensions
                print >> sys.stderr, 'Line {} is removed, since its number of dimension is {}'.format(i + 1, len(es))
                broken_ids.add(i)
            else:
                print >> f_dest, line_strip

    # remove broken lines from word.lst
    with open(args.words) as f_orig, open(args.words_out, 'w') as f_dest:
        for i, line in enumerate(f_orig):
            if i not in broken_ids:
                print >> f_dest, line.strip()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Remove lines with missing dimensions of word2vec files.')

    parser.add_argument('emb', help='embedding file')
    parser.add_argument('emb_out', help='destination of embedding file')

    parser.add_argument('words', help='words.lst')
    parser.add_argument('words_out', help='destination of words.lst')

    parser.add_argument('--dim', type=int, default=None, help='dimension of each line (default: use dimension of first line')

    main(parser.parse_args())