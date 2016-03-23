__author__ = 'hiroki'

import theano
import numpy as np

import nn_char_zeropad

theano.config.floatX = 'float32'
np.random.seed(0)


if __name__ == '__main__':
    import argparse
    import train

    parser = argparse.ArgumentParser(description='Train NN tagger.')

    parser.add_argument('-mode', type=str, default='word', help='char/word')
    parser.add_argument('--train_data', help='path to training data')
    parser.add_argument('--dev_data', help='path to development data')
    parser.add_argument('--test_data', help='path to test data')

    # NN architecture
    parser.add_argument('--vocab', type=int, default=100000000, help='vocabulary size')
    parser.add_argument('--emb', type=int, default=100, help='dimension of embeddings')
    parser.add_argument('--c_emb', type=int, default=10, help='dimension of char embeddings')
    parser.add_argument('--window', type=int, default=5, help='window size for convolution')
    parser.add_argument('--hidden', type=int, default=300, help='dimension of hidden layer')
    parser.add_argument('--c_hidden', type=int, default=50, help='dimension of char hidden layer')
    parser.add_argument('--tag', type=int, default=45, help='number of tags')
    parser.add_argument('--layer', type=int, default=2, help='number of layers')

    # training options
    parser.add_argument('--opt', default='sgd', help='optimization method')
    parser.add_argument('--reg', default=0.0001, help='L2 reg')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0075, help='learning rate')
    parser.add_argument('--init_emb', default=None, help='initial embedding file (word2vec output)')
    parser.add_argument('--check', default=False)

    argv = parser.parse_args()
    nn_char_zeropad.train(argv)

    """
    if argv.mode == 'word':
        train.train(argv)
    else:
        train.train_char(argv)
    """