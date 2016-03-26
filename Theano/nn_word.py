__author__ = 'hiroki'

import sys
import time
import math
import random

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d

import util
from nn_utils import sample_weights, relu
from optimizers import sgd, ada_grad

theano.config.floatX = 'float32'


class Model(object):
    def __init__(self, name, x, y, lr, init_emb, vocab_size, emb_dim, hidden_dim, output_dim, window, opt):

        assert window % 2 == 1, 'Window size must be odd'

        """ input """
        self.name = name
        self.x = x
        self.y = y
        self.lr = lr
        self.input = [self.x, self.y, self.lr]

        n_words = x.shape[0]

        """ params """
        if init_emb is not None:
            self.emb = theano.shared(init_emb)
        else:
            self.emb = theano.shared(sample_weights(vocab_size, emb_dim))

        self.W_in = theano.shared(sample_weights(hidden_dim, 1, window, emb_dim))
        self.W_out = theano.shared(sample_weights(hidden_dim, output_dim))

        self.b_in = theano.shared(sample_weights(hidden_dim, 1))
        self.b_y = theano.shared(sample_weights(output_dim))

        self.params = [self.W_in, self.W_out, self.b_in, self.b_y]

        """ pad """
        self.zero = theano.shared(np.zeros(shape=(1, 1, window / 2, emb_dim), dtype=theano.config.floatX))

        """ look up embedding """
        self.x_emb = self.emb[self.x]  # x_emb: 1D: n_words, 2D: n_emb

        """ convolution """
        self.x_in = self.conv(self.x_emb)

        """ feed-forward computation """
        self.h = relu(self.x_in.reshape((self.x_in.shape[1], self.x_in.shape[2])) + T.repeat(self.b_in, T.cast(self.x_in.shape[2], 'int32'), 1)).T
        self.o = T.dot(self.h, self.W_out) + self.b_y
        self.p_y_given_x = T.nnet.softmax(self.o)

        """ prediction """
        self.y_pred = T.argmax(self.o, axis=1)
        self.result = T.eq(self.y_pred, self.y)

        """ cost function """
        self.nll = -T.sum(T.log(self.p_y_given_x)[T.arange(n_words), self.y])
        self.cost = self.nll

        if opt == 'sgd':
            self.updates = sgd(self.cost, self.params, self.emb, self.x_emb, self.lr)
        else:
            self.updates = ada_grad(self.cost, self.params, self.emb, self.x_emb, self.x, self.lr)

    def conv(self, x_emb):
        x_padded = T.concatenate([self.zero, x_emb.reshape((1, 1, x_emb.shape[0], x_emb.shape[1])), self.zero], axis=2)  # x_padded: 1D: n_words + n_pad, 2D: n_phi
        return conv2d(input=x_padded, filters=self.W_in)


def convert_into_ids(corpus, vocab_word, vocab_tag):
    id_corpus_w = []
    id_corpus_t = []

    for sent in corpus:
        w_ids = []
        t_ids = []

        for w, t in sent:
            w_id = vocab_word.get_id(w.lower())
            t_id = vocab_tag.get_id(t)

            if w_id is None:
                w_id = vocab_word.get_id(util.UNK)

            assert w_id is not None
            assert t_id is not None

            w_ids.append(w_id)
            t_ids.append(t_id)

        id_corpus_w.append(np.asarray(w_ids, dtype='int32'))
        id_corpus_t.append(np.asarray(t_ids, dtype='int32'))

    assert len(id_corpus_w) == len(id_corpus_t)
    return id_corpus_w, id_corpus_t


def train(args):
    print '\nNEURAL POS TAGGER START\n'

    print '\tINITIAL EMBEDDING\t%s' % args.emb_list
    print '\tWORD VECTOR\t\tEmb Dim: %d  Hidden Dim: %d' % (args.emb, args.hidden)
    print '\tOPTIMIZATION\t\tMethod: %s  Learning Rate: %f  L2 Reg: %f' % (args.opt, args.lr, args.reg)

    train_corpus, vocab_word, vocab_char, vocab_tag = util.load_conll(args.train_data)

    if args.dev_data:
        print 'Loading test data...'
        dev_corpus, vocab_word_test, dev_vocab_char, dev_vocab_tag = util.load_conll(args.dev_data)

        for w in vocab_word_test.i2w:
            if args.vocab_size is None or vocab_word.size() < args.vocab_size:
                vocab_word.add_word(w)
        for c in dev_vocab_char.i2w:
            vocab_char.add_word(c)
        for t in dev_vocab_tag.i2w:
            vocab_tag.add_word(t)

    # load pre-trained embeddings
    init_emb = None
    if args.emb_list:
        print 'Loading word embeddings...'
        init_emb = util.load_init_emb(args.emb_list, args.vocab_list, vocab_word)
        emb_dim = init_emb.shape[1]
    else:
        emb_dim = args.emb_dim

    train_corpus = train_corpus[:argv.data_size]

    print '\tTrain Sentences: %d  Test Sentences: %d' % (len(train_corpus), len(dev_corpus))
    print '\tVocab size: %d' % vocab_word.size()

    """ converting into ids """
    print '\tConverting into IDs...'
    # batches: 1D: n_batch, 2D: [0]=word id 2D matrix, [1]=tag id 2D matrix
    tr_sample_x, tr_sample_y = convert_into_ids(train_corpus, vocab_word, vocab_tag)
    dev_sample_x, dev_sample_y = convert_into_ids(dev_corpus, vocab_word, vocab_tag)

    """ symbol definition """
    window = args.window
    opt = args.opt
    n_h = args.hidden
    n_y = args.tag

    print '\tCompiling Theano Code...'
    x = T.ivector()
    y = T.ivector()
    lr = T.fscalar('lr')

    """ tagger set up """
    tagger = Model(x=x, y=y, opt=opt, lr=lr, init_emb=init_emb, vocab_size=vocab_word.size(), window=window,
                   emb_dim=emb_dim, hidden_dim=n_h, output_dim=n_y)

    train_model = theano.function(
        inputs=[x, y, lr],
        outputs=[tagger.nll, tagger.result],
        updates=tagger.updates,
        mode='FAST_RUN'
    )

    valid_model = theano.function(
        inputs=[x, y],
        outputs=tagger.result,
        mode='FAST_RUN'
    )

    def _train():
        for epoch in xrange(args.epoch):
            _lr = argv.lr / float(epoch+1)
            indices = range(len(tr_sample_x))

            random.shuffle(indices)

            print '\nEpoch: %d' % (epoch + 1)
            print '\tBatch Index: ',
            start = time.time()

            total = 0.0
            correct = 0
            losses = 0.0
            for i, index in enumerate(indices):
                if i % 100 == 0 and i != 0:
                    print i,
                    sys.stdout.flush()

                loss, corrects = train_model(tr_sample_x[index], tr_sample_y[index], _lr)
                assert math.isnan(loss) is False, index

                print 'Index:%d  Loss:%f  Acc:%f  Len:%d' % (index, loss, np.sum(corrects) / float(len(corrects)), len(tr_sample_x[index]))

                total += len(corrects)
                correct += np.sum(corrects)
                losses += loss

            end = time.time()
            print '\tTime: %f seconds' % (end - start)
            print '\tNegative Log Likelihood: %f' % losses
            print '\tAccuracy:%f  Total:%d  Correct:%d' % ((correct / total), total, correct)

            _dev(valid_model)

    def _dev(model):
        print '\tBatch Index: ',
        start = time.time()

        total = 0.0
        correct = 0

        for index in xrange(len(dev_sample_x)):
            if index % 100 == 0 and index != 0:
                print index,
                sys.stdout.flush()

            corrects = model(dev_sample_x[index], dev_sample_y[index])

            total += len(corrects)
            correct += np.sum(corrects)

        end = time.time()

        print '\tTime: %f seconds' % (end - start)
        print '\tAccuracy:%f  Total:%d  Correct:%d' % ((correct / total), total, correct)

    _train()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train NN tagger.')

    parser.add_argument('--train_data', help='path to training data')
    parser.add_argument('--dev_data', help='path to development data')
    parser.add_argument('--test_data', help='path to test data')
    parser.add_argument('--data_size', type=int, default=100000)

    # NN architecture
    parser.add_argument('--vocab', type=int, default=100000000, help='vocabulary size')
    parser.add_argument('--emb', type=int, default=100, help='dimension of embeddings')
    parser.add_argument('--window', type=int, default=5, help='window size for convolution')
    parser.add_argument('--hidden', type=int, default=300, help='dimension of hidden layer')
    parser.add_argument('--tag', type=int, default=45, help='number of tags')
    parser.add_argument('--layer', type=int, default=2, help='number of layers')

    # training options
    parser.add_argument('--opt', default='sgd', help='optimization method')
    parser.add_argument('--reg', default=0.0001, help='L2 reg')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0075, help='learning rate')
    parser.add_argument('--emb_list', default=None, help='initial embedding file (word2vec output)')
    parser.add_argument('--vocab_list', default=None, help='initial vocab file (word2vec output)')
    parser.add_argument('--vocab_size', default=None)

    argv = parser.parse_args()
    train(argv)
