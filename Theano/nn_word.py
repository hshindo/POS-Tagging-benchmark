__author__ = 'hiroki'

import sys
import time
import math
from collections import defaultdict

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d

from util import load_init_emb, UNK, RE_NUM, Vocab
from nn_utils import sample_weights, relu
from optimizers import sgd, ada_grad

theano.config.floatX = 'float32'


class Model(object):
    def __init__(self, x, y, opt, lr, init_emb, vocab_size, window, n_emb, n_h, n_y):
        """
        :param n_emb: dimension of word embeddings
        :param window: window size
        :param n_h: dimension of hidden layer
        :param n_y: number of tags
        x: 1D: batch size * window, 2D: emb_dim
        h: 1D: batch_size, 2D: hidden_dim
        """

        assert window % 2 == 1, 'Window size must be odd'

        """ input """
        self.x = x
        self.y = y

        n_words = x.shape[0]

        """ params """
        if init_emb is not None:
            self.emb = theano.shared(init_emb)
        else:
            self.emb = theano.shared(sample_weights(vocab_size, n_emb))

        self.W_in = theano.shared(sample_weights(n_h, 1, window, n_emb))
        self.W_out = theano.shared(sample_weights(n_h, n_y))

        self.b_in = theano.shared(sample_weights(n_h, 1))
        self.b_y = theano.shared(sample_weights(n_y))

        self.params = [self.W_in, self.W_out, self.b_in, self.b_y]

        """ pad """
        self.zero = theano.shared(np.zeros(shape=(1, 1, window / 2, n_emb), dtype=theano.config.floatX))

        """ look up embedding """
        self.x_emb = self.emb[self.x]  # x_emb: 1D: n_words, 2D: n_emb

        """ convolution """
        self.x_in = self.conv(self.x_emb)

        """ feed-forward computation """
        self.h = relu(self.x_in.reshape((self.x_in.shape[1], self.x_in.shape[2])) + T.repeat(self.b_in, T.cast(self.x_in.shape[2], 'int32'), 1)).T
        self.p_y_given_x = T.nnet.softmax(T.dot(self.h, self.W_out) + self.b_y)

        """ prediction """
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.result = T.eq(self.y_pred, self.y)

        """ cost function """
        self.nll = -T.sum(T.log(self.p_y_given_x)[T.arange(n_words), self.y])
        self.cost = self.nll

        if opt == 'sgd':
            self.updates = sgd(self.cost, self.params, self.emb, self.x_emb, lr)
        else:
            self.updates = ada_grad(self.cost, self.params, self.emb, self.x_emb, self.x, lr)

    def conv(self, x_emb):
        x_padded = T.concatenate([self.zero, x_emb.reshape((1, 1, x_emb.shape[0], x_emb.shape[1])), self.zero], axis=2)  # x_padded: 1D: n_words + n_pad, 2D: n_phi
        return conv2d(input=x_padded, filters=self.W_in)


def load_conll(path, vocab_word, data_size=100000, vocab_tag=Vocab(), vocab_size=None, file_encoding='utf-8'):
    corpus = []
    word_freqs = defaultdict(int)

    if vocab_word is None:
        register = True
        vocab_word = Vocab()
    else:
        register = False

    if register:
        vocab_word.add_word(UNK)

    with open(path) as f:
        wts = []
        for line in f:
            es = line.rstrip().split('\t')
            if len(es) == 10:
                word = es[1].decode(file_encoding)
                word = RE_NUM.sub(u'0', word)
                tag = es[4].decode(file_encoding)

                wt = (word, tag)
                wts.append(wt)
                word_freqs[word.lower()] += 1
                vocab_tag.add_word(tag)
            else:
                # reached end of sentence
                corpus.append(wts)
                wts = []
                if len(corpus) == data_size:
                    break
        if wts:
            corpus.append(wts)

    if register:
        for w, f in sorted(word_freqs.items(), key=lambda (k, v): -v):
            if vocab_size is None or vocab_word.size() < vocab_size:
                vocab_word.add_word(w)
            else:
                break

    return corpus, vocab_word, vocab_tag


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
                w_id = vocab_word.get_id(UNK)

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

    print '\tINITIAL EMBEDDING\t%s' % args.init_emb
    print '\tWORD VECTOR\t\tEmb Dim: %d  Hidden Dim: %d' % (args.emb, args.hidden)
    print '\tOPTIMIZATION\t\tMethod: %s  Learning Rate: %f  L2 Reg: %f' % (args.opt, args.lr, args.reg)

    """ load pre-trained embeddings """
    vocab_word = None
    init_emb = None
    if args.init_emb:
        print '\tLoading Embeddings...'
        init_emb, vocab_word = load_init_emb(args.init_emb)
        n_emb = init_emb.shape[1]
    else:
        n_emb = args.emb

    """ load data """
    print '\tLoading Data...'
    train_corpus, vocab_word, vocab_tag = load_conll(path=args.train_data, data_size=args.data_size, vocab_word=vocab_word, vocab_size=args.vocab)
    dev_corpus, _, _ = load_conll(path=args.dev_data, vocab_word=vocab_word)
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
                   n_emb=n_emb, n_h=n_h, n_y=n_y)

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

            print '\nEpoch: %d' % (epoch + 1)
            print '\tBatch Index: ',
            start = time.time()

            total = 0.0
            correct = 0
            losses = 0.0
            for index in xrange(len(tr_sample_x)):
                if index % 100 == 0 and index != 0:
                    print index,
                    sys.stdout.flush()

                loss, corrects = train_model(tr_sample_x[index], tr_sample_y[index], _lr)
                assert math.isnan(loss) is False, index

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
    parser.add_argument('--init_emb', default=None, help='initial embedding file (word2vec output)')

    argv = parser.parse_args()
    train(argv)
