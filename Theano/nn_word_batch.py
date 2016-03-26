import sys
import time
import math
import random

import util
import numpy as np
import theano
import theano.tensor as T

from nn_utils import sample_weights, relu
from optimizers import sgd, ada_grad

theano.config.floatX = 'float32'


class Model(object):
    def __init__(self, x, y, n_words, batch_size, opt, lr, init_emb, vocab_size, window, n_emb, n_h, n_y):
        assert window % 2 == 1, 'Window size must be odd'

        """ input """
        self.x = x  # 1D: n_words * batch_size, 2D: window; elem=word id
        self.x_v = x.flatten()  # 1D: n_words * batch_size * window; elem=word id
        self.y = y
        self.batch_size = batch_size
        self.n_words = n_words
        self.lr = lr

        """ params """
        if init_emb is not None:
            self.emb = theano.shared(init_emb)
        else:
            self.emb = theano.shared(sample_weights(vocab_size, n_emb))

        self.W_in = theano.shared(sample_weights(n_emb * window, n_h))
        self.W_out = theano.shared(sample_weights(n_h, n_y))

        self.b_in = theano.shared(sample_weights(n_h))
        self.b_y = theano.shared(sample_weights(n_y))

        self.params = [self.W_in, self.W_out, self.b_in, self.b_y]

        """ look up embedding """
        self.x_emb = self.emb[self.x_v]  # x_emb: 1D: batch_size * n_words * window, 2D: emb_dim

        """ forward """
        self.h = relu(T.dot(self.x_emb.reshape((batch_size * n_words, n_emb * window)), self.W_in) + self.b_in)
        self.o = T.dot(self.h, self.W_out) + self.b_y
        self.p_y_given_x = T.nnet.softmax(self.o)

        """ predict """
        self.y_pred = T.argmax(self.o, axis=1)
        self.result = T.eq(self.y_pred, self.y)

        """ loss """
        self.log_p = T.log(self.p_y_given_x)[T.arange(batch_size * n_words), self.y]
#        self.nll = T.mean(-T.sum(self.log_p.reshape((batch_size, n_words)), 0))
        self.nll = -T.sum(self.log_p)
        self.cost = self.nll

        if opt == 'sgd':
            self.updates = sgd(self.cost, self.params, self.emb, self.x_emb, self.lr)
        else:
            self.updates = ada_grad(self.cost, self.params, self.emb, self.x_emb, self.x, self.lr)


def set_minibatch(id_x, id_y, batch_size, window=5):
    samples_x = []
    samples_y = []
    batch_indices = []

    p = window / 2
    zero_pad_w = [0 for i in xrange(p)]

    prev_sent_len = -1
    b_index = 0
    bob = 0
    eob = 0
    for i in xrange(len(id_x)):
        sent_x = id_x[i]  # sent_w: 1D: n_words; elem=word id
        sent_len = len(sent_x)
        sent_x = zero_pad_w + sent_x + zero_pad_w

        samples_x.extend([sent_x[j: j+window] for j in xrange(sent_len)])
        samples_y.extend(id_y[i])

        if b_index < batch_size and (sent_len == prev_sent_len or prev_sent_len < 0):
            pass
        else:
            batch_indices.append((bob, eob, prev_sent_len, b_index))
            bob = eob
            b_index = 0

        prev_sent_len = sent_len
        eob += sent_len
        b_index += 1

    batch_indices.append((bob, eob, prev_sent_len, b_index))
    return samples_x, samples_y, batch_indices


def shared_samples(samples_x, samples_y):
    def shared(samples):
        return theano.shared(np.asarray(samples, dtype='int32'))
    return shared(samples_x), shared(samples_y)


def train(args):
    print '\nNEURAL POS TAGGER START\n'

    print '\tINITIAL EMBEDDING\t%s %s' % (args.word_list, args.emb_list)
    print '\tWORD\t\t\tEmb Dim: %d  Hidden Dim: %d' % (args.w_emb_dim, args.w_hidden_dim)
    print '\tCHARACTER\t\tEmb Dim: %d  Hidden Dim: %d' % (args.c_emb_dim, args.c_hidden_dim)
    print '\tOPTIMIZATION\t\tMethod: %s  Learning Rate: %f\n' % (args.opt, args.lr)
    print '\tMINI-BATCH: %d\n' % args.batch_size

    """ load data """
    print 'Loading data sets...\n'
    train_corpus, vocab_word, vocab_char, vocab_tag = util.load_conll(args.train_data)
    train_corpus.sort(key=lambda a: len(a))

    dev_corpus = None
    if args.dev_data:
        dev_corpus, dev_vocab_word, dev_vocab_char, dev_vocab_tag = util.load_conll(args.dev_data)

        for w in dev_vocab_word.i2w:
            if args.vocab_size is None or vocab_word.size() < args.vocab_size:
                vocab_word.add_word(w)
        for c in dev_vocab_char.i2w:
            vocab_char.add_word(c)
        for t in dev_vocab_tag.i2w:
            vocab_tag.add_word(t)

    if args.save:
        util.dump_data(vocab_word, 'vocab_word')
        util.dump_data(vocab_char, 'vocab_char')
        util.dump_data(vocab_tag, 'vocab_tag')

    """ load pre-trained embeddings """
    init_w_emb = None
    if args.emb_list:
        print '\tLoading word embeddings...\n'
        init_w_emb = util.load_init_emb(args.emb_list, args.word_list, vocab_word)
        w_emb_dim = init_w_emb.shape[1]
    else:
        w_emb_dim = args.w_emb_dim

    """ limit data set """
    train_corpus = train_corpus[:args.data_size]

    """ converting into ids """
    print '\nConverting into IDs...\n'
    tr_x, tr_c, tr_b, tr_y = util.convert_into_ids_batch(train_corpus, vocab_word, vocab_char, vocab_tag)
    tr_x, tr_y, tr_b = set_minibatch(tr_x, tr_y, args.batch_size)
    tr_x, tr_y = shared_samples(tr_x, tr_y)

    if args.dev_data:
        dev_x, dev_c, dev_b, dev_y = util.convert_into_ids_batch(dev_corpus, vocab_word, vocab_char, vocab_tag)
        dev_x, dev_y, dev_b = set_minibatch(dev_x, dev_y, 1)
        dev_x, dev_y = shared_samples(dev_x, dev_y)
        print '\tTrain Sentences: %d  Dev Sentences: %d' % (len(train_corpus), len(dev_corpus))
    else:
        print '\tTrain Sentences: %d' % len(train_corpus)

    print '\tWord size: %d  Char size: %d' % (vocab_word.size(), vocab_char.size())

    """ symbol definition """
    n_h = args.w_hidden_dim
    n_y = vocab_tag.size()
    window = args.window
    opt = args.opt

    print '\tCompiling Theano Code...'
    bos = T.iscalar('bos')
    eos = T.iscalar('eos')
    n_words = T.iscalar('n_words')
    batch_size = T.iscalar('batch_size')
    x = T.imatrix('x')
    y = T.ivector('y')
    lr = T.fscalar('lr')

    """ tagger set up """
    tagger = Model(x=x, y=y, n_words=n_words, batch_size=batch_size, opt=opt, lr=lr, init_emb=init_w_emb,
                   vocab_size=vocab_word.size(), window=window, n_emb=w_emb_dim, n_h=n_h, n_y=n_y)

    train_model = theano.function(
        inputs=[bos, eos, n_words, batch_size, lr],
        outputs=[tagger.nll, tagger.result],
        updates=tagger.updates,
        givens={
            x: tr_x[bos: eos],
            y: tr_y[bos: eos]
        },
        mode='FAST_RUN'
    )

    valid_model = theano.function(
        inputs=[bos, eos, n_words, batch_size],
        outputs=tagger.result,
        givens={
            x: dev_x[bos: eos],
            y: dev_y[bos: eos]
        },
        mode='FAST_RUN'
    )

    def _train():
        for epoch in xrange(args.epoch):
            _lr = args.lr / float(epoch+1)
            indices = range(len(tr_b))
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

                boundary = tr_b[index]
                loss, corrects = train_model(boundary[0], boundary[1], boundary[2],boundary[3],  _lr)

                assert math.isnan(loss) is False, i

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

        for index in xrange(len(dev_b)):
            if index % 100 == 0 and index != 0:
                print index,
                sys.stdout.flush()

            boundary = dev_b[index]
            corrects = model(boundary[0], boundary[1], boundary[2], boundary[3])

            total += len(corrects)
            correct += np.sum(corrects)

        end = time.time()
        print '\tTime: %f seconds' % (end - start)
        print '\tAccuracy:%f  Total:%d  Correct:%d' % ((correct / total), total, correct)

    _train()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train NN tagger.')

    """ Mode """
    parser.add_argument('-mode', default='train', help='train/test')

    """ Model """
    parser.add_argument('--model', default='char', help='word/char')
    parser.add_argument('--save', type=bool, default=True, help='save model')
    parser.add_argument('--load', type=str, default=None, help='load model')

    """ Data """
    parser.add_argument('--train_data', help='path to training data')
    parser.add_argument('--dev_data', help='path to development data')
    parser.add_argument('--test_data', help='path to test data')
    parser.add_argument('--data_size', type=int, default=100000)
    parser.add_argument('--vocab_size', type=int, default=100000000)

    """ Neural Architectures """
    parser.add_argument('--emb_list', default=None, help='initial embedding list file (word2vec output)')
    parser.add_argument('--word_list', default=None, help='initial word list file (word2vec output)')
    parser.add_argument('--w_emb_dim', type=int, default=100, help='dimension of word embeddings')
    parser.add_argument('--c_emb_dim', type=int, default=10, help='dimension of char embeddings')
    parser.add_argument('--w_hidden_dim', type=int, default=300, help='dimension of word hidden layer')
    parser.add_argument('--c_hidden_dim', type=int, default=50, help='dimension of char hidden layer')
    parser.add_argument('--window', type=int, default=5, help='window size for convolution')
    parser.add_argument('--batch_size', type=int, default=32, help='mini batch size')

    """ Training Parameters """
    parser.add_argument('--opt', default='sgd', help='optimization method')
    parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=0.0075, help='learning rate')

    args = parser.parse_args()

    train(args=args)
