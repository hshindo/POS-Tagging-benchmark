import sys
import time
import math
from collections import defaultdict

from util import load_init_emb, PAD, UNK, RE_NUM, Vocab

import numpy as np
import theano
import theano.tensor as T

from nn_utils import build_shared_zeros, sample_weights, sample_norm_dist, relu
from optimizers import sgd, ada_grad

theano.config.floatX = 'float32'
#np.random.seed(0)


class Model(object):
    def __init__(self, x, c, y, opt, lr, init_emb, vocab_size, char_size, window, n_emb, n_c_emb, n_h, n_c_h, n_y):
        assert window % 2 == 1, 'Window size must be odd'

        """ input """
        self.x = x  # 1D: n_words, 2D: window; elem=word id
        self.x_v = x.flatten()
        self.c = c  # 1D: n_words, 2D: window, 3D: n_chars; elem=char id
        self.y = y

        n_phi = (n_emb + n_c_h) * window
        n_words = T.cast(self.x.shape[0], dtype='int32')

        """ params """
#        self.zero = theano.shared(np.zeros(shape=(1, n_emb), dtype=theano.config.floatX))
        if init_emb is not None:
            self.emb = theano.shared(init_emb)
#            self.E = T.concatenate([self.zero, self.emb], 0)
            self.E = self.emb
        else:
            self.emb = theano.shared(sample_weights(vocab_size, n_emb))
            self.E = self.emb

        self.pad = build_shared_zeros((1, n_c_emb))
        self.e_c = theano.shared(sample_norm_dist(char_size - 1, n_c_emb))
        self.emb_c = T.concatenate([self.pad, self.e_c], 0)

        self.W_in = theano.shared(sample_weights(n_phi, n_h))
        self.W_c = theano.shared(sample_weights(n_c_emb, n_c_h))
        self.W_out = theano.shared(sample_weights(n_h, n_y))

        self.b_in = theano.shared(sample_weights(n_h))
        self.b_c = theano.shared(sample_weights(n_c_h))
        self.b_y = theano.shared(sample_weights(n_y))

        self.params = [self.e_c, self.W_in, self.W_c, self.W_out, self.b_in, self.b_c, self.b_y]

        """ look up embedding """
        self.x_emb = self.E[self.x_v]  # x_emb: 1D: n_words, 2D: window, 3D: n_emb
        self.x_emb_r = self.x_emb.reshape((x.shape[0], x.shape[1], -1))
        self.c_emb = self.emb_c[self.c]  # c_emb: 1D: n_words, 2D: window, 3D: n_chars, 4D: n_c_emb

        """ convolution """
        self.c_phi = T.max(T.dot(self.c_emb, self.W_c) + self.b_c, 2)  # 1D: n_words, 2D: window, 3D: n_h_c
        self.x_phi = T.concatenate([self.x_emb_r, self.c_phi], axis=2)

        """ output """
        self.h = relu(T.dot(self.x_phi.reshape((self.x_phi.shape[0], -1)), self.W_in) + self.b_in)
        self.p_y_given_x = T.nnet.softmax(T.dot(self.h, self.W_out) + self.b_y)

        """ predict """
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.result = T.eq(self.y_pred, self.y)

        """ loss """
        self.nll = -T.sum(T.log(self.p_y_given_x)[T.arange(n_words), self.y])
        self.cost = self.nll

        if opt == 'sgd':
            self.updates = sgd(self.cost, self.params, self.emb, self.x_emb, lr)
        else:
            self.updates = ada_grad(self.cost, self.params, self.emb, self.x_emb, self.x, lr)


def load_conll(path, _train, vocab_word, data_size=100000, vocab_char=Vocab(), vocab_tag=Vocab(), vocab_size=None, file_encoding='utf-8'):
    corpus = []
    word_freqs = defaultdict(int)
    char_freqs = defaultdict(int)
    max_char_len = -1

    if vocab_word is None:
        register = True
        vocab_word = Vocab()
    else:
        register = False

    if register:
        vocab_word.add_word(PAD)
        vocab_word.add_word(UNK)

    if _train:
        vocab_char.add_word(PAD)
        vocab_char.add_word(UNK)

    with open(path) as f:
        wts = []
        for line in f:
            es = line.rstrip().split('\t')
            if len(es) == 10:
                word = es[1].decode(file_encoding)
                word = RE_NUM.sub(u'0', word)
                tag = es[4].decode(file_encoding)

                max_char_len = len(word) if max_char_len < len(word) else max_char_len

                for c in word:
                    char_freqs[c] += 1

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
    if _train:
        for c, f in sorted(char_freqs.items(), key=lambda (k, v): -v):
            vocab_char.add_word(c)

    return corpus, vocab_word, vocab_char, vocab_tag, max_char_len


def convert_into_ids(corpus, vocab_word, vocab_char, vocab_tag, max_char_len):
    id_corpus_w = []
    id_corpus_c = []
    id_corpus_t = []

    for sent in corpus:
        w_ids = []
        c_ids = []

        for w, t in sent:
            w_id = vocab_word.get_id(w.lower())
            t_id = vocab_tag.get_id(t)
            c_id_w = []

            if w_id is None:
                w_id = vocab_word.get_id(UNK)

            assert w_id is not None
            assert t_id is not None

            for c in w:
                c_id = vocab_char.get_id(c)
                if c_id is None:
                    c_id = vocab_char.get_id(UNK)
                c_id_w.append(c_id)

            w_ids.append(w_id)
            id_corpus_t.append(t_id)
            c_ids.append(c_id_w)

        id_corpus_w.append(w_ids)
        id_corpus_c.append(zero_pad_char(c_ids, max_char_len))

    assert len(id_corpus_w) == len(id_corpus_c)
    return id_corpus_w, id_corpus_c, id_corpus_t


def zero_pad_char(char_ids, max_char_len, window=5):
    new = []
    p = window / 2
    for c_ids in char_ids:  # char_ids: 1D: n_words, 2D: n_chars
        pre = [0 for i in xrange(p)]
        pad = [0 for i in xrange(max_char_len - len(c_ids) + p)]
        new.append(pre + c_ids + pad)
    return new


def get_samples(id_corpus_w, id_corpus_c, max_char_len, window=5):
    assert len(id_corpus_w) == len(id_corpus_c)

    samples_w = []
    samples_c = []
    sent_boundary = []

    p = window / 2
    zero_pad_w = [0 for i in xrange(p)]
    zero_char = [0 for i in xrange(max_char_len + 2 * p)]
    zero_pad_c = [zero_char for i in xrange(p)]

    bos = 0
    for i in xrange(len(id_corpus_w)):
        sent_w = id_corpus_w[i]  # sent_w: 1D: n_words; elem=word id
        sent_c = id_corpus_c[i]  # sent_c: 1D: n_words, 2D: n_chars; elem=char_id
        sent_len = len(sent_w)

        sent_w = zero_pad_w + sent_w + zero_pad_w
        sent_c = zero_pad_c + sent_c + zero_pad_c

        for j in xrange(sent_len):
            center = j + p
            sample_w = sent_w[center-p: center+p+1]
            sample_c = sent_c[center-p: center+p+1]

            samples_w.append(sample_w)
            samples_c.append(sample_c)

            assert len(sample_w) == len(sample_c)

        sent_boundary.append((bos, bos + sent_len))
        bos += sent_len

    assert len(samples_w) == len(samples_c)
    return samples_w, samples_c, sent_boundary


def shared_samples(samples_w, samples_c, samples_y):
    def shared(samples):
        return theano.shared(np.asarray(samples, dtype='int32'))
    return shared(samples_w), shared(samples_c), shared(samples_y)


def train(args):
    print '\nNEURAL POS TAGGER START\n'

    print '\tINITIAL EMBEDDING\t%s' % args.init_emb
    print '\tWORD VECTOR\t\tEmb Dim: %d  Hidden Dim: %d' % (args.emb, args.hidden)
    print '\tCHARACTER VECTOR\tEmb Dim: %d  Hidden Dim: %d' % (args.c_emb, args.c_hidden)
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
    train_corpus, vocab_word, vocab_char, vocab_tag, max_char_len = load_conll(path=args.train_data, _train=True,
                                                                               vocab_word=vocab_word, data_size=argv.data_size, vocab_size=args.vocab)
    dev_corpus, _, _, _, mcl = load_conll(path=args.dev_data, _train=False, vocab_word=vocab_word,
                                          data_size=100000, vocab_char=vocab_char)
    print '\tTrain Sentences: %d  Test Sentences: %d' % (len(train_corpus), len(dev_corpus))
    print '\tVocab size: %d  Char size: %d  Max Char Len: %d' % (vocab_word.size(), vocab_char.size(), max_char_len)

    """ converting into ids """
    print '\tConverting into IDs...'
    # batches: 1D: n_batch, 2D: [0]=word id 2D matrix, [1]=tag id 2D matrix
    tr_sample_x, tr_sample_c, tr_sample_y = convert_into_ids(train_corpus, vocab_word, vocab_char, vocab_tag, max_char_len)
    dev_sample_x, dev_sample_c, dev_sample_y = convert_into_ids(dev_corpus, vocab_word, vocab_char, vocab_tag, max_char_len)

    tr_sample_x, tr_sample_c, tr_sample_b = get_samples(tr_sample_x, tr_sample_c, max_char_len)
    dev_sample_x, dev_sample_c, dev_sample_b = get_samples(dev_sample_x, dev_sample_c, max_char_len)

    tr_sample_x, tr_sample_c, tr_sample_y = shared_samples(tr_sample_x, tr_sample_c, tr_sample_y)
    dev_sample_x, dev_sample_c, dev_sample_y = shared_samples(dev_sample_x, dev_sample_c, dev_sample_y)

    """ symbol definition """
    n_c_emb = args.c_emb
    window = args.window
    opt = args.opt
#    lr = args.lr
    reg = args.reg
    n_h = args.hidden
    n_c_h = args.c_hidden
    n_y = args.tag
    n_chars = max_char_len + (window / 2) * 2

    print '\tCompiling Theano Code...'
    bos = T.iscalar('bos')
    eos = T.iscalar('eos')
    x = T.imatrix('word')
    c = T.itensor3('char')
    y = T.ivector('tag')
    lr = T.fscalar('lr')

    """ tagger set up """
    tagger = Model(x=x, c=c, y=y, opt=opt, lr=lr, init_emb=init_emb,
                   vocab_size=vocab_word.size(), char_size=vocab_char.size(), window=window,
                   n_emb=n_emb, n_c_emb=n_c_emb, n_h=n_h, n_c_h=n_c_h, n_y=n_y)

    train_model = theano.function(
        inputs=[bos, eos, lr],
        outputs=[tagger.nll, tagger.result],
        updates=tagger.updates,
        givens={
            x: tr_sample_x[bos: eos],
            c: tr_sample_c[bos: eos],
            y: tr_sample_y[bos: eos]
        },
        mode='FAST_RUN'
    )

    valid_model = theano.function(
        inputs=[bos, eos],
        outputs=tagger.result,
        givens={
            x: dev_sample_x[bos: eos],
            c: dev_sample_c[bos: eos],
            y: dev_sample_y[bos: eos]
        },
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

            for index in xrange(len(tr_sample_b)):
                if index % 100 == 0 and index != 0:
                    print index,
                    sys.stdout.flush()

                boundary = tr_sample_b[index]
                loss, corrects = train_model(boundary[0], boundary[1], _lr)

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

        for index in xrange(len(dev_sample_b)):
            if index % 100 == 0 and index != 0:
                print index,
                sys.stdout.flush()

            boundary = dev_sample_b[index]
            corrects = model(boundary[0], boundary[1])

            total += len(corrects)
            correct += np.sum(corrects)

        end = time.time()
        print '\tTime: %f seconds' % (end - start)
        print '\tAccuracy:%f  Total:%d  Correct:%d' % ((correct / total), total, correct)

    _train()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train NN tagger.')

    parser.add_argument('-mode', type=str, default='word', help='char/word')
    parser.add_argument('--train_data', help='path to training data')
    parser.add_argument('--dev_data', help='path to development data')
    parser.add_argument('--test_data', help='path to test data')
    parser.add_argument('--data_size', type=int, default=100000)

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

    argv = parser.parse_args()
    train(argv)
