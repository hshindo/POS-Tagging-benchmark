import sys
import time
import math
import random

import numpy as np
import theano
import theano.tensor as T

import util
import nn_word
import nn_char


def train(args):
    print '\nNEURAL POS TAGGER START\n'

    print '\tINITIAL EMBEDDING\t%s %s' % (args.word_list, args.emb_list)
    print '\tWORD\t\t\tEmb Dim: %d  Hidden Dim: %d' % (args.w_emb_dim, args.w_hidden_dim)
    print '\tCHARACTER\t\tEmb Dim: %d  Hidden Dim: %d' % (args.c_emb_dim, args.c_hidden_dim)
    print '\tOPTIMIZATION\t\tMethod: %s  Learning Rate: %f\n' % (args.opt, args.lr)

    """ load data """
    print 'Loading data sets...\n'
    train_corpus, vocab_word, vocab_char, vocab_tag = util.load_conll(args.train_data)

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
        init_w_emb = util.load_init_emb(args.emb_list, args.vocab_list, vocab_word)
        w_emb_dim = init_w_emb.shape[1]
    else:
        w_emb_dim = args.w_emb_dim

    """ limit data set """
    train_corpus = train_corpus[:args.data_size]

    """ converting into ids """
    print '\nConverting into IDs...\n'
    # batches: 1D: n_batch, 2D: [0]=word id 2D matrix, [1]=tag id 2D matrix
    tr_x, tr_c, tr_b, tr_y = util.convert_into_ids(train_corpus, vocab_word, vocab_char, vocab_tag)
    if args.dev_data:
        dev_x, dev_c, dev_b, dev_y = util.convert_into_ids(dev_corpus, vocab_word, vocab_char, vocab_tag)
        print '\tTrain Sentences: %d  Dev Sentences: %d' % (len(train_corpus), len(dev_corpus))
    else:
        print '\tTrain Sentences: %d' % len(train_corpus)

    print '\tWord size: %d  Char size: %d' % (vocab_word.size(), vocab_char.size())

    """ tagger set up """
    tagger = set_model(args, init_w_emb, w_emb_dim, vocab_word, vocab_char, vocab_tag)

    train_f = theano.function(
        inputs=tagger.input,
        outputs=[tagger.nll, tagger.result],
        updates=tagger.updates,
        mode='FAST_RUN'
    )

    dev_f = theano.function(
        inputs=tagger.input[:-1],
        outputs=tagger.result,
        mode='FAST_RUN'
    )

    def _train():
        print '\nTRAINING START\n'

        for epoch in xrange(args.epoch):
            _lr = args.lr / float(epoch+1)
            indices = range(len(tr_x))
            random.shuffle(indices)

            print '\nEpoch: %d' % (epoch + 1)
            print '\n\tTrain set'
            print '\t\tBatch Index: ',
            start = time.time()

            total = 0.0
            correct = 0
            losses = 0.0

            for i, index in enumerate(indices):
                if i % 100 == 0 and i != 0:
                    print i,
                    sys.stdout.flush()

                if args.model == 'char':
                    loss, corrects = train_f(tr_x[index], tr_c[index], tr_b[index], tr_y[index], _lr)
                else:
                    loss, corrects = train_f(tr_x[index], tr_y[index], _lr)

                assert math.isnan(loss) is False, index

                total += len(corrects)
                correct += np.sum(corrects)
                losses += loss

            end = time.time()
            print '\n\t\tTime: %f seconds' % (end - start)
            print '\t\tNegative Log Likelihood: %f' % losses
            print '\t\tAccuracy:%f  Total:%d  Correct:%d' % ((correct / total), total, correct)

            if args.save:
                util.dump_data(tagger, 'model-%s.epoch-%d' % (args.model, epoch+1))

            _dev(dev_f)

    def _dev(_dev_f):
        print '\n\tDev set'
        print '\t\tBatch Index: ',
        start = time.time()

        total = 0.0
        correct = 0

        for index in xrange(len(dev_x)):
            if index % 100 == 0 and index != 0:
                print index,
                sys.stdout.flush()

            if args.model == 'char':
                corrects = _dev_f(dev_x[index], dev_c[index], dev_b[index], dev_y[index])
            else:
                corrects = _dev_f(dev_x[index], dev_y[index])

            total += len(corrects)
            correct += np.sum(corrects)

        end = time.time()

        print '\n\t\tTime: %f seconds' % (end - start)
        print '\t\tAccuracy:%f  Total:%d  Correct:%d' % ((correct / total), total, correct)

    _train()


def set_model(args, init_w_emb, w_emb_dim, vocab_word, vocab_char, vocab_tag):
    print '\nBuilding a neural model: %s\n' % args.model

    """ neural architecture parameters """
    c_emb_dim = args.c_emb_dim
    w_hidden_dim = args.w_hidden_dim
    c_hidden_dim = args.c_hidden_dim
    output_dim = vocab_tag.size()
    window = args.window
    opt = args.opt

    """ symbol definition """
    x = T.ivector()
    c = T.ivector()
    b = T.ivector()
    y = T.ivector()
    lr = T.fscalar('lr')

    if args.model == 'char':
        return nn_char.Model(name=args.model, w=x, c=c, b=b, y=y, lr=lr,
                             init_w_emb=init_w_emb, vocab_w_size=vocab_word.size(), vocab_c_size=vocab_char.size(),
                             w_emb_dim=w_emb_dim, c_emb_dim=c_emb_dim, w_hidden_dim=w_hidden_dim,
                             c_hidden_dim=c_hidden_dim, output_dim=output_dim,
                             window=window, opt=opt)
    else:
        return nn_word.Model(name=args.model, x=x, y=y, lr=lr,
                             init_emb=init_w_emb, vocab_size=vocab_word.size(),
                             emb_dim=w_emb_dim, hidden_dim=w_hidden_dim, output_dim=output_dim,
                             window=window, opt=opt)

