__author__ = 'hiroki'

import sys
import time
import math

from util import load_conll, load_init_emb, convert_words_into_ids, create_samples, flatten, shared_data, load_conll_char, convert_into_ids
import nn
import nn_char
import nn_word

import theano
import theano.tensor as T
import numpy as np


def train_char(args):
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
    train_corpus, vocab_word, vocab_char, vocab_tag = load_conll_char(path=args.train_data, train=True, vocab_word=vocab_word, vocab_size=args.vocab)
    dev_corpus, _, _, _ = load_conll_char(path=args.dev_data, train=False, vocab_word=vocab_word, vocab_char=vocab_char)
    print '\tTrain Sentences: %d  Test Sentences: %d' % (len(train_corpus), len(dev_corpus))
    print '\tVocab size: %d  Char size: %d' % (vocab_word.size(), vocab_char.size())

    """ converting into ids """
    print '\tConverting into IDs...'
    # batches: 1D: n_batch, 2D: [0]=word id 2D matrix, [1]=tag id 2D matrix
    tr_sample_x, tr_sample_c, tr_sample_b, tr_sample_y = convert_into_ids(train_corpus, vocab_word, vocab_char, vocab_tag)
    dev_sample_x, dev_sample_c, dev_sample_b, dev_sample_y = convert_into_ids(dev_corpus, vocab_word, vocab_char, vocab_tag)

    """ symbol definition """
    n_c_emb = args.c_emb
    window = args.window
    opt = args.opt
    lr = args.lr
    reg = args.reg
    n_h = args.hidden
    n_c_h = args.c_hidden
    n_y = args.tag

    print '\tCompiling Theano Code...'
    x = T.ivector()
    c = T.ivector()
    b = T.ivector()
    y = T.ivector()

    """ tagger set up """
    if args.mode == 'char':
        tagger = nn_char.Model(x=x, c=c, b=b, y=y, opt=opt, lr=lr, init_emb=init_emb,
                               vocab_size=vocab_word.size(), char_size=vocab_char.size(), window=window,
                               n_emb=n_emb, n_c_emb=n_c_emb, n_h=n_h, n_c_h=n_c_h, n_y=n_y, reg=reg)
        train_model = theano.function(
            inputs=[x, c, b, y],
            outputs=tagger.nll,
            updates=tagger.updates,
            mode='FAST_RUN'
        )

        valid_model = theano.function(
            inputs=[x, c, b, y],
            outputs=tagger.errors,
            mode='FAST_RUN'
        )
    else:
        tagger = nn_word.Model(x=x, y=y, opt=opt, lr=lr, init_emb=init_emb,
                               vocab_size=vocab_word.size(), window=window,
                               n_emb=n_emb, n_c_emb=n_c_emb, n_h=n_h, n_y=n_y, reg=reg)
        if args.check is False:
            train_model = theano.function(
                inputs=[x, y],
                outputs=tagger.nll,
                updates=tagger.updates,
                mode='FAST_RUN'
            )

            valid_model = theano.function(
                inputs=[x, y],
                outputs=tagger.errors,
                mode='FAST_RUN'
            )

    def _train():
        for epoch in xrange(args.epoch):
            print '\nEpoch: %d' % (epoch + 1)
            print '\tBatch Index: ',
            start = time.time()

            losses = []
            for index in xrange(len(tr_sample_x)):
                if index % 100 == 0 and index != 0:
                    print index,
                    sys.stdout.flush()

                if args.mode == 'char':
                    loss = train_model(tr_sample_x[index], tr_sample_c[index], tr_sample_b[index], tr_sample_y[index])
                else:
                    loss = train_model(tr_sample_x[index], tr_sample_y[index])

                assert math.isnan(loss) is False, index

                losses.append(loss)

            avg_loss = np.mean(losses)
            end = time.time()
            print '\tTime: %f seconds' % (end - start)
            print '\tAverage Negative Log Likelihood: %f' % avg_loss

            _dev(valid_model)

    def _dev(model):
        print '\tBatch Index: ',
        start = time.time()

        errors = []
        for index in xrange(len(dev_sample_x)):
            if index % 100 == 0 and index != 0:
                print index,
                sys.stdout.flush()

            if args.mode == 'char':
                error = model(dev_sample_x[index], dev_sample_c[index], dev_sample_b[index], dev_sample_y[index])
            else:
                error = model(dev_sample_x[index], dev_sample_y[index])

            errors.append(error)

        end = time.time()

        total = 0.0
        correct = 0
        for sent in errors:
            total += len(sent)
            for y_pred in sent:
                if y_pred == 0:
                    correct += 1
        print '\tTime: %f seconds' % (end - start)
        print '\tTest Accuracy:%f  Total:%d  Correct:%d' % ((correct / total), total, correct)

    def check():
        print '\nCHECK'

        model = theano.function(
            inputs=[x, y],
            outputs=[tagger.x_emb, tagger.x_in, tagger.nll],
            updates=tagger.updates,
            mode='FAST_RUN'
        )

        print '\tBatch Index: ',
        start = time.time()

        for index in xrange(len(dev_sample_x)):
            if index % 100 == 0 and index != 0:
                print index,
                sys.stdout.flush()
                result = model(tr_sample_x[index], tr_sample_y[index])

                for r in result:
                    print r
                    print
                exit()

        end = time.time()
        print '\tTime: %f seconds' % (end - start)

    if args.check:
        check()
    else:
        _train()


def train(args):
    print '\nNEURAL POS TAGGER START\n'

    print '\tMODE\t%s' % args.mode
    print '\tINITIAL EMBEDDING\t%s' % args.init_emb
    print '\tWORD VECTOR\t\tEmb Dim: %d  Hidden Dim: %d' % (args.emb, args.hidden)
    print '\tCHARACTER VECTOR\tEmb Dim: %d  Hidden Dim: %d' % (args.c_emb, args.c_hidden)
    print '\tOPTIMIZATION\t\tMethod: %s  Learning Rate: %f  L2 Reg: %f' % (args.opt, args.lr, args.reg)

    """ load pre-trained embeddings """
    vocab_word = None
    init_emb = None
    if args.init_emb:
        print >> sys.stderr, 'Loading Embeddings...'
        init_emb, vocab_word = load_init_emb(args.init_emb)
        emb_dim = init_emb.shape[1]
    else:
        emb_dim = args.emb

    """ load data """
    print 'Loading Data...'
    train_corpus, vocab_word_tmp, vocab_tag = load_conll(path=args.train_data, vocab_size=args.vocab)
    dev_corpus, _, _ = load_conll(path=args.dev_data, train=True)

    if vocab_word is None:
        vocab_word = vocab_word_tmp

    """ converting into ids """
    print 'Converting into IDs...'
    # batches: 1D: n_batch, 2D: [0]=word id 2D matrix, [1]=tag id 2D matrix
    tr_id_x, tr_id_y = convert_words_into_ids(train_corpus, vocab_word, vocab_tag)
    dev_id_x, dev_id_y = convert_words_into_ids(dev_corpus, vocab_word, vocab_tag)

    """ creating samples """
    print 'Creating Samples...'
    # batches: 1D: n_batch, 2D: [0]=word id 2D matrix, [1]=tag id 2D matrix
    tr_sample_x, tr_sample_b = create_samples(tr_id_x)
    tr_sample_y = flatten(tr_id_y)
    dev_sample_x, dev_sample_b = create_samples(dev_id_x)
    dev_sample_y = flatten(dev_id_y)

    """ converting into ids """
    print 'Creating Batches...'
    train_sample_x, train_sample_y = shared_data(tr_sample_x, tr_sample_y)
    dev_sample_x, dev_sample_y = shared_data(dev_sample_x, dev_sample_y)

    """ symbol definition """
    batch_size = args.batch
    window = args.window
    opt = args.opt
    lr = args.lr
    reg = args.reg

    w = T.imatrix()
    t = T.ivector()

    """ tagger set up """
    tagger = nn.NnTagger(x=w, y=t, opt=opt, lr=lr, init_emb=init_emb, vocab_size=vocab_word.size(),
                         emb_dim=emb_dim, window=window, hidden_dim=args.hidden, tag_num=args.tag, reg=reg)

    def _train():
        bos = T.iscalar()
        eos = T.iscalar()
        train_model = theano.function(
            inputs=[bos, eos],
            outputs=tagger.nll,
            updates=tagger.updates,
            givens={
                w: train_sample_x[bos: eos],
                t: train_sample_y[bos: eos]
            },
            mode='FAST_RUN'
        )

        valid_model = theano.function(
            inputs=[bos, eos],
            outputs=tagger.errors,
            givens={
                w: dev_sample_x[bos: eos],
                t: dev_sample_y[bos: eos]
            },
            mode='FAST_RUN'
        )

        n_train_batches = train_sample_x.get_value().shape[0] / batch_size
        print 'Training Batch Samples: %d' % n_train_batches
        print 'Parameter Optimization Method: %s' % args.opt
        print 'Vocabulary Size: %d' % vocab_word.size()

        for epoch in xrange(args.epoch):
            print '\nEpoch: %d' % (epoch + 1)
            print '\tBatch Index: ',
            start = time.time()

            losses = []
            for b_index in xrange(n_train_batches + 1):
                if b_index % 100 == 0 and b_index != 0:
                    print b_index,
                    sys.stdout.flush()
                b = tr_sample_b[b_index]
                loss = train_model(b[0], b[1])
                if math.isnan(loss):
                    print >> sys.stderr, 'Loss is nan, Batch Index: %d' % b_index
                    exit()
                losses.append(loss)
            avg_loss = np.mean(losses)
            end = time.time()
            print '\tTime: %f seconds' % (end - start)
            print '\tAverage Negative Log Likelihood: %f' % avg_loss

            _dev(valid_model)

    def _dev(_valid_model):
        n_dev_batches = dev_sample_x.get_value().shape[0] / batch_size / args.window
        print '\tBatch Index: ',
        errors = []
        for b_index in xrange(n_dev_batches + 1):
            if b_index % 100 == 0 and b_index != 0:
                print b_index,
                sys.stdout.flush()
            b = dev_sample_b[b_index]
            error = _valid_model(b[0], b[1])
            errors.append(error)

        total = 0.0
        correct = 0
        for sent in errors:
            total += len(sent)
            for y_pred in sent:
                if y_pred == 0:
                    correct += 1
        print '\tTest Accuracy: %f' % (correct / total)

    _train()

