import sys
import time

import numpy as np
import theano

import util


def test(args):
    print '\nNEURAL POS TAGGER START\n'

    print '\tINITIAL EMBEDDING\t%s %s' % (args.word_list, args.emb_list)
    print '\tWORD\t\t\tEmb Dim: %d  Hidden Dim: %d' % (args.w_emb_dim, args.w_hidden_dim)
    print '\tCHARACTER\t\tEmb Dim: %d  Hidden Dim: %d' % (args.c_emb_dim, args.c_hidden_dim)
    print '\tOPTIMIZATION\t\tMethod: %s  Learning Rate: %f\n' % (args.opt, args.lr)

    """ load vocab """
    print 'Loading vocabularies...\n'
    vocab_word = util.load_data('vocab_word')
    vocab_char = util.load_data('vocab_char')
    vocab_tag = util.load_data('vocab_tag')
    print '\tWord size: %d  Char size: %d' % (vocab_word.size(), vocab_char.size())

    """ load data """
    print '\nLoading data set...\n'
    test_corpus, test_vocab_word, test_vocab_char, test_vocab_tag = util.load_conll(args.dev_data)
    print '\tTest Sentences: %d' % len(test_corpus)

    """ converting into ids """
    print '\nConverting into IDs...\n'
    test_x, test_c, test_b, test_y = util.convert_into_ids(test_corpus, vocab_word, vocab_char, vocab_tag)

    """ tagger set up """
    tagger = util.load_data(args.load)

    dev_f = theano.function(
        inputs=tagger.input[:-1],
        outputs=tagger.result,
        mode='FAST_RUN'
    )

    """ Prediction """
    print '\nPREDICTION START\n'

    print '\tBatch Index: ',
    start = time.time()

    total = 0.0
    correct = 0

    for index in xrange(len(test_x)):
        if index % 100 == 0 and index != 0:
            print index,
            sys.stdout.flush()

        if tagger.name == 'char':
            corrects = dev_f(test_x[index], test_c[index], test_b[index], test_y[index])
        else:
            corrects = dev_f(test_x[index], test_y[index])

        total += len(corrects)
        correct += np.sum(corrects)

    end = time.time()

    print '\n\tTime: %f seconds' % (end - start)
    print '\tAccuracy:%f  Total:%d  Correct:%d' % ((correct / total), total, correct)
