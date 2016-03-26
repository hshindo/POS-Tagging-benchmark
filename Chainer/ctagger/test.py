from . import nn
from . import util

import sys
import time
import os

import numpy as np
from chainer import Variable


def test(args):
    print >> sys.stderr, 'Loading model...'
    tagger = nn.NnTagger.load(args.model)

    print >> sys.stderr, 'Loading vocabulary...'
    model_dir_path = os.path.dirname(args.model)
    vocab_word_path = os.path.join(model_dir_path, 'vocab_word')
    vocab_char_path = os.path.join(model_dir_path, 'vocab_char')
    vocab_tag_path = os.path.join(model_dir_path, 'vocab_tag')
    vocab_word = util.Vocab.load(vocab_word_path)
    vocab_char = util.Vocab.load(vocab_char_path)
    vocab_tag = util.Vocab.load(vocab_tag_path)

    print >> sys.stderr, 'Loading data...'
    corpus = util.load_conll(args.data, 0)[0]

    print >> sys.stderr, 'Creating batches...'
    batches = util.create_batches(corpus, vocab_word, vocab_char, vocab_tag, args.batch,
                                  linear_conv=tagger.linear_conv, window_size=tagger.word_window_size,
                                  pad_char=tagger.pad_char, gpu=None, shuffle=True)
    batch_num = len(batches)

    # main loop
    total = 0
    correct = 0
    processed_num = 0
    time_begin = time.time()
    for i, ((word_ids_data, (char_ids_data, char_boundaries)), t_data) in enumerate(batches):
        processed_num += 1

        word_ids = Variable(word_ids_data, volatile='off')
        char_ids = Variable(char_ids_data, volatile='off')
        batch = word_ids, (char_ids, char_boundaries)
        pred = tagger(batch)

        predicted_tags = np.argmax(pred.data, axis=1)
        for t, pred_tag in zip(t_data, predicted_tags):
            if t == pred_tag:
                correct += 1
            total += 1
        print >> sys.stderr, 'Processed {:.2%} [{}/{}]'.format(
            float(processed_num) / batch_num,
            processed_num, batch_num)
    time_end = time.time()

    # report result
    print '{:.2%}'.format(float(correct) / total)
    print correct
    print total
    print 'Time elapsed: {} sec'.format(time_end - time_begin)


