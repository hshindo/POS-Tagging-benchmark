from . import nn
from . import util

import sys
import os

import chainer.optimizers as O
import chainer.links as L
from chainer import Variable
from chainer import cuda


def train(args):
    if args.gpu is not None:
        cuda.get_device(args.gpu).use()

    # set up optimizer
    optim_name = args.optim[0]
    assert not args.decay_lr or optim_name == 'SGD', 'learning-rate decay is only supported for SGD'
    optim_args = map(float, args.optim[1:])
    optimizer = getattr(O, optim_name)(*optim_args)

    # load data
    print >> sys.stderr, 'Loading data...'
    corpus, vocab_word, vocab_char, vocab_tag = util.load_conll(args.data, args.vocab)

    # load pre-trained embeddings
    init_emb = None
    if args.init_emb:
        print >> sys.stderr, 'Loading word embeddings...'
        assert args.init_emb_words
        init_emb = util.load_init_emb(args.init_emb, args.init_emb_words, vocab_word)
        emb_dim = init_emb.shape[1]
    else:
        emb_dim = args.word_emb

    # create batches
    print >> sys.stderr, 'Creating batches...'
    batches = util.create_batches(corpus, vocab_word, vocab_char, vocab_tag, args.batch,
                                  args.word_window, args.char_window, gpu=args.gpu, shuffle=not args.no_shuffle)

    # set up tagger
    tagger = nn.NnTagger(
            word_vocab_size=vocab_word.size(), word_emb_dim=emb_dim, word_window_size=args.word_window, word_init_emb=init_emb, word_hidden_dim=args.word_hidden,
            use_char=args.use_char, char_vocab_size=vocab_char.size(), char_emb_dim=args.char_emb, char_window_size=args.char_window, char_hidden_dim=args.char_hidden,
            tag_num=vocab_tag.size())
    classifier = L.Classifier(tagger)

    initial_lr = None
    if args.decay_lr:
        initial_lr = optimizer.lr

    # set up GPU
    if args.gpu is not None:
        classifier.to_gpu()

    optimizer.setup(classifier)

    # create directory
    os.makedirs(args.model)
    vocab_word.save(os.path.join(args.model, 'vocab_word'))
    vocab_char.save(os.path.join(args.model, 'vocab_char'))
    vocab_tag.save(os.path.join(args.model, 'vocab_tag'))

    # training loop
    for n in range(args.epoch):
        # decay learning rate
        if args.decay_lr:
            optimizer.lr = initial_lr / (n + 1)
            print >> sys.stderr, 'Learning rate set to: {}'.format(optimizer.lr)

        for i, ((word_ids_data, (char_ids_data, char_boundaries)), t_data) in enumerate(batches):
            batch_size = word_ids_data.shape[0]

            word_ids = Variable(word_ids_data)
            char_ids = Variable(char_ids_data)
            t = Variable(t_data)
            batch = word_ids, (char_ids, char_boundaries)
            optimizer.update(classifier, batch, t)

            print >> sys.stderr, 'Epoch:{}\tBatch:{}\tloss:{}\tacc:{}\tsize:{}'.format(
                n, i, classifier.loss.data, classifier.accuracy.data, batch_size)

        # save current model
        dest_path = os.path.join(args.model, 'epoch' + str(n))
        tagger.save(dest_path)
