from . import nn
from . import util

import os
import logging
import time

import chainer.optimizers as O
import chainer.functions as F
import chainer.links as L
from chainer import Variable
from chainer import cuda


def _log_str(lst):
    s = []
    for k, v in lst:
        s.append(k + ':')
        if isinstance(v, float):
            v_str = '{:.6f}'.format(v)
        else:
            v_str = str(v)
        s.append(v_str)
    return '\t'.join(s)


def _softmax_cross_entropy_no_normalize(y, t):
    return F.softmax_cross_entropy(y, t) * y.data.shape[0]


def train(args):
    if args.gpu is not None:
        cuda.get_device(args.gpu).use()

    os.makedirs(args.model)

    # set up logger
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    log_path = os.path.join(args.model, 'log')
    file_handler = logging.FileHandler(log_path)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # set up optimizer
    optim_name = args.optim[0]
    assert not args.decay_lr or optim_name == 'SGD', 'learning-rate decay is only supported for SGD'
    optim_args = map(float, args.optim[1:])
    optimizer = getattr(O, optim_name)(*optim_args)

    # load data
    logger.info('Loading word list...')
    ws = set()
    with open(args.words) as f:
        for line in f:
            w = line.rstrip().decode('utf-8')
            ws.add(w)
    logger.info('Loading training data...')
    corpus, vocab_word, vocab_char, vocab_tag = util.load_conll(args.data, args.vocab_size, limit_vocab=ws)
    if args.test:
        logger.info('Loading test data...')
        corpus_test, vocab_word_test, vocab_char_test, vocab_tag_test = util.load_conll(args.test, None, limit_vocab=ws)
        for w in vocab_word_test.i2w:
            if args.vocab_size is None or vocab_word.size() < args.vocab_size:
                vocab_word.add_word(w)
        for c in vocab_char_test.i2w:
            vocab_char.add_word(c)
        for t in vocab_tag_test.i2w:
            vocab_tag.add_word(t)

    # load pre-trained embeddings
    init_emb = None
    if args.init_emb:
        logger.info('Loading word embeddings...')
        init_emb = util.load_init_emb(args.init_emb, args.words, vocab_word)
        emb_dim = init_emb.shape[1]
    else:
        emb_dim = args.word_emb

    # create batches
    logger.info('Creating batches...')
    batches = util.create_batches(corpus, vocab_word, vocab_char, vocab_tag, args.batch,
                                  gpu=args.gpu, shuffle=not args.no_shuffle)

    # set up tagger
    tagger = nn.NnTagger(
            word_vocab_size=vocab_word.size(), word_emb_dim=emb_dim, word_window_size=args.word_window, word_init_emb=init_emb, word_hidden_dim=args.word_hidden,
            use_char=args.use_char, char_vocab_size=vocab_char.size(), char_emb_dim=args.char_emb, char_window_size=args.char_window, char_hidden_dim=args.char_hidden,
            tag_num=vocab_tag.size())
    classifier = L.Classifier(tagger, lossfun=_softmax_cross_entropy_no_normalize)

    initial_lr = None
    if args.decay_lr:
        initial_lr = optimizer.lr

    # set up GPU
    if args.gpu is not None:
        classifier.to_gpu()

    optimizer.setup(classifier)

    # create directory
    vocab_word.save(os.path.join(args.model, 'vocab_word'))
    vocab_char.save(os.path.join(args.model, 'vocab_char'))
    vocab_tag.save(os.path.join(args.model, 'vocab_tag'))

    # training loop
    for n in range(args.epoch):
        # decay learning rate
        if args.decay_lr:
            optimizer.lr = initial_lr / (n + 1)
            logger.info('Learning rate set to: {}'.format(optimizer.lr))

        for i, ((word_ids_data, (char_ids_data, char_boundaries)), t_data) in enumerate(batches):
            batch_size, batch_length = word_ids_data.shape

            time_start = time.time()
            word_ids = Variable(word_ids_data)
            char_ids = Variable(char_ids_data)
            t = Variable(t_data)
            batch = word_ids, (char_ids, char_boundaries)
            optimizer.update(classifier, batch, t)
            time_end = time.time()
            time_delta = time_end - time_start

            logger.info(_log_str([
                ('epoch', n),
                ('batch', i),
                ('loss', float(classifier.loss.data)),
                ('acc', float(classifier.accuracy.data)),
                ('size', batch_size),
                ('len', batch_length),
                ('time', int(time_delta * 1000)),
            ]))

        # save current model
        dest_path = os.path.join(args.model, 'epoch' + str(n))
        tagger.save(dest_path)
