__author__ = 'hiroki'

from collections import defaultdict
import re
import random
import gzip
import cPickle

import numpy as np
import theano

random.seed(0)

PAD = u'<PAD>'
EOS = u'<EOS>'
UNK = u'<UNK>'

RE_NUM = re.compile(ur'[0-9]')


class Vocab(object):
    """Mapping between words and IDs."""

    def __init__(self):
        self.i2w = []
        self.w2i = {}

    def add_word(self, word):
        assert isinstance(word, unicode)
        if word not in self.w2i:
            new_id = self.size()
            self.i2w.append(word)
            self.w2i[word] = new_id

    def get_id(self, word):
        assert isinstance(word, unicode)
        return self.w2i.get(word)

    def get_word(self, w_id):
        return self.i2w[w_id]

    def size(self):
        return len(self.i2w)

    def save(self, path):
        with open(path, 'w') as f:
            for i, w in enumerate(self.i2w):
                print >> f, str(i) + '\t' + w.encode('utf-8')

    @classmethod
    def load(cls, path):
        vocab = Vocab()
        with open(path) as f:
            for line in f:
                w = line.strip().split('\t')[1].decode('utf-8')
                vocab.add_word(w)
        return vocab


def load_init_emb(init_emb, init_emb_words, vocab):
    unk_id = vocab.get_id(UNK)

    # read first line and get dimension
    with open(init_emb) as f_emb:
        line = f_emb.readline()
        dim = len(line.split())
    assert dim > 0

    # initialize embeddings
    emb = np.random.randn(vocab.size(), dim).astype(np.float32)

    # corresponding IDs in given vocabulary
    ids = []

    with open(init_emb_words) as f_words:
        for i, line in enumerate(f_words):
            word = line.strip().decode('utf-8')

            # convert special characters
            if word == u'PADDING':
                word = EOS
            elif word == u'UNKNOWN':
                word = UNK

            w_id = vocab.get_id(word)

            # don't map unknown words to <UNK> unless it's really UNKNOWN
            if w_id == unk_id:
                if word == UNK:
                    ids.append(unk_id)
                else:
                    # no corresponding word in vocabulary
                    ids.append(None)
            else:
                ids.append(w_id)

    with open(init_emb) as f_emb:
        for i, emb_str in enumerate(f_emb):
            w_id = ids[i]
            if w_id is not None:
                emb[w_id] = emb_str.split()

    return emb


def load_conll(path, vocab_size=None, file_encoding='utf-8', limit_vocab=None):
    """Load CoNLL-format file.
    :return tuple of corpus (pairs of words and tags), word vocabulary, char vocabulary, tag vocabulary"""

    corpus = []
    word_freqs = defaultdict(int)
    char_freqs = defaultdict(int)

    vocab_word = Vocab()
    vocab_char = Vocab()
    vocab_tag = Vocab()
    vocab_word.add_word(EOS)
    vocab_word.add_word(UNK)
    vocab_char.add_word(EOS)
    #vocab_char.add_word(UNK)

    with open(path) as f:
        wts = []
        for line in f:
            es = line.rstrip().split('\t')
            if len(es) == 10:
                word = es[1].decode(file_encoding)
                # replace numbers with 0
                word = RE_NUM.sub(u'0', word)
                tag = es[4].decode(file_encoding)
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
        if wts:
            corpus.append(wts)

    for w, f in sorted(word_freqs.items(), key=lambda (k, v): -v):
        if limit_vocab is None or w in limit_vocab:
            # register only words in limit_vocab
            if vocab_size is None or vocab_word.size() < vocab_size:
                vocab_word.add_word(w)
            else:
                break

    for c, f in sorted(char_freqs.items(), key=lambda (k, v): -v):
        vocab_char.add_word(c)

    return corpus, vocab_word, vocab_char, vocab_tag


def load_conll_char(path, vocab_size=None, file_encoding='utf-8', limit_vocab=None):
    """Load CoNLL-format file.
    :return tuple of corpus (pairs of words and tags), word vocabulary, char vocabulary, tag vocabulary"""

    corpus = []
    word_freqs = defaultdict(int)
    char_freqs = defaultdict(int)
    max_char_len = -1

    vocab_word = Vocab()
    vocab_char = Vocab()
    vocab_tag = Vocab()
    vocab_word.add_word(EOS)
    vocab_word.add_word(UNK)
    vocab_char.add_word(EOS)

    with open(path) as f:
        wts = []
        for line in f:
            es = line.rstrip().split('\t')
            if len(es) == 10:
                word = es[1].decode(file_encoding)
                # replace numbers with 0
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
        if wts:
            corpus.append(wts)

    for w, f in sorted(word_freqs.items(), key=lambda (k, v): -v):
        if limit_vocab is None or w in limit_vocab:
            # register only words in limit_vocab
            if vocab_size is None or vocab_word.size() < vocab_size:
                vocab_word.add_word(w)
            else:
                break

    for c, f in sorted(char_freqs.items(), key=lambda (k, v): -v):
        vocab_char.add_word(c)

    return corpus, vocab_word, vocab_char, vocab_tag, max_char_len


def convert_into_ids(corpus, vocab_word, vocab_char, vocab_tag):
    id_corpus_w = []
    id_corpus_c = []
    id_corpus_b = []
    id_corpus_t = []

    for sent in corpus:
        w_ids = []
        c_ids = []
        bs = []
        t_ids = []
        b = 0
        for w, t in sent:
            w_id = vocab_word.get_id(w.lower())
            t_id = vocab_tag.get_id(t)

            if w_id is None:
                w_id = vocab_word.get_id(UNK)

            assert w_id is not None
            assert t_id is not None

            w_ids.append(w_id)
            t_ids.append(t_id)
            c_ids.extend([vocab_char.get_id(c) for c in w])
            b += len(w)
            bs.append(b)

        id_corpus_w.append(np.asarray(w_ids, dtype='int32'))
        id_corpus_c.append(np.asarray(c_ids, dtype='int32'))
        id_corpus_b.append(np.asarray(bs, dtype='int32'))
        id_corpus_t.append(np.asarray(t_ids, dtype='int32'))

    assert len(id_corpus_w) == len(id_corpus_c) == len(id_corpus_b) == len(id_corpus_t)
    return id_corpus_w, id_corpus_c, id_corpus_b, id_corpus_t


def convert_into_ids_batch(corpus, vocab_word, vocab_char, vocab_tag):
    id_corpus_w = []
    id_corpus_c = []
    id_corpus_b = []
    id_corpus_t = []

    for sent in corpus:
        w_ids = []
        c_ids = []
        bs = []
        t_ids = []
        b = 0
        for w, t in sent:
            w_id = vocab_word.get_id(w.lower())
            t_id = vocab_tag.get_id(t)

            if w_id is None:
                w_id = vocab_word.get_id(UNK)

            assert w_id is not None
            assert t_id is not None

            w_ids.append(w_id)
            t_ids.append(t_id)
            c_ids.extend([vocab_char.get_id(c) for c in w])
            b += len(w)
            bs.append(b)

        id_corpus_w.append(w_ids)
        id_corpus_c.append(c_ids)
        id_corpus_b.append(bs)
        id_corpus_t.append(t_ids)

    assert len(id_corpus_w) == len(id_corpus_c) == len(id_corpus_b) == len(id_corpus_t)
    return id_corpus_w, id_corpus_c, id_corpus_b, id_corpus_t


def split_into_batches(corpus, batch_size, length_func=lambda t: len(t[0])):
    batches = []
    batch = []
    last_len = 0
    for sen in corpus:
        current_len = length_func(sen)
        if (last_len != current_len and len(batch) > 0) or len(batch) == batch_size:
            # next batch
            batches.append(batch)
            batch = []
        last_len = current_len
        batch.append(sen)
    if batch:
        batches.append(batch)
    return batches


def create_batches(corpus, vocab_word, vocab_char, vocab_tag, batch_size, char_window, shuffle):
    char_padding_size = char_window / 2
    char_padding = [vocab_char.get_id(EOS)] * char_padding_size

    # convert to IDs
    id_corpus = []
    for sen in corpus:
        w_ids = []
        t_ids = []
        c_ids = []
        for w, t in sen:
            w_id = vocab_word.get_id(w)
            t_id = vocab_tag.get_id(t)
            if w_id is None:
                w_id = vocab_word.get_id(UNK)
            assert w_id is not None
            if t_id is None:
                # ID for unknown tag
                t_id = -1
            w_ids.append(w_id)
            t_ids.append(t_id)
            c_ids.append([vocab_char.get_id(c) for c in w])
        id_corpus.append((w_ids, t_ids, c_ids))

    # sort by lengths
    id_corpus.sort(key=lambda w_t: len(w_t[0]))

    # split into batches
    word_batches = split_into_batches(id_corpus, batch_size)

    # shuffle batches
    if shuffle:
        random.shuffle(word_batches)

    # character IDs
    batches = []
    for word_batch in word_batches:
        word_ids, tag_ids, char_ids = zip(*word_batch)

        char_boundaries = []
        char_batch = []
        i = 0
        char_batch.extend(char_padding)

        for word_id_list, char_id_lists in zip(word_ids, char_ids):
            for w_id, c_ids in zip(word_id_list, char_id_lists):
                char_batch.extend(c_ids)
                char_batch.extend(char_padding)
                i += char_padding_size
                char_boundaries.append(i)
                i += len(c_ids)
                char_boundaries.append(i)

        word_ids_data = np.asarray(word_ids, dtype=np.int32)
        char_ids_data = np.asarray(char_batch, dtype=np.int32)
        tag_ids_data = np.asarray(tag_ids, dtype=np.int32).flatten()

        batches.append(((word_ids_data, (char_ids_data, char_boundaries)), tag_ids_data))

    return batches


def create_samples(data, window=5):
    """
    :param data: 1D: n_sents, 2D: n_words; elem=word id
    :return: samples: 1D: n_samples, 2D: window; elem=window feature
    """
    def extract_one_sample(w_i):
        if sent_len < window:
            sample = [0] * (p - w_i)
            if w_i > p:
                sample.extend(sent[w_i - p:w_i + p + 1])
            else:
                sample.extend(sent[:w_i + p + 1])
            sample.extend([0] * (window - len(sample)))
        elif w_i < p:
            sample = [0] * (p - w_i)
            sample.extend(sent[: w_i + p + 1])
        elif w_i > end:
            sample = sent[w_i - p:]
            sample.extend([0] * (window - len(sample)))
        else:
            sample = sent[w_i - p: w_i + p + 1]
        return sample

    samples = []
    sent_boundary = []
    s_i = 0
    p = window / 2

    for sent in data:
        sent_len = len(sent)
        end = sent_len - p - 1
        samples.extend([extract_one_sample(i) for i in xrange(sent_len)])
        sent_boundary.append((s_i, s_i + sent_len))
        s_i += sent_len

    return np.asarray(samples, dtype=np.int32), np.asarray(sent_boundary, dtype=np.int32)


def flatten(data):
    return [t for sent in data for t in sent]


def shared_data(x, y):
    def shared(samples):
        return theano.shared(samples, borrow=True)

    assert x.shape[0] == len(y)

    x = shared(x)
    y = shared(np.asarray(y, dtype=np.int32))
    return x, y


def dump_data(data, fn):
    with gzip.open(fn + '.pkl.gz', 'wb') as gf:
        cPickle.dump(data, gf, cPickle.HIGHEST_PROTOCOL)


def load_data(fn):
    if fn[-7:] != '.pkl.gz':
        fn += '.pkl.gz'

    with gzip.open(fn, 'rb') as gf:
        return cPickle.load(gf)

