from collections import defaultdict
import random
import re

import numpy as np
from chainer import cuda


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


def load_init_emb(init_emb, init_emb_words, vocab):
    """Load embedding file and create vocabulary.

    :return: tuple of embedding numpy array and vocabulary"""

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
            #elif word == u'-lrb-':
            #    word = u'('
            #elif word == u'-rrb-':
            #    word = u')'
            #else:
            #    # TODO: convert numbers appropriately
            #    pass

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


def _expand(w_ids, window_size, eos_id):
    res = np.ones((len(w_ids), window_size), dtype=np.int32) * eos_id
    for i, w_id in enumerate(w_ids):
        for j in range(window_size):
            i2 = window_size // 2 + i - j
            if 0 <= i2 < len(w_ids):
                res[i2][j] = w_id
    return res


def _test_expand():
    w_ids = [3, 5, 2, 1, 8, 7]
    res = _expand(w_ids, 5, 0)
    ref = [
        [0, 0, 3, 5, 2],
        [0, 3, 5, 2, 1],
        [3, 5, 2, 1, 8],
        [5, 2, 1, 8, 7],
        [2, 1, 8, 7, 0],
        [1, 8, 7, 0, 0],
    ]
    assert np.array_equal(np.asarray(res), np.asarray(ref))


def create_batches(corpus, vocab_word, vocab_char, vocab_tag, batch_size, linear_conv, window_size, pad_char, gpu, shuffle):
    # convert to IDs
    id_corpus = []
    for sen in corpus:
        w_ids = []
        t_ids = []
        c_ids = []
        for w, t in sen:
            w_id = vocab_word.get_id(w.lower())
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

    if shuffle:
        random.shuffle(id_corpus)

    # sort by sentence lengths
    id_corpus.sort(key=lambda w_t: len(w_t[0]))

    # split into batches
    word_batches = split_into_batches(id_corpus, batch_size)

    # character IDs
    batches = []
    for word_batch in word_batches:
        word_ids, tag_ids, char_ids = zip(*word_batch)

        if linear_conv:
            eos_id = vocab_word.get_id(EOS)
            word_ids = map(lambda ids: _expand(ids, window_size, eos_id), word_ids)

        char_boundaries = []
        if pad_char:
            max_word_len = -1
            sen_len = len(word_ids[0])
            for char_id_lists in char_ids:
                for c_ids in char_id_lists:
                    max_word_len = max(max_word_len, len(c_ids))
            eos_id = vocab_char.get_id(EOS)     # padded with EOS
            char_batch = np.ones((len(word_ids), sen_len, max_word_len)) * eos_id  # 3D array
            for i, char_id_lists in enumerate(char_ids):
                for j, c_ids in enumerate(char_id_lists):
                    char_batch[i, j, :len(c_ids)] = c_ids
        else:
            char_batch = []
            i = 0
            for char_id_lists in char_ids:
                for c_ids in char_id_lists:
                    i += len(c_ids)
                    char_boundaries.append(i)
                    char_batch.extend(c_ids)
            char_boundaries.pop()

        word_ids_data = np.asarray(word_ids, dtype=np.int32)
        char_ids_data = np.asarray(char_batch, dtype=np.int32)
        tag_ids_data = np.asarray(tag_ids, dtype=np.int32).flatten()

        if gpu is not None:
            word_ids_data = cuda.to_gpu(word_ids_data)
            char_ids_data = cuda.to_gpu(char_ids_data)
            tag_ids_data = cuda.to_gpu(tag_ids_data)

        batches.append(((word_ids_data, (char_ids_data, char_boundaries)), tag_ids_data))

    return batches


if __name__ == '__main__':
    _test_expand()
