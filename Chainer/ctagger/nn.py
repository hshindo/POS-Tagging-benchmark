import cPickle as pickle

from chainer import Chain
import chainer.functions as F
import chainer.links as L


class NnTagger(Chain):
    """Neural network tagger by dos (Santos and Zadrozny, ICML 2014)."""

    def __init__(self, word_vocab_size=10000, word_emb_dim=100, word_window_size=5, word_init_emb=None, word_hidden_dim=100,
                 use_char=True, char_vocab_size=50, char_emb_dim=10, char_window_size=5, char_init_emb=None, char_hidden_dim=50,
                 tag_num=45):

        assert word_window_size % 2 == 1, 'Word indow size must be odd.'
        if use_char:
            assert char_window_size % 2 == 1, 'Character window size must be odd.'

        if use_char:
            word_dim = word_emb_dim + char_hidden_dim
        else:
            word_dim = word_emb_dim

        super(NnTagger, self).__init__(
                # TODO: make these optional
                char_emb=L.EmbedID(char_vocab_size, char_emb_dim),
                char_conv=L.Convolution2D(1, char_hidden_dim,
                                          ksize=(char_window_size, char_emb_dim),
                                          stride=(1, char_emb_dim),
                                          pad=(char_window_size/2, 0)),

                word_emb=L.EmbedID(word_vocab_size, word_emb_dim),
                word_conv=L.Convolution2D(1, word_hidden_dim,
                                          ksize=(word_window_size, word_dim),
                                          stride=(1, word_dim),
                                          pad=(word_window_size/2, 0)),
                linear=L.Linear(word_hidden_dim, tag_num),
        )

        self.word_vocab_size = word_vocab_size
        self.word_emb_dim = word_emb_dim
        self.word_window_size = word_window_size
        self.word_hidden_dim = word_hidden_dim
        self.use_char = use_char
        self.char_vocab_size = char_vocab_size
        self.char_emb_dim = char_emb_dim
        self.char_window_size = char_window_size
        self.char_hidden_dim = char_hidden_dim
        self.tag_num = tag_num
        self.word_dim = word_dim

        # initialize embeddings
        if word_init_emb is not None:
            self.word_emb.W = word_init_emb
        if use_char and char_init_emb is not None:
            self.char_emb.W = char_init_emb

    def __call__(self, batch):
        word_ids, (char_ids, char_boundaries) = batch
        batch_size = word_ids.data.shape[0]

        # word lookup table
        word_embs = self.word_emb(word_ids)     # batch x len x dim

        if self.use_char:
            # character lookup table
            char_embs = self.char_emb(char_ids)     # total_len x dim
            char_embs_reshape = F.reshape(char_embs, (1, 1, -1, self.char_emb_dim))     # 1 x 1 x total_len x dim

            # convolution
            char_emb_conv = self.char_conv(char_embs_reshape)     # 1 x dim x total_len x 1
            char_emb_conv_reshape = F.reshape(char_emb_conv, (self.char_hidden_dim, -1))     # dim x total_len

            # max
            embs = []
            for i, char_emb_conv_word in enumerate(F.split_axis(char_emb_conv_reshape, char_boundaries, axis=1)):
                if i % 2 == 1:
                    # not pad
                    embs.append(F.max(char_emb_conv_word, axis=1))
            char_emb_conv = F.reshape(F.concat(embs, axis=0), (batch_size, -1, self.char_hidden_dim))

            # concatenate
            word_embs = F.concat([word_embs, char_emb_conv], axis=2)     # batch x len x dim

        word_embs_reshape = F.reshape(word_embs, (batch_size, 1, -1, self.word_dim))

        h = self.word_conv(word_embs_reshape)   # batch x dim x len x 1
        #h_transpose = F.swapaxes(h, 1, 2)  # TODO: maybe inefficient
        h_transpose = F.transpose(h, (0, 2, 1, 3))  # TODO: maybe inefficient
        h_reshape = F.reshape(h_transpose, (-1, self.word_hidden_dim))

        y = self.linear(F.relu(h_reshape))

        return y

    def save(self, path):
        with open(path, 'wb') as f:
            is_cpu = self._cpu
            if is_cpu:
                pickle.dump(self, f)
            else:
                self.to_cpu()
                pickle.dump(self, f)
                self.to_gpu()

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
