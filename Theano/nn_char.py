import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d

from nn_utils import sample_weights, sample_norm_dist, relu
from optimizers import sgd, ada_grad

theano.config.floatX = 'float32'


class Model(object):
    def __init__(self, w, c, b, y, lr,
                 init_w_emb, vocab_w_size, vocab_c_size,
                 w_emb_dim, c_emb_dim, w_hidden_dim, c_hidden_dim, output_dim,
                 window, opt):

        assert window % 2 == 1, 'Window size must be odd'

        """ input """
        self.w = w
        self.c = c
        self.b = b
        self.y = y
        self.lr = lr
        self.input = [self.w, self.c, self.b, self.y, self.lr]

        n_phi = w_emb_dim + c_emb_dim * window
        n_words = w.shape[0]

        """ params """
        if init_w_emb is not None:
            self.emb = theano.shared(init_w_emb)
        else:
            self.emb = theano.shared(sample_weights(vocab_w_size, w_emb_dim))

        self.emb_c = theano.shared(sample_norm_dist(vocab_c_size, c_emb_dim))
        self.W_in = theano.shared(sample_weights(w_hidden_dim, 1, window, n_phi))
        self.W_c = theano.shared(sample_weights(c_hidden_dim, 1, window, c_emb_dim))
        self.W_out = theano.shared(sample_weights(w_hidden_dim, output_dim))

        self.b_in = theano.shared(sample_weights(w_hidden_dim, 1))
        self.b_c = theano.shared(sample_weights(c_hidden_dim))
        self.b_y = theano.shared(sample_weights(output_dim))

        """ pad """
        self.zero = theano.shared(np.zeros(shape=(1, 1, window / 2, n_phi), dtype=theano.config.floatX))
        self.zero_c = theano.shared(np.zeros(shape=(1, 1, window / 2, c_emb_dim), dtype=theano.config.floatX))

        self.params = [self.emb_c, self.W_in, self.W_c, self.W_out, self.b_in, self.b_c, self.b_y]

        """ look up embedding """
        x_emb = self.emb[self.w]  # x_emb: 1D: n_words, 2D: n_emb
        c_emb = self.emb_c[self.c]  # c_emb: 1D: n_char of a sent, 2D: n_c_emb

        """ create feature """
        c_phi = self.create_char_feature(self.b, c_emb, self.zero_c) + self.b_c
        x_phi = T.concatenate([x_emb, c_phi], axis=1)

        """ convolution """
        x_padded = T.concatenate([self.zero, x_phi.reshape((1, 1, x_phi.shape[0], x_phi.shape[1])), self.zero], axis=2)  # x_padded: 1D: n_words + n_pad, 2D: n_phi
        x_in = conv2d(input=x_padded, filters=self.W_in)

        """ feed-forward computation """
        h = relu(x_in.reshape((x_in.shape[1], x_in.shape[2])) + T.repeat(self.b_in, T.cast(x_in.shape[2], 'int32'), 1)).T
        self.o = T.dot(h, self.W_out) + self.b_y
        self.p_y_given_x = T.nnet.softmax(self.o)

        """ prediction """
        self.y_pred = T.argmax(self.o, axis=1)
        self.result = T.eq(self.y_pred, self.y)

        """ cost function """
        self.nll = -T.sum(T.log(self.p_y_given_x)[T.arange(n_words), self.y])

        self.cost = self.nll

        if opt == 'sgd':
            self.updates = sgd(self.cost, self.params, self.emb, x_emb, self.lr)
        else:
            self.updates = ada_grad(self.cost, self.params, self.emb, x_emb, self.w, self.lr)

    def create_char_feature(self, b, c_emb, zero_c):
        def forward(b_t, b_tmp, c_emb, zero, W):
            c_tmp = c_emb[b_tmp: b_t]
            c_padded = T.concatenate([zero, c_tmp.reshape((1, 1, c_tmp.shape[0], c_tmp.shape[1])), zero], axis=2)
            c_conv = conv2d(input=c_padded, filters=W)  # c_conv: 1D: n_c_h, 2D: n_char * slide
            c_t = T.max(c_conv.reshape((c_conv.shape[1], c_conv.shape[2])), axis=1)  # c_t.shape: (1, 50, b_t-b_tm1, 1)
            return c_t, b_t

        [c, _], _ = theano.scan(fn=forward,
                                sequences=[b],
                                outputs_info=[None, T.cast(0, 'int32')],
                                non_sequences=[c_emb, zero_c, self.W_c])

        return c
