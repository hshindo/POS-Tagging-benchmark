__author__ = 'hiroki'

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d

from nn_utils import sample_weights, relu
from optimizers import sgd, ada_grad


class Model(object):
    def __init__(self, x, c, b, y, opt, lr, init_emb, vocab_size, char_size, window, n_emb, n_c_emb, n_h, n_c_h, n_y, reg=0.0001):
        """
        :param n_emb: dimension of word embeddings
        :param window: window size
        :param n_h: dimension of hidden layer
        :param n_y: number of tags
        x: 1D: batch size * window, 2D: emb_dim
        h: 1D: batch_size, 2D: hidden_dim
        """

        assert window % 2 == 1, 'Window size must be odd'

        """ input """
        self.x = x
        self.c = c
        self.b = b
        self.y = y

        n_phi = n_emb + n_c_emb * window
        n_words = x.shape[0]

        """ params """
        if init_emb is not None:
            self.emb = theano.shared(init_emb)
        else:
            self.emb = theano.shared(sample_weights(vocab_size, n_emb))

        self.emb_c = theano.shared(sample_weights(char_size, n_c_emb))
        self.W_in = theano.shared(sample_weights(n_h, 1, window, n_phi))
        self.W_c = theano.shared(sample_weights(n_c_h, 1, window, n_c_emb))
        self.W_h = theano.shared(sample_weights(n_h, n_h))
        self.W_out = theano.shared(sample_weights(n_h, n_y))

        """ pad """
        self.zero = theano.shared(np.zeros(shape=(1, 1, window / 2, n_phi), dtype=theano.config.floatX))
        self.zero_c = theano.shared(np.zeros(shape=(1, 1, window / 2, n_c_emb), dtype=theano.config.floatX))

        self.params = [self.emb_c, self.W_in, self.W_c, self.W_h, self.W_out]

        """ look up embedding """
        x_emb = self.emb[self.x]  # x_emb: 1D: n_words, 2D: n_emb
        c_emb = self.emb_c[self.c]  # c_emb: 1D: n_char of a sent, 2D: n_c_emb

        """ create feature """
        c_phi = self.create_char_feature(self.b, c_emb, self.zero_c)
        x_phi = T.concatenate([x_emb, c_phi], axis=1)

        """ convolution """
        x_padded = T.concatenate([self.zero, x_phi.reshape((1, 1, x_phi.shape[0], x_phi.shape[1])), self.zero], axis=2)  # x_padded: 1D: n_words + n_pad, 2D: n_phi
        x_in = conv2d(input=x_padded, filters=self.W_in)

        """ feed-forward computation """
        h = relu(T.dot(x_in.reshape((x_in.shape[1], x_in.shape[2])).T, self.W_h))
        self.p_y_given_x = T.nnet.softmax(T.dot(h, self.W_out))

        """ prediction """
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.errors = T.neq(self.y_pred, self.y)

        """ cost function """
        self.nll = -T.sum(T.log(self.p_y_given_x)[T.arange(n_words), self.y])

#        self.L2_sqr = (self.emb ** 2).sum()
#        for p in self.params:
#            self.L2_sqr += (p ** 2).sum()

        self.cost = self.nll  # + reg * self.L2_sqr / 2

        if opt == 'sgd':
            self.updates = sgd(self.cost, self.params, self.emb, x_emb, lr)
        else:
            self.updates = ada_grad(self.cost, self.params, self.emb, x_emb, self.x, lr)

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
