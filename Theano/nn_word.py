__author__ = 'hiroki'

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d

from nn_utils import sample_weights, relu
from optimizers import sgd, ada_grad


class Model(object):
    def __init__(self, x, y, opt, lr, init_emb, vocab_size, window, n_emb, n_c_emb, n_h, n_y, reg=0.0001):
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
        self.y = y

        n_words = x.shape[0]

        """ params """
        if init_emb is not None:
            self.emb = theano.shared(init_emb)
        else:
            self.emb = theano.shared(sample_weights(vocab_size, n_emb))

        self.W_in = theano.shared(sample_weights(n_h, 1, window, n_emb))
        self.W_h = theano.shared(sample_weights(n_h, n_h))
        self.W_out = theano.shared(sample_weights(n_h, n_y))
        self.params = [self.W_in, self.W_h, self.W_out]

        """ pad """
        self.zero = theano.shared(np.zeros(shape=(1, 1, window / 2, n_emb), dtype=theano.config.floatX))

        """ look up embedding """
        self.x_emb = self.emb[self.x]  # x_emb: 1D: n_words, 2D: n_emb

        """ convolution """
        self.x_in = self.conv(self.x_emb)

        """ feed-forward computation """
        self.h = relu(T.dot(self.x_in.reshape((self.x_in.shape[1], self.x_in.shape[2])).T, self.W_h))
        self.p_y_given_x = T.nnet.softmax(T.dot(self.h, self.W_out))

        """ prediction """
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.errors = T.neq(self.y_pred, self.y)

        """ cost function """
        self.nll = -T.sum(T.log(self.p_y_given_x)[T.arange(n_words), self.y])
        self.cost = self.nll

        if opt == 'sgd':
            self.updates = sgd(self.cost, self.params, self.emb, self.x_emb, lr)
        else:
            self.updates = ada_grad(self.cost, self.params, self.emb, self.x_emb, self.x, lr)

    def conv(self, x_emb):
        x_padded = T.concatenate([self.zero, x_emb.reshape((1, 1, x_emb.shape[0], x_emb.shape[1])), self.zero], axis=2)  # x_padded: 1D: n_words + n_pad, 2D: n_phi
        return conv2d(input=x_padded, filters=self.W_in)
