__author__ = 'hiroki'

import theano
import theano.tensor as T

from nn_utils import sample_weights, relu, tanh
from optimizers import sgd, ada_grad


class NnTagger(object):
    def __init__(self, x, y, opt, lr, init_emb, vocab_size=10000, emb_dim=100, window=5, hidden_dim=100, tag_num=45, reg=0.0001):
        """
        :param emb_dim: dimension of word embeddings
        :param window: window size
        :param hidden_dim: dimension of hidden layer
        :param tag_num: number of tags
        x: 1D: batch size * window, 2D: emb_dim
        h: 1D: batch_size, 2D: hidden_dim
        """

        assert window % 2 == 1, 'Window size must be odd'

        self.x = x
        self.x_v = self.x.flatten()
        self.y = y

        batch = x.shape[0]

        """ params """
        if init_emb is not None:
            self.emb = theano.shared(init_emb)
        else:
            self.emb = theano.shared(sample_weights(vocab_size, emb_dim))

        self.W_in = theano.shared(sample_weights(window * emb_dim, hidden_dim))
        self.W_h = theano.shared(sample_weights(hidden_dim, hidden_dim))
        self.W_out = theano.shared(sample_weights(hidden_dim, tag_num))

        self.params = [self.W_in, self.W_h, self.W_out]

        self.x_in = self.emb[self.x_v]
        self.x_in_u = T.dot(self.x_in.reshape((batch, -1)), self.W_in)
        self.h = relu(T.dot(self.x_in_u, self.W_h))
        self.p_y_given_x = T.nnet.softmax(T.dot(self.h, self.W_out))

        self.nll = -T.sum(T.log(self.p_y_given_x)[T.arange(batch), self.y])
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.errors = T.neq(self.y_pred, self.y)

        self.cost = self.nll

        if opt == 'sgd':
            self.updates = sgd(self.cost, self.params, self.emb, self.x_in, lr)
        else:
            self.updates = ada_grad(self.cost, self.params, self.emb, self.x_in, self.x_v, lr)
