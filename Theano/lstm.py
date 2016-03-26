__author__ = 'hiroki'


import numpy as np
import theano
import theano.tensor as T

from nn_utils import sigmoid, tanh, sample_weights


class LSTM(object):
    def __init__(self,
                 w,
                 d,
                 n_layer,
                 vocab_size,
                 n_in,
                 n_hidden = 50,
                 n_i = 50,
                 n_c = 50,
                 n_o = 50,
                 n_f = 50,
                 n_y = 45,
                 activation=tanh):

        self.w = w
        self.d = d
        self.activation = activation

        """embeddings"""
        self.emb  = theano.shared(sample_weights(vocab_size, n_in))

        """input gate parameters"""
        self.W_xi = theano.shared(sample_weights(n_in, n_i))
        self.W_hi = theano.shared(sample_weights(n_hidden, n_i))

        """forget gate parameters"""
        self.W_xf = theano.shared(sample_weights(n_in, n_f))
        self.W_hf = theano.shared(sample_weights(n_hidden, n_f))

        """cell parameters"""
        self.W_xc = theano.shared(sample_weights(n_in, n_c))
        self.W_hc = theano.shared(sample_weights(n_hidden, n_c))

        """output gate parameters"""
        self.W_xo = theano.shared(sample_weights(n_in, n_o))
        self.W_ho = theano.shared(sample_weights(n_hidden, n_o))

        """output parameters"""
        self.W_hy = theano.shared(sample_weights(n_hidden, n_y))

        self.c0 = theano.shared(np.zeros(n_hidden, dtype=theano.config.floatX))
        self.h0 = self.activation(self.c0)

        self.params = [self.W_xi, self.W_hi, self.W_xf, self.W_hf, self.W_xc,
                       self.W_hc, self.W_xo, self.W_ho, self.W_hy, self.c0]

        self.x = self.emb[self.w]

        self.layer_output = self.layers(n_layers=n_layer)

        self.y, _ = theano.scan(fn=self.output_forward,
                                sequences=self.layer_output[-1],
                                outputs_info=[None])

        self.y = self.y[::-1]
        self.p_y_given_x = self.y.reshape((self.y.shape[0], self.y.shape[2]))
        self.nll = -T.mean(T.log(self.p_y_given_x)[T.arange(d.shape[0]), d])
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        self.errors = T.neq(self.y_pred, d)

    def layers(self, n_layers=2):
        layer_output = []
        for i in xrange(n_layers):
            if i == 0:
                layer_input = self.x
            else:
                layer_input = layer_output[-1][::-1]
            [h, c], _ = theano.scan(fn=self.forward,
                                    sequences=layer_input,
                                    outputs_info=[self.h0, self.c0])
            layer_output.append(h)
        return layer_output

    def forward(self, x_t, h_tm1, c_tm1):
        '''
        sequences: x_t
        prior results: h_tm1, c_tm1
        '''
        i_t = sigmoid(T.dot(x_t, self.W_xi) + T.dot(h_tm1, self.W_hi) + c_tm1)
        f_t = sigmoid(T.dot(x_t, self.W_xf) + T.dot(h_tm1, self.W_hf) + c_tm1)
        c_t = f_t * c_tm1 + i_t * self.activation(T.dot(x_t, self.W_xc) + T.dot(h_tm1, self.W_hc))
        o_t = sigmoid(T.dot(x_t, self.W_xo) + T.dot(h_tm1, self.W_ho) + c_t)
        h_t = o_t * self.activation(c_t)
        return h_t, c_t

    def output_forward(self, h_t):
        y_t = T.nnet.softmax(T.dot(h_t, self.W_hy))
        return y_t
