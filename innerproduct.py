#This code is implemented for Machine Learning class in UNCC
#License: non-commercial use only
#Author : Tianyi Zhao <tzhao4@uncc.edu>
#Created : <2016-05-02>

import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import relu


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=relu,**kwargs):
        """
        Fully-connected with Relu activation function as default.
	Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
	self.W_learning_rate=kwargs.get('W_lr_mult', 0.01)
	self.W_decay_mult = kwargs.get('W_decay_mult', 0)
	self.b_learning_rate=kwargs.get('b_lr_mult', 0.01)
	self.b_decay_mult = kwargs.get('b_decay_mult', 0)

        # `W` is initialized 
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
        	W_values *= 4
	else:
	    #print('inner',W.shape,(n_in, n_out))
	    W_values =  W.reshape((n_in, n_out))
	    W_values =  W_values.astype(theano.config.floatX)

        W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
	else:
	    b_values = b.reshape((n_out,))
	    b_values = b_values.astype(theano.config.floatX)
        b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b


        # compute output
        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]
