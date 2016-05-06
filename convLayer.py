#This code is implemented for Machine Learning class in UNCC
#License: non-commercial use only
#Author : Tianyi Zhao <tzhao4@uncc.edu>
#Created : <2016-05-02>

import numpy
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d,relu

class ConvLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, **kwargs):
        """
        Convolutional layer

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
	
        """
        pad = kwargs.get('pad', 0)
        subsample = kwargs.get('subsample', (1,1))
	self.W_learning_rate=kwargs.get('W_lr_mult', 0.01)
	self.W_decay_mult = kwargs.get('W_decay_mult', 0)
	self.b_learning_rate=kwargs.get('b_lr_mult', 0.01)
	self.b_decay_mult = kwargs.get('b_decay_mult', 0)


        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" 
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:])) #//numpy.prod(poolsize))

        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
	W_value = kwargs.get('W', numpy.asarray(
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                dtype=theano.config.floatX
            ))
	W_value = W_value.astype(theano.config.floatX)
        self.W = theano.shared(
            W_value,
            borrow=True
        )

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = kwargs.get('b',numpy.zeros((filter_shape[0],), dtype=theano.config.floatX))
	b_values = b_values.astype(theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
	
        conv_out = conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
	    border_mode = pad,
	    subsample = subsample
        )


        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input
