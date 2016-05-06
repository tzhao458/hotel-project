#This code is implemented for Machine Learning class in UNCC
#License: non-commercial use only
#Author : Tianyi Zhao <tzhao4@uncc.edu>
#Created : <2016-05-02>

from __future__ import print_function

__docformat__ = 'restructedtext en'


import os
import sys
import timeit
import time

import numpy

import theano
import theano.tensor as T


from innerproduct import HiddenLayer 
from convLayer import ConvLayer
from poolLayer import PoolLayer
from lrnLayer import LrnLayer
from dropoutLayer import DropoutLayer
from softmaxLossLayer import SoftmaxLossLayer

class AlexNet(object):
    """same with AlexNet network,
	the last layer is rescaled to 8 classes 
    """

    def __init__(self, rng, input, batch_size,**kwargs):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type batch_size: int
        :param batch_size: 


        """

	#conv1
	conv1_input = input.reshape((batch_size, 3, 227, 227))

	conv1_n_out = 96
	filter_shape=(conv1_n_out, 3, 11, 11)
	W_loc = kwargs.get('conv1_W_loc',0)
	W_std = kwargs.get('conv1_W_std',0.01)
	W_value = kwargs.get('conv1_W', numpy.asarray(
                rng.normal(loc = W_loc, scale = W_std, size=filter_shape),
                dtype=theano.config.floatX
        ))
	b_constant = kwargs.get('conv1_b_constant',0)
	b_values = kwargs.get('conv1_b',numpy.asarray([b_constant for j in range(conv1_n_out)]))
	
	
	self.conv1 = ConvLayer(
            rng,
            input=conv1_input,
            filter_shape=filter_shape,
	    subsample=(4,4),
	    W=W_value,
	    b=b_values,
	    W_lr_mult = 1,
	    W_decay_mult = 1,
	    b_lr_mult = 2,
	    b_decay_mult = 0
    	)

        self.pool1 = PoolLayer(
            input=self.conv1.output,
            poolsize=(3, 3)
        )

	self.norm1 = LrnLayer(
            input=self.pool1.output
        )
	


	#conv2
	conv2_n_out = 256
	filter_shape=(conv2_n_out, conv1_n_out/2, 5, 5)
	W_loc = kwargs.get('conv2_W_loc',0)
	W_std = kwargs.get('conv2_W_std',0.01)
	W_value = kwargs.get('conv2_W', numpy.asarray(
                rng.normal(loc = W_loc, scale = W_std, size=filter_shape),
                dtype=theano.config.floatX
        ))
	b_constant = kwargs.get('conv2_b_constant',1)
	b_value = kwargs.get('conv2_b',numpy.asarray([b_constant for j in range(conv2_n_out)]))

	conv2_1_input = self.norm1.output[:,:conv1_n_out/2]
        self.conv2_1 = ConvLayer(
            rng,
            input=conv2_1_input,
            filter_shape=(conv2_n_out/2, conv1_n_out/2, 5, 5),
	    pad = 2,
	    W=W_value[:conv2_n_out/2],
	    b=b_value[:conv2_n_out/2],
	    W_lr_mult = 1,
	    W_decay_mult = 1,
	    b_lr_mult = 2,
	    b_decay_mult = 0
        )
	conv2_2_input = self.norm1.output[:,conv1_n_out/2:]
        self.conv2_2 = ConvLayer(
            rng,
            input=conv2_2_input,
            filter_shape=(conv2_n_out/2, conv1_n_out/2, 5, 5),
	    pad = 2,
	    W=W_value[conv2_n_out/2:],
	    b=b_value[conv2_n_out/2:],
	    W_lr_mult = 1,
	    W_decay_mult = 1,
	    b_lr_mult = 2,
	    b_decay_mult = 0
        )
  
	pool2_input = T.concatenate((self.conv2_1.output,self.conv2_2.output),axis=1)
        self.pool2 = PoolLayer(
            input=pool2_input,
            poolsize=(3, 3)
        )

	self.norm2 = LrnLayer(
            input=self.pool2.output
        )
	

	
	#conv3
	conv3_n_out = 384
	filter_shape = (conv3_n_out, conv2_n_out, 3, 3)
	W_loc = kwargs.get('conv3_W_loc',0)
	W_std = kwargs.get('conv3_W_std',0.01)
	W_value = kwargs.get('conv3_W', numpy.asarray(
                rng.normal(loc = W_loc, scale = W_std, size=filter_shape),
                dtype=theano.config.floatX
        ))
	b_constant = kwargs.get('conv3_b_constant',0)
	b_value = kwargs.get('conv3_b',numpy.asarray([b_constant for j in range(conv3_n_out)]))
        self.conv3 = ConvLayer(
            rng,
            input=self.norm2.output,
            filter_shape=filter_shape,
	    pad = 1,
	    W=W_value,
	    b=b_value,
	    W_lr_mult = 1,
	    W_decay_mult = 1,
	    b_lr_mult = 2,
	    b_decay_mult = 0
        )


	#conv4
	conv4_n_out = 384
	filter_shape = (conv4_n_out, conv3_n_out/2, 3, 3)
	W_loc = kwargs.get('conv4_W_loc',0)
	W_std = kwargs.get('conv4_W_std',0.01)
	W_value = kwargs.get('conv4_W', numpy.asarray(
                rng.normal(loc = W_loc, scale = W_std, size=filter_shape),
                dtype=theano.config.floatX
        ))
	b_constant = kwargs.get('conv4_b_constant',1)
	b_value = kwargs.get('conv4_b',numpy.asarray([b_constant for j in range(conv4_n_out)]))

	conv4_1_input = self.conv3.output[:,:conv3_n_out/2]
        self.conv4_1 = ConvLayer(
            rng,
            input=conv4_1_input,
            filter_shape=(conv4_n_out/2, conv3_n_out/2, 3, 3),
	    pad = 1,
	    W=W_value[:conv4_n_out/2],
	    b=b_value[:conv4_n_out/2],
	    W_lr_mult = 1,
	    W_decay_mult = 1,
	    b_lr_mult = 2,
	    b_decay_mult = 0
        )
	conv4_2_input = self.conv3.output[:,conv3_n_out/2:]
        self.conv4_2 = ConvLayer(
            rng,
            input=conv4_2_input,
            filter_shape=(conv4_n_out/2, conv3_n_out/2, 3, 3),
	    pad = 1,
	    W=W_value[conv4_n_out/2:],
	    b=b_value[conv4_n_out/2:],
	    W_lr_mult = 1,
	    W_decay_mult = 1,
	    b_lr_mult = 2,
	    b_decay_mult = 0
        )




	#conv5
	conv5_n_out = 256
	filter_shape = (conv5_n_out, conv4_n_out/2, 3, 3)
	W_loc = kwargs.get('conv5_W_loc',0)
	W_std = kwargs.get('conv5_W_std',0.01)
	W_value = kwargs.get('conv5_W', numpy.asarray(
                rng.normal(loc = W_loc, scale = W_std, size=filter_shape),
                dtype=theano.config.floatX
        ))
	b_constant = kwargs.get('conv5_b_constant',1)
	b_value = kwargs.get('conv5_b',numpy.asarray([b_constant for j in range(conv5_n_out)]))
        self.conv5_1 = ConvLayer(
            rng,
            input=self.conv4_1.output,
            filter_shape=(conv5_n_out/2, conv4_n_out/2, 3, 3),
	    pad = 1,
	    W=W_value[:conv5_n_out/2],
	    b=b_value[:conv5_n_out/2],
	    W_lr_mult = 1,
	    W_decay_mult = 1,
	    b_lr_mult = 2,
	    b_decay_mult = 0
        )
  
        self.conv5_2 = ConvLayer(
            rng,
            input=self.conv4_2.output,
            filter_shape=(conv5_n_out/2, conv4_n_out/2, 3, 3),
	    pad = 1,
	    W=W_value[conv5_n_out/2:],
	    b=b_value[conv5_n_out/2:],
	    W_lr_mult = 1,
	    W_decay_mult = 1,
	    b_lr_mult = 2,
	    b_decay_mult = 0
        )


	pool5_input = T.concatenate((self.conv5_1.output,self.conv5_2.output),axis=1)
        self.pool5 = PoolLayer(
            input=pool5_input,
            poolsize=(3, 3)
        )



	#fc6
        fc6_input = self.pool5.output.flatten(2)
	fc6_n_out = 4096
	n_in = conv5_n_out*6*6
	filter_shape = (n_in,fc6_n_out)
	W_loc = kwargs.get('fc6_W_loc',0)
	W_std = kwargs.get('fc6_W_std',0.005)
	W_value = kwargs.get('fc6_W', numpy.asarray(
                rng.normal(loc = W_loc, scale = W_std, size=filter_shape),
                dtype=theano.config.floatX
        ))
	b_constant = kwargs.get('fc6_b_constant',1)
	b_value = kwargs.get('fc6_b',numpy.asarray([b_constant for j in range(fc6_n_out)]))
        self.fc6 = HiddenLayer(
	    rng=rng,
	    input=fc6_input,
	    n_in=n_in,  #####6 
	    n_out=fc6_n_out,
	    W=W_value,
	    b=b_value,
	    W_lr_mult = 1,
	    W_decay_mult = 1,
	    b_lr_mult = 2,
	    b_decay_mult = 0
        )

	self.drop_prob = kwargs.get('drop_prob',0.5)
	self.dropout1 = DropoutLayer(
	    rng,
            input=self.fc6.output,
	    prob_drop=self.drop_prob
        )
	

	#fc7
	fc7_n_out = 4096
	n_in = fc6_n_out
	filter_shape = (n_in,fc7_n_out)
	W_loc = kwargs.get('fc7_W_loc',0)
	W_std = kwargs.get('fc7_W_std',0.005)
	W_value = kwargs.get('fc7_W', numpy.asarray(
                rng.normal(loc = W_loc, scale = W_std, size=filter_shape),
                dtype=theano.config.floatX
        ))
	b_constant = kwargs.get('fc7_b_constant',1)
	b_value = kwargs.get('fc7_b',numpy.asarray([b_constant for j in range(fc7_n_out)]))
        self.fc7 = HiddenLayer(
	    rng=rng,
	    input=self.dropout1.output,
	    n_in=n_in,
	    n_out=fc7_n_out,
	    W=W_value,
	    b=b_value,
	    W_lr_mult = 1,
	    W_decay_mult = 1,
	    b_lr_mult = 2,
	    b_decay_mult = 0
        )


	self.dropout2 = DropoutLayer(
	    rng,
            input=self.fc7.output,
	    prob_drop=self.drop_prob
        )



	#fc8
	fc8_n_out = 8
	n_in = fc7_n_out
	filter_shape = (n_in,fc8_n_out)
	W_loc = kwargs.get('fc8_W_loc',0)
	W_std = kwargs.get('fc8_W_std',0.01)
	W_value = kwargs.get('fc8_W', numpy.asarray(
                rng.normal(loc = W_loc, scale = W_std, size=filter_shape),
                dtype=theano.config.floatX
        ))
	b_constant = kwargs.get('fc8_b_constant',0)
	b_value = kwargs.get('fc8_b',numpy.asarray([b_constant for j in range(fc8_n_out)]))
        self.fc8 = HiddenLayer(
	    rng=rng,
	    input=self.dropout2.output,
	    n_in=n_in,
	    n_out=fc8_n_out,
	    W=W_value,
	    b=b_value,
	    W_lr_mult = 1,
	    W_decay_mult = 1,
	    b_lr_mult = 2,
	    b_decay_mult = 0,
	    activation = None
        )

        self.loss = SoftmaxLossLayer(
	    input=self.fc8.output
        )

        
        
        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # SoftmaxLossLayer layer
        self.negative_log_likelihood = (
            self.loss.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.loss.errors

        # the parameters of the model are the parameters of the 8 layers it is
        # made out of
        self.params = self.fc8.params + self.fc7.params + self.fc6.params + self.conv4_1.params + self.conv4_2.params + self.conv5_1.params + self.conv5_2.params +self.conv3.params + self.conv2_2.params + self.conv2_1.params + self.conv1.params

     
	self.y_pred = self.loss.y_pred
        # keep track of model input
        self.input = input


    def trun_on_dropoff(self):
	self.dropout1.flag_on.set_value(1.0)
	self.dropout2.flag_on.set_value(1.0)

    def trun_off_dropoff(self):
	self.dropout1.flag_on.set_value(0.0)
	self.dropout2.flag_on.set_value(0.0)


