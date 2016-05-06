#This code is implemented for Machine Learning class in UNCC
#License: non-commercial use only
#Author : Tianyi Zhao <tzhao4@uncc.edu>
#Created : <2016-05-02>



from __future__ import print_function

__docformat__ = 'restructedtext en'

import six.moves.cPickle as pickle
import gzip
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T



class SoftmaxLossLayer(object):
    """SoftmaxLossLayer.
    """

    def __init__(self, input):
        """ 
        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        """
       

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        self.p_y_given_x = T.nnet.softmax(input)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # keep track of model input
        self.input = input

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
       

    def errors(self, y):
	return T.mean(T.neq(self.y_pred, y))
       


