#This code is implemented for Machine Learning class in UNCC
#License: non-commercial use only
#Author : Tianyi Zhao <tzhao4@uncc.edu>
#Created : <2016-05-02>

import theano.tensor as T
import theano
import numpy as np
class DropoutLayer(object):
    """Dropout Layer of a convolutional network """

    def __init__(self, rng, input, prob_drop=1):
        
	self.input = input
        self.flag_on = theano.shared(np.cast[theano.config.floatX](1.0))
	seed_this = rng.randint(0, 2**31-1)
	mask_rng = T.shared_randomstreams.RandomStreams(seed_this)
	self.output = \
		self.flag_on*T.switch(mask_rng.binomial(size=input.shape,p=prob_drop),input,0) \
		+ (1.0 - self.flag_on)*prob_drop*input
	
 
        
