#This code is implemented for Machine Learning class in UNCC
#License: non-commercial use only
#Author : Tianyi Zhao <tzhao4@uncc.edu>
#Created : <2016-05-02>

from pylearn2.expr.normalize import CrossChannelNormalization,CrossChannelNormalizationBC01
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool
class LrnLayer(object):
    """Norm Layer of a convolutional network """

    def __init__(self, input):
        """
        LRN norm layer
        """
	
	lrc_fuc = CrossChannelNormalization()
	lrn_input = input.transpose([1,2,3,0])
        self.input = input
        normed_out = lrc_fuc(
     		lrn_input
        )
	
        self.output = normed_out.transpose((3,0,1,2))

