#This code is implemented for Machine Learning class in UNCC
#License: non-commercial use only
#Author : Tianyi Zhao <tzhao4@uncc.edu>
#Created : <2016-05-02>

from theano.tensor.signal import pool

class PoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, input, poolsize=(2, 2), st=(2,2), padding = (0,0),mode = 'max'):
        """
        Allocate a PoolLayer with shared variable internal parameters.

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)

        :type st: tuple or list of length 2
        :param st: stride (#rows, #cols)

        :type padding: tuple or list of length 2
        :param padding: 

        :type mode: String
        :param mode: 'max','sum','average_inc_pad', 'average_exc_pad'
        """

        self.input = input
        pooled_out = pool.pool_2d(
            input=input,
            ds=poolsize,
            ignore_border=True,
	    st = st,
	    padding = padding,
	    mode = mode
        )

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = pooled_out

