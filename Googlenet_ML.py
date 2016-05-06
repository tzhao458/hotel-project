#This code is implemented for Machine Learning class in UNCC
#License: non-commercial use only
#Author : Tianyi Zhao <tzhao4@uncc.edu>
#Created : <2016-05-02>

from lasagne.layers import InputLayer
from lasagne.layers import DenseLayer
from lasagne.layers import ConcatLayer
from lasagne.layers import NonlinearityLayer
from lasagne.layers import GlobalPoolLayer
from lasagne.layers.dnn import Conv2DDNNLayer as ConvLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as PoolLayerDNN
from lasagne.layers import MaxPool2DLayer as PoolLayer
from lasagne.layers import LocalResponseNormalization2DLayer as LRNLayer
from lasagne.nonlinearities import softmax, linear
import lasagne


import theano
import theano.tensor as T
import numpy as np
from lasagne.utils import floatX
import os
import sys
import time


sys.path.append('/usr/local/lib/python2.7/dist-packages')
import cv2

import lmdb
import linecache
import pickle


max_n_key = 30000
labels = np.load('labels.npy')
train_path = '/media/zbp/Elements/nc/ML_project/project/train/'
imglist = np.load('keys.npy')

def shared_dataset_x(data_x, borrow=True):
        """
	Transfer numpy type data to shared valued, 
	which can be stored in GPU memory
	
        """
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x



def load_data_lmdb(index,batch_size,dshape = (227,227), max_load_n = max_n_key,start = 0):
    """
    load data from images located in train_path
    """
    data = []
    max_index = max_n_key/batch_size
    if index > max_index:
	index = index%max_index

    for i in range(batch_size):
	img_path = train_path+imglist[start + batch_size*index+i]+'.jpg'
	img = cv2.imread(img_path)
	img = np.pad(img,((max(0,228-img.shape[0])/2,max(0,228-img.shape[0])/2),(max(0,228-img.shape[1])/2,max(0,228-img.shape[1])/2),(0,0)),'constant',constant_values=0)
	img = cv2.resize(img,(227,227))
	img = np.transpose(img,(2,0,1))	
	data += [img]

    label = labels[start + batch_size*index:start + batch_size*index+batch_size].astype(np.int32)
    return data,label


def build_inception_module(name, input_layer, nfilters):
    """
    inception graph
    """
    # nfilters: (pool_proj, 1x1, 3x3_reduce, 3x3, 5x5_reduce, 5x5)
    net = {}
    net['pool'] = PoolLayerDNN(input_layer, pool_size=3, stride=1, pad=1)
    net['pool_proj'] = ConvLayer(
        net['pool'], nfilters[0], 1, flip_filters=False)

    net['1x1'] = ConvLayer(input_layer, nfilters[1], 1, flip_filters=False)

    net['3x3_reduce'] = ConvLayer(
        input_layer, nfilters[2], 1, flip_filters=False)
    net['3x3'] = ConvLayer(
        net['3x3_reduce'], nfilters[3], 3, pad=1, flip_filters=False)

    net['5x5_reduce'] = ConvLayer(
        input_layer, nfilters[4], 1, flip_filters=False)
    net['5x5'] = ConvLayer(
        net['5x5_reduce'], nfilters[5], 5, pad=2, flip_filters=False)

    net['output'] = ConcatLayer([
        net['1x1'],
        net['3x3'],
        net['5x5'],
        net['pool_proj'],
        ])

    return {'{}/{}'.format(name, k): v for k, v in net.items()}

layers = ['conv1/7x7_s2', 'conv2/3x3_reduce', 'conv2/3x3', 'inception_3a/1x1', 'inception_3a/3x3_reduce', 'inception_3a/3x3', 'inception_3a/5x5_reduce', 'inception_3a/5x5', 'inception_3a/pool_proj', 'inception_3b/1x1', 'inception_3b/3x3_reduce', 'inception_3b/3x3', 'inception_3b/5x5_reduce', 'inception_3b/5x5', 'inception_3b/pool_proj', 'inception_4a/1x1', 'inception_4a/3x3_reduce', 'inception_4a/3x3', 'inception_4a/5x5_reduce', 'inception_4a/5x5', 'inception_4a/pool_proj', 'inception_4b/1x1', 'inception_4b/3x3_reduce', 'inception_4b/3x3', 'inception_4b/5x5_reduce', 'inception_4b/5x5', 'inception_4b/pool_proj', 'inception_4c/1x1', 'inception_4c/3x3_reduce', 'inception_4c/3x3', 'inception_4c/5x5_reduce', 'inception_4c/5x5', 'inception_4c/pool_proj', 'inception_4d/1x1', 'inception_4d/3x3_reduce', 'inception_4d/3x3', 'inception_4d/5x5_reduce', 'inception_4d/5x5', 'inception_4d/pool_proj', 'inception_4e/1x1', 'inception_4e/3x3_reduce', 'inception_4e/3x3', 'inception_4e/5x5_reduce', 'inception_4e/5x5', 'inception_4e/pool_proj', 'inception_5a/1x1', 'inception_5a/3x3_reduce', 'inception_5a/3x3', 'inception_5a/5x5_reduce', 'inception_5a/5x5', 'inception_5a/pool_proj', 'inception_5b/1x1', 'inception_5b/3x3_reduce', 'inception_5b/3x3', 'inception_5b/5x5_reduce', 'inception_5b/5x5', 'inception_5b/pool_proj']

def build_model(input_var=None):
    """
    googlenet graph
	
    """
    net = {}
    net['input'] = InputLayer((None, 3, None, None),input_var=input_var)
    net['conv1/7x7_s2'] = ConvLayer(
        net['input'], 64, 7, stride=2, pad=3, flip_filters=False)
    net['pool1/3x3_s2'] = PoolLayer(
        net['conv1/7x7_s2'], pool_size=3, stride=2, ignore_border=False)
    net['pool1/norm1'] = LRNLayer(net['pool1/3x3_s2'], alpha=0.00002, k=1)
    net['conv2/3x3_reduce'] = ConvLayer(
        net['pool1/norm1'], 64, 1, flip_filters=False)
    net['conv2/3x3'] = ConvLayer(
        net['conv2/3x3_reduce'], 192, 3, pad=1, flip_filters=False)
    net['conv2/norm2'] = LRNLayer(net['conv2/3x3'], alpha=0.00002, k=1)
    net['pool2/3x3_s2'] = PoolLayer(
      net['conv2/norm2'], pool_size=3, stride=2, ignore_border=False)

    net.update(build_inception_module('inception_3a',
                                      net['pool2/3x3_s2'],
                                      [32, 64, 96, 128, 16, 32]))
    net.update(build_inception_module('inception_3b',
                                      net['inception_3a/output'],
                                      [64, 128, 128, 192, 32, 96]))
    net['pool3/3x3_s2'] = PoolLayer(
      net['inception_3b/output'], pool_size=3, stride=2, ignore_border=False)

    net.update(build_inception_module('inception_4a',
                                      net['pool3/3x3_s2'],
                                      [64, 192, 96, 208, 16, 48]))
    net.update(build_inception_module('inception_4b',
                                      net['inception_4a/output'],
                                      [64, 160, 112, 224, 24, 64]))
    net.update(build_inception_module('inception_4c',
                                      net['inception_4b/output'],
                                      [64, 128, 128, 256, 24, 64]))
    net.update(build_inception_module('inception_4d',
                                      net['inception_4c/output'],
                                      [64, 112, 144, 288, 32, 64]))
    net.update(build_inception_module('inception_4e',
                                      net['inception_4d/output'],
                                      [128, 256, 160, 320, 32, 128]))
    net['pool4/3x3_s2'] = PoolLayer(
      net['inception_4e/output'], pool_size=3, stride=2, ignore_border=False)

    net.update(build_inception_module('inception_5a',
                                      net['pool4/3x3_s2'],
                                      [128, 256, 160, 320, 32, 128]))
    net.update(build_inception_module('inception_5b',
                                      net['inception_5a/output'],
                                      [128, 384, 192, 384, 48, 128]))

    net['pool5/7x7_s1'] = GlobalPoolLayer(net['inception_5b/output'])
    net['loss3/classifier'] = DenseLayer(net['pool5/7x7_s1'],
                                         num_units=8,
                                         nonlinearity=linear)
    net['prob'] = NonlinearityLayer(net['loss3/classifier'],
                                    nonlinearity=softmax)
    return net



def load_Googlenet():
    """
	train googlenet

	
    """
    new_v_data,new_v_label = load_data_lmdb(0,200)
    
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    
    # create lasagne network
    my_net = build_model(input_var)
    
    
    #load trained params
    params_dir = './Googlenet/' 
    for l in layers:
        try:
	    f = open(params_dir+l+'.npy','r')
	    w = np.load(f)
	    b = np.load(f)
	    f.close()
            my_net[l].W.set_value(w)
            my_net[l].b.set_value(b)
        except AttributeError:
            continue
        except KeyError:
            continue


    #randomly initial last layer
    rng = np.random.RandomState(1234)
    W_value = np.asarray(rng.normal(loc = 0, scale = 0.005, size=(1024,8)),
                dtype=theano.config.floatX)
    b_value = np.asarray([0 for j in range(8)],dtype=theano.config.floatX)
    my_net['loss3/classifier'].W.set_value(W_value)
    my_net['loss3/classifier'].b.set_value(b_value)



    #___________building test________________
    batch_size = 200
    test_set_x =  shared_dataset_x(np.zeros((batch_size,3,227,227)))
    test_set_label = np.zeros((batch_size)).astype(np.int32)
    test_set_y =  theano.shared(test_set_label,
 		borrow=True
		)
    test_output = lasagne.layers.get_output(my_net['prob'], deterministic=True)
    test_pred = T.argmax(test_output, axis=1)
    test_error = T.mean(T.neq(test_pred, test_set_y))
    #test_cost = lasagne.objectives.categorical_crossentropy(test_output, target_var).mean()
    test_cost = -T.mean(T.log(test_output)[T.arange(target_var.shape[0]), target_var])
    print(time.strftime("%H:%M:%S"),'building test')
    test_model = theano.function(
        inputs=[],
        outputs=(test_cost,test_error),
        givens={
            input_var: test_set_x,
            target_var: test_set_y
        }
    )
    




    #___________building train________________
    print(time.strftime("%H:%M:%S"),'building train')
    train_set_x =  shared_dataset_x(np.zeros((batch_size,3,227,227)))
    train_set_label = np.zeros((batch_size)).astype(np.int32)
    train_set_y =  theano.shared(train_set_label,
 		borrow=True
		)
    train_output = lasagne.layers.get_output(my_net['prob'], deterministic=True)
    train_pred = T.argmax(train_output, axis=1)
    train_error = T.mean(T.neq(train_pred, target_var))
    c1 = T.log(train_output)[T.arange(target_var.shape[0]), target_var]
    cost = -T.mean(T.log(train_output)[T.arange(target_var.shape[0]), target_var])
    #cost = lasagne.objectives.categorical_crossentropy(train_output, target_var).mean()

    params = lasagne.layers.get_all_params(my_net['loss3/classifier'], trainable=True)
    #params = [my_net['loss3/classifier'].W,my_net['loss3/classifier'].b]
    learning_rate = T.fscalar()
    updates = lasagne.updates.nesterov_momentum(
            cost, params, learning_rate=learning_rate, momentum=0.9)

    train_model = theano.function(
        inputs=[learning_rate],
        outputs=(cost,train_error,train_output,train_pred,c1),
        updates=updates,
        givens={
            input_var: train_set_x,
            target_var: train_set_y
        }
    )
    

    
    #___________training________________
    best_cs_err = np.inf
    last_cs_err = np.inf
    print((time.strftime("%H:%M:%S")),'training')
    cs_list = []
    err_list = []   
    lr = 0.0001
    lr_count = 0
    for epoch in range(200000):
	#______________data_____________________
	
	new_b_data,new_b_label = load_data_lmdb(epoch,batch_size)	
	train_set_x.set_value(np.asarray(new_b_data,
                                               dtype=theano.config.floatX))
	train_set_label[:] = new_b_label

    	(cs,err,out,pred,cc1) = train_model(lr)
	cs_list += [cs]
	err_list += [err]
	#print('out',out[:2])
	#print('pred',pred[:20])
	#print('new_b_label',new_b_label[:20])
	#print('c1',cc1[:20])
	
        
	if epoch%20 == 0:
    		print((time.strftime("%H:%M:%S")),'epoch',epoch,'cost',np.mean(cs_list),'train err',np.mean(err_list),'lr',lr)
		#print(yp)
		cs_list = []
		err_list = []

	#______________val_____________________
	if epoch%150 == 0 and epoch >0:
	    val_cs_list = []
	    val_err_list = []
	    this_start = max_n_key
	    for v_epoch in range(40):
		new_v_data,new_v_label = load_data_lmdb(v_epoch,batch_size,start = this_start)	
		test_set_x.set_value(np.asarray(new_v_data,
                                               dtype=theano.config.floatX))
		test_set_label[:] = new_v_label
		
		val_cs,val_err = test_model()
		val_cs_list += [val_cs]
		val_err_list += [val_err]
	    
	    	print(time.strftime("%H:%M:%S"),'v_epoch',v_epoch,'val cs',np.mean(val_cs_list),'val err',np.mean(val_err_list))

	    #if epoch%5000 == 0 and epoch >0:
	    #		lr /= 10
	    
	    lr_count += 1
	    if np.mean(val_cs_list)>(last_cs_err+0.01) or lr_count>= 30:
	    		lr /= 10
			lr_count = 0
			
	    
	    if np.mean(val_cs_list)<best_cs_err:
		with open('/media/zbp/Elements/nc/ML_project/project/loss3_'+str(np.mean(val_cs_list))[:5]+'.pkl', 'wb') as f:
                    pickle.dump(params, f)
		best_cs_err=np.mean(val_cs_list)


if __name__ == "__main__":
    load_Googlenet()
