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

import numpy as np

import theano
import theano.tensor as T

from alexNet_ML import AlexNet

#########
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


	

def test_alexnet(batch_size=200,eta = 0.0005, mu = 0.9,max_iter=200000,val_frequence=20):
    

    #___________building model________________
    print((time.strftime("%H:%M:%S")),'building model')
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.tensor4('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of # [int] labels

    rng = np.random.RandomState(1234)

  

    #load trained params
    with open('/media/zbp/Elements/nc/ML_project/project/alexnet_ML/12345678_0.444.pkl','r') as f:
    	coarse_params = pickle.load(f)
	fc8_W = coarse_params[0].get_value()
	fc8_b = coarse_params[1].get_value()
	fc7_W = coarse_params[2].get_value()
	fc7_b = coarse_params[3].get_value()
	fc6_W = coarse_params[4].get_value()
	fc6_b = coarse_params[5].get_value()
	conv4_W1 = coarse_params[6].get_value()
	conv4_b1 = coarse_params[7].get_value()
	conv4_W2 = coarse_params[8].get_value()
	conv4_b2 = coarse_params[9].get_value()
	conv4_W = np.concatenate((conv4_W1,conv4_W2))
	conv4_b = np.concatenate((conv4_b1,conv4_b2)) 
	conv5_W1 = coarse_params[10].get_value()
	conv5_b1 = coarse_params[11].get_value()
	conv5_W2 = coarse_params[12].get_value()
	conv5_b2 = coarse_params[13].get_value()
	conv5_W = np.concatenate((conv5_W1,conv5_W2))
	conv5_b = np.concatenate((conv5_b1,conv5_b2))
	conv3_W = coarse_params[14].get_value()
	conv3_b = coarse_params[15].get_value()
	conv2_W2 = coarse_params[16].get_value()
	conv2_b2 = coarse_params[17].get_value()
	conv2_W1 = coarse_params[18].get_value()
	conv2_b1 = coarse_params[19].get_value()
	conv2_W = np.concatenate((conv2_W1,conv2_W2))
	conv2_b = np.concatenate((conv2_b1,conv2_b2))
	conv1_W = coarse_params[20].get_value()
	conv1_b = coarse_params[21].get_value()



    classifier = AlexNet(
        rng=rng,
        input=x,
        batch_size=batch_size,
	conv1_W = conv1_W,
	conv1_b = conv1_b,
	conv2_W = conv2_W,
	conv2_b = conv2_b,
	conv3_W = conv3_W,
	conv3_b = conv3_b,
	conv4_W = conv4_W,
	conv4_b = conv4_b,
	conv5_W = conv5_W,
	conv5_b = conv5_b,
	fc6_W = fc6_W,
	fc6_b = fc6_b,
	fc7_W = fc7_W,
	fc7_b = fc7_b,
	fc8_W = fc8_W,
	fc8_b = fc8_b
    )
    


    #___________building test________________
    print((time.strftime("%H:%M:%S")),'building test')
    test_set_x =  shared_dataset_x(np.zeros((batch_size,3,227,227)))
    test_set_label = np.zeros((batch_size)).astype(np.int32)
    test_set_y =  theano.shared(test_set_label,
 		borrow=True
		)
    
    test_model = theano.function(
        inputs=[],
        outputs=(classifier.negative_log_likelihood(y),classifier.errors(y)),
        givens={
            x: test_set_x,
            y: test_set_y
        }
    )




    #___________building train________________
    print((time.strftime("%H:%M:%S")),'building train')
    y_p_give_x = classifier.loss.p_y_given_x
    y_p_true =  theano.shared(np.zeros([batch_size,8]).astype(np.float32))
    beta = 0.005
    cost = T.nnet.categorical_crossentropy(y_p_give_x,(1-beta)*y_p_true+beta*(y_p_give_x)).mean()
    #cost = classifier.negative_log_likelihood(y)
    params = classifier.params
    gparams = T.grad(cost, params)
    vels = [theano.shared(param_i.get_value() * 0.)
            for param_i in params]
    updates = []
    learning_rate = T.fscalar()

    i=0
    for param_i, grad_i, vel_i in zip(params, gparams, vels):
	if i%2==0: #w 
	    real_grad = grad_i +  eta * param_i
	    real_lr = learning_rate
	else:
	    real_grad = grad_i
	    real_lr = 2*learning_rate

	vel_i_next = mu * vel_i - real_lr * real_grad
	i+=1

	updates.append((vel_i, vel_i_next))
	updates.append((param_i, param_i + vel_i_next))



    train_set_x =  shared_dataset_x(np.zeros((batch_size,3,227,227)))
    minibatch_label = np.zeros((batch_size)).astype(np.int32)
    train_set_y =  theano.shared(minibatch_label,
 		borrow=True
		)
    
    train_model = theano.function(
        inputs=[learning_rate],
        outputs=(cost,classifier.errors(y)),
        updates=updates,
        givens={
            x: train_set_x,
            y: train_set_y
        }
    )

    
    predict_model = theano.function(
        inputs=[],
        outputs=(classifier.loss.y_pred,classifier.loss.p_y_given_x),
        givens={
            x: test_set_x
        }
    )

    #___________training________________
    best_val_cs = 0.443
    print((time.strftime("%H:%M:%S")),'training')
    cs_list = []
    err_list = []  
    lr = 0.0001
    last_var_err_mean = np.inf
    lr_ada_n = 0
    
    classifier.trun_on_dropoff()
    for epoch in range(max_iter):
	#______________data_____________________
	
	new_b_data,new_b_label = load_data_lmdb(epoch,batch_size)	
	train_set_x.set_value(np.asarray(new_b_data,
                                               dtype=theano.config.floatX))
	minibatch_label[:] = new_b_label

	label_dis = np.zeros([batch_size,8])
	label_dis[np.arange(batch_size),new_b_label] = 1
	y_p_true.set_value(label_dis.astype(np.float32))
    
	#_____________train_____________________
    	(cs,err) = train_model(lr)
	#print('len c1',c1_dex[0].shape,c1_dex)
	cs_list += [cs]
	err_list += [err]

	if epoch%val_frequence == 0:
    		print((time.strftime("%H:%M:%S")),'epoch',epoch,'cost',np.mean(cs_list),'train err',np.mean(err_list),'lr',lr)
		cs_list = []
		err_list = []

	#______________val_____________________
	if epoch%150 == 0 and epoch > 0:
	    classifier.trun_off_dropoff()
	    val_cs_list = []
	    val_err_list = []
	    ycsl = []
	    ypl = []
	    this_start = max_n_key
	    for v_epoch in range(40):
		new_v_data,new_v_label = load_data_lmdb(v_epoch,batch_size,start=this_start)	
		test_set_x.set_value(np.asarray(new_v_data,
                                               dtype=theano.config.floatX))
		test_set_label[:] = new_v_label
		
		
		(val_cs,val_err)= test_model()
		val_cs_list += [val_cs]
		val_err_list += [val_err]

		
	    print((time.strftime("%H:%M:%S")),'epoch',epoch,'val cost',np.mean(val_cs_list),'val err',np.mean(val_err_list))
	    
	    if last_var_err_mean < np.mean(val_cs_list) and lr_ada_n<3:
			lr /= 10
			lr_ada_n += 1
			
	    
	    if np.mean(val_cs_list)<best_val_cs:
		with open('/media/zbp/Elements/nc/ML_project/project/alexnet_ML/12345678_'+str(np.mean(val_cs_list))[:5]+'.pkl', 'wb') as f:
                    pickle.dump(params, f)
		best_val_err = np.mean(val_err_list)

	    last_var_err_mean = np.mean(val_cs_list)
	    classifier.trun_on_dropoff()
		

	

if __name__ == '__main__':
    test_alexnet()






