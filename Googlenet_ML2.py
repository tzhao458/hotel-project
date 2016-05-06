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

from Googlenet_ML import build_model

max_n_key = 30000
labels = np.load('labels.npy')
test_path = '/media/zbp/Elements/nc/ML_project/project/test/'
imglist = os.listdir(test_path)
imglist = np.sort(imglist)
def shared_dataset_x(data_x, borrow=True):
    """
    Transfer numpy type data to shared valued, 
    which can be stored in GPU memory
	
    """
    shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
    return shared_x



def load_data(index,batch_size,dshape = (227,227)):
    """
    load data from images located in test_path
	
    """
    
    data = []
    idxs = []
    keys = []
    nonidx = []
    nonkeys = []
    batch_size_temp = batch_size
    i=0
    while i < batch_size_temp:
	idx = batch_size*index+i
	i+=1
	idx = idx%len(imglist)
	key = imglist[idx][:len(imglist[idx])-4]

	img_path = test_path+imglist[idx]
	img = cv2.imread(img_path)
	if img==None:
		batch_size_temp += 1
		nonidx += [idx]
		nonkeys += [key]
		continue
	img = np.pad(img,((max(0,228-img.shape[0])/2,max(0,228-img.shape[0])/2),(max(0,228-img.shape[1])/2,max(0,228-img.shape[1])/2),(0,0)),'constant',constant_values=0)
	img = cv2.resize(img,(227,227))
	img = np.transpose(img,(2,0,1))	
	data += [img]
	idxs += [idx]
	keys += [key]
    return data,idxs,keys,nonidx,nonkeys



layers = ['conv1/7x7_s2', 'conv2/3x3_reduce', 'conv2/3x3', 'inception_3a/1x1', 'inception_3a/3x3_reduce', 'inception_3a/3x3', 'inception_3a/5x5_reduce', 'inception_3a/5x5', 'inception_3a/pool_proj', 'inception_3b/1x1', 'inception_3b/3x3_reduce', 'inception_3b/3x3', 'inception_3b/5x5_reduce', 'inception_3b/5x5', 'inception_3b/pool_proj', 'inception_4a/1x1', 'inception_4a/3x3_reduce', 'inception_4a/3x3', 'inception_4a/5x5_reduce', 'inception_4a/5x5', 'inception_4a/pool_proj', 'inception_4b/1x1', 'inception_4b/3x3_reduce', 'inception_4b/3x3', 'inception_4b/5x5_reduce', 'inception_4b/5x5', 'inception_4b/pool_proj', 'inception_4c/1x1', 'inception_4c/3x3_reduce', 'inception_4c/3x3', 'inception_4c/5x5_reduce', 'inception_4c/5x5', 'inception_4c/pool_proj', 'inception_4d/1x1', 'inception_4d/3x3_reduce', 'inception_4d/3x3', 'inception_4d/5x5_reduce', 'inception_4d/5x5', 'inception_4d/pool_proj', 'inception_4e/1x1', 'inception_4e/3x3_reduce', 'inception_4e/3x3', 'inception_4e/5x5_reduce', 'inception_4e/5x5', 'inception_4e/pool_proj', 'inception_5a/1x1', 'inception_5a/3x3_reduce', 'inception_5a/3x3', 'inception_5a/5x5_reduce', 'inception_5a/5x5', 'inception_5a/pool_proj', 'inception_5b/1x1', 'inception_5b/3x3_reduce', 'inception_5b/3x3', 'inception_5b/5x5_reduce', 'inception_5b/5x5', 'inception_5b/pool_proj']



def load_Googlenet(max_iter=99):
    """
    test googlenet
    generate test.csv
	
    """
    
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


    #load initial last layer

    with open('loss3_0.437.pkl','r') as f:
    	params = pickle.load(f)
	fc8_W = params[0].get_value()
	fc8_b = params[1].get_value()
    my_net['loss3/classifier'].W.set_value(fc8_W)
    my_net['loss3/classifier'].b.set_value(fc8_b)



    #___________building predict________________
    batch_size = 200
    test_set_x =  shared_dataset_x(np.zeros((batch_size,3,227,227)))

    test_output = lasagne.layers.get_output(my_net['prob'], deterministic=True)
    print(time.strftime("%H:%M:%S"),'building predict')
    predict_model = theano.function(
        inputs=[],
        outputs=(test_output),
        givens={
            input_var: test_set_x
        }
    )
    


    #___________testing________________
    print((time.strftime("%H:%M:%S")),'testing')
    result = np.ones((len(imglist),8))
    label = np.zeros(len(imglist))
    for epoch in range(max_iter):
	    #______________data_____________________
	
	    new_b_data,new_b_idx,new_b_key,nonidx,nonkey = load_data(epoch,batch_size)	
	    test_set_x.set_value(np.asarray(new_b_data,
                                               dtype=theano.config.floatX))

	    #_____________train_____________________
    	    y = predict_model()
	    result[new_b_idx] = y
	    label[new_b_idx] = np.argmax(y,axis=1)
	    if epoch%20 ==0:
		print(epoch,y.shape)


    np.save('result_ML4',result)
    print(result.shape,result[0])



    #___________generating test.csv________________
    with open('test4.csv', 'wb') as csvfile:
	spamwriter = csv.writer(csvfile, delimiter=' ',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    	spamwriter.writerow(['id,col1,col2,col3,col4,col5,col6,col7,col8'])
	for i,y in enumerate(result):
		row = imglist[i][:len(imglist[i])-4]
		for n in y:
		    row += ','+str(n)
    		spamwriter.writerow([row])
		if i%2000 ==0:
			print(i,row)
    
    


if __name__ == "__main__":
    load_Googlenet()
