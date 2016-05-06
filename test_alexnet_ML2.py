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

sys.path.append('/usr/local/lib/python2.7/dist-packages')
import cv2
import pickle
import csv


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


	

def test_alexnet(batch_size=200,eta = 0.0005, mu = 0.9,max_iter=99,val_frequence=20):
    """
	test trained alexnet
	generate test.csv

    """
    
    #___________building model________________
    print((time.strftime("%H:%M:%S")),'building model')
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.tensor4('x')  # the data is presented as rasterized images

    rng = np.random.RandomState(1234)


    #load trained params
    with open('12345678_0.441.pkl','r') as f:
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
    


    #___________building predict theano graph________________
    print((time.strftime("%H:%M:%S")),'building predict')
    test_set_x =  shared_dataset_x(np.zeros((batch_size,3,227,227)))
    
    predict_model = theano.function(
        inputs=[],
        outputs=(classifier.loss.p_y_given_x),
        givens={
            x: test_set_x
        }
    )




   
    #___________testing________________
    print((time.strftime("%H:%M:%S")),'testing')
    classifier.trun_off_dropoff()
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


    np.save('result_ML3',result)
    #result=np.load('result_ML2.npy')
    print(result.shape,result[0])



    #___________generating test.csv________________
    with open('test3.csv', 'wb') as csvfile:
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

	
		

	

if __name__ == '__main__':
    test_alexnet()






