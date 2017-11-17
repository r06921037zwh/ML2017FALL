# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop, SGD, Adam
from keras.layers import Conv2D, MaxPooling2D, Flatten, ZeroPadding2D, Input
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
#from keras.models import Sequential,Model 
#from keras.layers.core import Dense, Dropout, Activation
#from keras.optimizers import SGD, Adam
import numpy as np
import csv
import sys
import argparse

####################################################################################
#-----------------------------------for testing------------------------------------#
####################################################################################
    
def load_test(test_data_path):
    with open(test_data_path, 'r') as test_csv:
    	test = csv.reader(test_csv)
    	x_test = []
    	next(test)               #from the second line
    	for row in test:
    		t = [float(i)/255 for i in row[1].split()]
    		x_test.append(t)
    x_test = np.array(x_test)
    x_test = x_test.reshape(len(x_test),48,48,1)
    
    return x_test

def get_result(predict_prob, outputs_dir):
    result = []
    for i in range(len(predict_prob)):
        result.append(np.argmax(predict_prob[i]))
        
    with open(outputs_dir, 'w', newline='') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow(['id', 'label'])
        index = 0
        for element in result:
            writer.writerow([str(index), str(int(element))])
            index += 1
    

def main():
    x_test = load_test(sys.argv[1])
    model = load_model('hw3_model.h5')
    prediction_prob = model.predict(x_test)
    get_result(prediction_prob, sys.argv[2])

    
if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser(description="Image Sentiment Classification")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--test', action='store_true',default=True,
                        dest='test', help='Input --infer to Infer')
    parser.add_argument('--test_data_path', type=str,
                        default=sys.argv[1], dest='test_data_path',
                        help='Path to testing data')
    parser.add_argument('--results_dir', type=str,
                        default=sys.argv[2], dest='outputs_dir',
                        help='prediction path')
    args = parser.parse_args()
    '''
    main()

