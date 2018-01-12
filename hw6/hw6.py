# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 20:35:16 2017

@author: zhewei
"""

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam
from keras.models import Model, load_model
import csv
import pandas as pd
import sys

def write_ans(test, label, outputs_dir):   
    print('Writing Answer ...')
    result = []
    for i in range(len(test)):
        if label[test[i,0]] == label[test[i,1]]:
            result.append(1)
        else:
            result.append(0)
    with open(outputs_dir, 'w', newline='') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow(['ID', 'Ans'])
        index = 0
        for element in result:
            writer.writerow([str(index), str(int(element))])
            index += 1
                
def read_test(testfile):
    print('Reading Testing File ...')
    df = pd.read_csv('test_case.csv')
    img1 = df['image1_index']
    img2 = df['image2_index']
    test = np.vstack((np.array(img1), np.array(img2))).T
    return test    
'''
def random_shuffle(x_train):
    random_num = np.arange(len(x_train)) 
    np.random.shuffle(random_num)
    return x_train[random_num]

def autoencoder(img, split_ratio):
    # Encoder   
    input_img = Input(shape = (784, ))    
    #encoded = Dense(256, activation = 'relu')(input_img)
    encoded = Dense(128, activation = 'relu')(input_img)    
    encoded = Dense(32, activation = 'relu')(encoded)
    # Decoder
    #decoded = Dense(32, activation = 'relu')(encoded)    
    decoded = Dense(128, activation = 'relu')(encoded)
    #decoded = Dense(256, activation = 'relu')(decoded)
    decoded = Dense(784, activation = 'sigmoid')(decoded)
    
    # Image Compression using encoder
    encoder = Model(input_img, output = encoded)
    # Training using autoencoder
    autoencoder = Model(input_img, output = decoded)
    # Compile and training
    autoencoder.compile(optimizer = 'adam', loss = 'binary_crossentropy')
    history = autoencoder.fit(X_train, X_train, validation_split=split_ratio, epochs = 550,batch_size = 256)
       
    encoder.save('model')
    return encoder
'''
def main(): 
    img = np.load(sys.argv[1])
    X_train = img / 256
    #encoder = autoencoder(X_train, 0.05)
    encoder = load_model('model')
    encoded_img = encoder.predict(X_train)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_img)
    label = kmeans.labels_
    test = read_test(sys.argv[2])
    write_ans(test, label, sys.argv[3])
    
    
if __name__ == '__main__':
    main()