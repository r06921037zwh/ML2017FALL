# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 08:44:08 2017

@author: zhewei
"""

from keras.models import load_model
import numpy as np
import pickle
import sys
import csv

MAX_SENTENCE_LENGTH = 30
EMBEDDING_SIZE = 128

def read_test(filename):
    print("Load the test data ...")
    with open(filename,'r', encoding='utf8') as test_data:
        x_test = []
        next(test_data)
        for line in test_data:
            index, sentence = line.strip().split(",",1)
            x_test.append(sentence)
    return x_test


def word_embedding(x, dict_name):
    print("Word-embedding ...")
    dictionary = pickle.load(open(dict_name, 'rb'))
    x_token = []
    for line in x:
        x_token.append(line.strip().split())
      
    X = np.zeros((len(x), MAX_SENTENCE_LENGTH, EMBEDDING_SIZE))
    for idx, sentence in enumerate(x_token):
        for jdx, word in enumerate(sentence):
            if jdx == MAX_SENTENCE_LENGTH:
                break            
            else:
                if word in dictionary:
                    X[idx, jdx, :] = dictionary[word]
    return X

def get_result(predict_class, outputs_dir):
    print("Get results ...")
    with open(outputs_dir, 'w', newline='') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow(['id', 'label'])
        index = 0
        for element in predict_class:
            writer.writerow([str(index), str(int(element))])
            index += 1
            
def get_ensemble_result(predict_prob, outputs_dir):
    print("Get ensemble results ...")
    result = []
    for i in range(len(predict_prob)):
        if predict_prob[i] >= 0.5:
            result.append(int(1))
        else:
            result.append(int(0))
        
    with open(outputs_dir, 'w', newline='') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow(['id', 'label'])
        index = 0
        for element in result:
            writer.writerow([str(index), str(int(element))])
            index += 1

def main():
    x_test = read_test(sys.argv[1])
    X_test = word_embedding(x_test, dict_name='dictionary')
    model1 = load_model("model1.h5")
    model2 = load_model("model2.h5")
    model3 = load_model("model3.h5")
    
    print("predict using the model1 ...")
    prediction1 = model1.predict(X_test)
    print("predict using the model2 ...")
    prediction2 = model2.predict(X_test)
    print("predict using the model3 ...")
    prediction3 = model3.predict(X_test)
    
    prediction = (prediction1 + prediction2 + prediction3) / 3
    get_ensemble_result(prediction, sys.argv[2])
    
if __name__ == '__main__':
    main()
