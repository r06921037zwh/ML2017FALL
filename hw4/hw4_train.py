# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 07:49:20 2017

@author: zhewei
"""
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.core import Activation, Dense, Dropout
from keras.layers import LSTM 
from keras.models import Sequential
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
from collections import Counter
import multiprocessing 
import numpy as np
import pickle
import sys
import csv

DICT_SIZE = 5000
MAX_SENTENCE_LENGTH = 30

EMBEDDING_SIZE = 128
BATCH_SIZE = 64
NUM_EPOCHS = 4


def read_train(filename):
    print ('Loading data...')
    with open(filename, 'r', encoding='utf8') as train_data:
        x_train = []
        y_train = []
        for line in train_data:
            label, sentence = line.strip().split(sep = "+++$+++")
            y_train.append(label)
            x_train.append(sentence)
    return x_train, y_train

def read_nolabel_train(filename):
    print ("Loading unlabeled data ...")
    with open(filename, 'r', encoding='utf8') as train_data:
        x_train = []
        for line in train_data:
            x_train.append(line.strip())
    return x_train

def save_corpus(corpus, filename):
    print("Saving the corpus as %s" %(filename))
    with open(filename, 'w', encoding='utf8') as f:
        for line in corpus:
            f.write(str(line) + '\n')

def build_w2v_model(corpus):
    print("Building word2vec model ...")
    sentences = word2vec.LineSentence('corpus.txt')
    cell = multiprocessing.cpu_count()
    w2v_model = word2vec.Word2Vec(sentences, size=EMBEDDING_SIZE, window=5, min_count=1, workers=cell)
    print("Saving the word2vec model ...")
    w2v_model.save("w2v.model")
    
def make_dict(num_of_words):
    model = word2vec.Word2Vec.load('w2v.model') 
    text = []
    with open('corpus.txt', 'r', encoding='utf8') as fin:
        for line in fin:
            text.extend(line.strip().split())
    counter = Counter(text).most_common(num_of_words)
    vocab_list = [w for w, _ in counter]
    dictionary = {}
    for word in vocab_list:
        dictionary[str(word)] = model[str(word)]
    print("Saving the dictionary ...")
    pickle.dump(dictionary, open('dictionary', 'wb'))
    return dictionary
    
def word_embedding(x, dict_name):
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

def make_BOW(x_train, token):
    print("Making Bag of Words ...") 
    x_BOW = token.texts_to_matrix(x_train, mode='count')        
    return x_BOW
  
def build_model():  
    print("Build LSTM(RNN) model ...")
    model = Sequential()
    #model.add(Embedding(input_dim=DICT_SIZE, output_dim=EMBEDDING_SIZE, input_length=MAX_SENTENCE_LENGTH))

    model.add(LSTM(128, return_sequences=True, input_shape=(MAX_SENTENCE_LENGTH, EMBEDDING_SIZE)))
    model.add(Dropout(0.2))
    
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.2))
    
    model.add(LSTM(32))
    model.add(Dropout(0.2))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    print (model.summary())   
    return model

def Model_train(x_train, y_train, model, save_name):
    #split the validation data 
    Xtrain, Xtest, ytrain, ytest = train_test_split(x_train, y_train, test_size=0.05, random_state=37)
    #earlystopping = EarlyStopping(monitor='val_acc', patience = 3, verbose=1, mode='max')
    checkpoint = ModelCheckpoint(filepath=save_name, 
                                     verbose=0,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_acc',
                                     mode='max' )
    
    print ("Train the model & save it...")

    history = model.fit(Xtrain, ytrain, batch_size=BATCH_SIZE, 
                        epochs=NUM_EPOCHS,
                        validation_data=(Xtest, ytest),
                        callbacks=[checkpoint])
    
    score,	acc	=	model.evaluate(Xtest,	ytest,	batch_size=BATCH_SIZE)
    print("Test	_Loss:	%.3f,	accuracy:%.3f"	%	(score,	acc))
    
    print("Model saved as %s" %(save_name))
    model.save(save_name)

'''
def read_test(filename):
    print("Load the test data ...")
    with open(filename,'r', encoding='utf8') as test_data:
        x_test = []
        next(test_data)
        for line in test_data:
            index, sentence = line.strip().split(",",1)
            x_test.append(sentence)
    return x_test
'''

def get_result(predict_class, outputs_dir):
    with open(outputs_dir, 'w', newline='') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow(['id', 'label'])
        index = 0
        for element in predict_class:
            writer.writerow([str(index), str(int(element))])
            index += 1
            
def get_ensemble_result(predict_prob, outputs_dir):
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
    # read in training data
    x_train, y_train = read_train(sys.argv[1])
    x_train_nolabel = read_nolabel_train(sys.argv[2])
    #x_test = read_test('testing_data.txt')  
    # fuse the corpus
    corpus = x_train + x_train_nolabel 
    
    # save the corpus 
    save_corpus(corpus, 'corpus.txt')
    
    # build the w2v_model
    build_w2v_model(corpus)
    
    # make dictionary in num_of_words
    dictionary = make_dict(num_of_words=35000)
     
    # word-embedding the training data
    X_train = word_embedding(x_train, dict_name='dictionary')
       
    model = build_model()
    Model_train(X_train, y_train, model, 'model.h5')

if __name__ == '__main__':
    main()




    