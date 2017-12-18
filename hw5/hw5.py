# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 22:11:13 2017

@author: zhewei
"""

import keras
import keras.backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Merge, Dense, Dropout, Activation, Flatten, Concatenate,Input,Dot,Add
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.regularizers import l1, l2
import csv
import sys
import pandas as pd
import numpy as np

NUM_EPOCHS = 70
BATCH_SIZE = 1024
SPLIT_RATIO = 0.1
LATENT_DIM = 64

def load_train(filename):
    print("Loading training file ...")
    df_train = pd.read_csv(filename)
    users = df_train['UserID'].values
    movies = df_train['MovieID'].values
    rating = df_train['Rating'].values 
    n_users = np.max(df_train['UserID']) + 1
    n_movies = np.max(df_train['MovieID']) + 1
    return [users, movies], rating, n_users, n_movies

def normalize(X):
    mu = (sum(X) / len(X))
    sigma = np.std(X)
    X = (X - mu) / sigma
    return X, sigma, mu

def load_test(filename):
    print("Loading testing file ...")
    df_test = pd.read_csv(filename)
    users = df_test['UserID'].values
    movies = df_test['MovieID'].values
    return [users, movies]

def random_shuffle(x_train, y_train):
    x_train, y_train =np.transpose(np.array(x_train)), np.array(y_train)
    random_num = np.arange(len(x_train)) 
    np.random.shuffle(random_num)
    return x_train[random_num], y_train[random_num]

def train_test_split(x_train, y_train, split_ratio=SPLIT_RATIO):
    print("Split the validation data ...")
    x_train, y_train = random_shuffle(x_train, y_train)
    take_num = int(len(x_train) * split_ratio)
    x_train, x_valid = x_train[0:-take_num][:], x_train[-take_num:][:]
    y_train, y_valid = y_train[0:-take_num], y_train[-take_num:]
    return x_train, x_valid, y_train, y_valid

def build_MF_model(n_users, n_movies):
    print('Building MF_model ...')
    user_input = Input(shape = (1,))
    user_vec = Embedding(n_users, LATENT_DIM, embeddings_initializer='random_normal', embeddings_regularizer=l2(0.0))(user_input)
    user_vec = Flatten()(user_vec)
    user_vec = BatchNormalization()(user_vec)
    user_vec = Dropout(0.5)(user_vec)

    user_bias = Embedding(n_users, 1, embeddings_initializer='random_normal')(user_input)
    user_bias = Flatten()(user_bias)
    #user_bias = BatchNormalization()(user_bias)
    #user_bias = Dropout(0.5)(user_bias)

    movie_input =Input(shape = (1,))
    movie_vec = Embedding(n_movies, LATENT_DIM, embeddings_initializer='random_normal', embeddings_regularizer=l2(0.0))(movie_input)
    movie_vec = Flatten()(movie_vec)
    movie_vec = BatchNormalization()(movie_vec)
    movie_vec = Dropout(0.5)(movie_vec)

    movie_bias = Embedding(n_movies, 1, embeddings_initializer='random_normal')(movie_input)
    movie_bias = Flatten()(movie_bias)
    #movie_bias = BatchNormalization()(movie_bias)
    #movie_bias = Dropout(0.5)(movie_bias)

    r_hat = Dot(axes = 1)([user_vec, movie_vec])
    r_hat = Add()([r_hat,user_bias,movie_bias])
    model = keras.models.Model([user_input, movie_input], r_hat)
    
    return model

def train_MF_model(x_train, y_train, model, save_name):
    X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train, SPLIT_RATIO)

    checkpoint = ModelCheckpoint(filepath=save_name,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             save_weights_only=True)
    
    print("Model compiling ...")
    model.compile(loss='mse', optimizer='adam')
    
    print("Model training ...")
    history = model.fit([X_train[:,0], X_train[:,1]], y_train, 
                        batch_size=BATCH_SIZE, 
                        epochs=NUM_EPOCHS,
                        validation_data=([X_valid[:,0], X_valid[:,1]], y_valid),
                        callbacks=[checkpoint])
    
    y_pred = model.predict([X_train[:,0], X_train[:,1]]).flatten()
    rmse = np.sqrt(np.mean(np.square(y_train-y_pred)))
    print('RMSE on training data = ', rmse)

    y_pred = model.predict([X_valid[:,0], X_valid[:,1]]).flatten()
    rmse = np.sqrt(np.mean(np.square(y_valid-y_pred)))
    print('RMSE on validation data =', rmse)
        
    print("Model saved as %s" %(save_name))
    model.save(save_name)

def get_result(X_test, sigma, mu, outfile, model_name):
    print("Loading model ...")
    model = load_model(model_name)
    
    print("Predict the outcome ...")
    X_test = np.transpose(X_test)
    y_test = model.predict([X_test[:,0], X_test[:,1]]).flatten()
    result = []
    for i in range(len(y_test)):
        #result.append(y_test[i])
        result.append(y_test[i]*sigma + mu)
    
    print("Write to %s" %(outfile))
    with open(outfile, 'w', newline='') as fout:
        w = csv.writer(fout)
        w.writerow(["TestDataID","Rating"])
        index = 1
        for element in result:
            w.writerow([str(index), str(element)])
            index += 1
        
    
    
def main():
    #x_train, y_train, n_users, n_movies = load_train('train.csv')
    #y_train, sigma, mu  = normalize(y_train)
    #MF_model = build_MF_model(n_users, n_movies)
    #train_MF_model(x_train, y_train, MF_model, 'model.h5')
        
    # For testing (parameter are calculted from train.csv)
    sigma = 1.116897661146206
    mu = 3.5817120860388076
    x_test = load_test(sys.argv[1])
    get_result(x_test, sigma, mu, sys.argv[2], 'model.h5')
    
if __name__ == '__main__':
    main()
    
    

