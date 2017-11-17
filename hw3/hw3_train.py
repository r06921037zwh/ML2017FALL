# -*- coding: utf-8 -*-
"""
Spyder Editor
-- zhewei hsu---
"""

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import RMSprop, SGD, Adam
from keras.layers import Conv2D, MaxPooling2D, Flatten, ZeroPadding2D, Input
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.data_utils import get_file
from keras.models import load_model

import numpy as np
import csv
import sys
import argparse
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

####################################################################################
#-----------------------------------for training-----------------------------------#
####################################################################################

def load_train(train_data_path):
    with open(train_data_path,'r') as train_csv:
        train = csv.reader(train_csv)
        train_x = []
        train_y = []
        next(train)      #from the second line
        for row in train:
            train_y.append(row[0])
            t = [float(i)/255 for i in row[1].split()]
            train_x.append(t)

    return (train_x, train_y)


def split_valid_set(train_x, train_y, valid_per=0.1):        
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    
    #split for validation
    train_num = train_y.shape[0]
    random_num = np.arange(train_num)
    np.random.shuffle(random_num)
    
    validation_perc = valid_per                             #percentage of validation_set
    ind_TS = random_num[0:-int(train_num*validation_perc)]  #train_set index
    ind_VS = random_num[-int(train_num*validation_perc):]   #validation_set index
    x_train_set = np.array(train_x[ind_TS])
    x_validation_set = np.array(train_x[ind_VS])
    y_train_set = np.array(train_y[ind_TS])
    y_validation_set = np.array(train_y[ind_VS])
    
    #convert label to one-hot encoding
    y_train_set = np_utils.to_categorical(y_train_set, num_classes=7)
    y_validation_set = np_utils.to_categorical(y_validation_set, num_classes=7)

    return (x_train_set, y_train_set, x_validation_set, y_validation_set)    

def build_model_CNN():
    model = Sequential()
    
    model.add(Conv2D(filters = 32,kernel_size = (3,3),padding = 'same',input_shape=(48,48,1), activation='relu'))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters = 32,kernel_size = (3,3),padding = 'same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters =64,kernel_size = (3,3),padding='same',activation='relu'))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters =64,kernel_size = (3,3),padding='same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters = 128,kernel_size =(3,3),padding = 'same',activation='relu'))
    model.add(BatchNormalization())
    #model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(filters = 128,kernel_size =(3,3),padding = 'same',activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))
    model.add(Dropout(0.25))
        
    model.add(Flatten())
    
    model.add(Dense(units=512,activation='relu', kernel_initializer='RandomNormal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(units=256,activation='relu', kernel_initializer='RandomNormal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(units=7,activation='softmax'))
    model.summary()
    
    # opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    opt = Adam(lr=1e-3)
    # opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])
    return model
'''
def build_model_DNN():
    model = Sequential()
    
    model.add(Dense(units=32, input_dim=2304, activation='relu', kernel_initializer='RandomNormal'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.25))
    
    model.add(Dense(units=32, activation='relu', kernel_initializer='RandomNormal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Dense(units=64, activation='relu', kernel_initializer='RandomNormal'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.25))
    
    model.add(Dense(units=64, activation='relu', kernel_initializer='RandomNormal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
    
    model.add(Dense(units=128, activation='relu', kernel_initializer='RandomNormal'))
    model.add(BatchNormalization())
    #model.add(Dropout(0.25))
    
    model.add(Dense(units=128, activation='relu', kernel_initializer='RandomNormal'))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))
        
    #model.add(Dense(units=512,activation='relu', kernel_initializer='RandomNormal'))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.3))
    
    #model.add(Dense(units=256,activation='relu', kernel_initializer='RandomNormal'))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.3))
    
    model.add(Dense(units=7,activation='softmax'))
    model.summary()
    
    # opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    opt = Adam(lr=1e-3)
    # opt = Adadelta(lr=0.1, rho=0.95, epsilon=1e-08)
    model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])
    return model

def train_DNN(train_x, train_y, save_model_path, batch_size=32, epoch=100):
    x_train, y_train, x_validation, y_validation = split_valid_set(train_x, train_y, valid_per=0.1)  
    #train_x = train_x.reshape(len(train_x),2304).astype('float32')
    # construct model
    model = build_model_DNN()
    
    # Fit the model on the batches generated by datagen.flow().
    train_history = model.fit(x=train_x, y=train_y, validation_split=0.1, epochs=200, batch_size = 50, verbose=2)
    
    #print(train_history.history.keys())
    #show_train_history(train_history, 'acc', 'val_acc')
    score = model.evaluate(train_x, train_y)
    
    print('\nTrain Acc:', score[1])
    
    model.save(save_model_path)
    return train_history
'''

def train(train_x, train_y, save_model_path, batch_size, epoch):
    x_train, y_train, x_validation, y_validation = split_valid_set(train_x, train_y, valid_per=0.1)  
      
    x_train = x_train.reshape(x_train.shape[0],48,48,1)
    x_validation = x_validation.reshape(x_validation.shape[0],48,48,1)
      
    datagen = ImageDataGenerator(
            featurewise_center=False,               # set input mean to 0 over the dataset
            samplewise_center=False,                # set each sample mean to 0
            featurewise_std_normalization=False,    # divide inputs by std of the dataset
            samplewise_std_normalization=False,     # divide each input by its std
            zca_whitening=False,                    # apply ZCA whitening
            rotation_range=10,                      # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,                  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,                 # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,                   # randomly flip images
            zoom_range = 0.05,
            vertical_flip=False)                    # randomly flip images
    
    datagen.fit(x_train)
    
    # construct model
    model = build_model_CNN()
    
    # Fit the model on the batches generated by datagen.flow().
    train_history = model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),
                                    steps_per_epoch=len(x_train) // batch_size,
                                    epochs=epoch,
                                    verbose=1,
                                    validation_data=(x_validation,y_validation))
    #
    #print(train_history.history.keys())
    #show_train_history(train_history, 'acc', 'val_acc')
    score = model.evaluate(x_train,y_train)
    
    print('\nTrain Acc:', score[1])
    
    model.save(save_model_path)
    return train_history

'''
def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title(['Train History'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    fig = plt.gcf()
    plt.show()
    fig.savefig("report_p1.png", dpi=300)
'''

def main():
    save_model_path = 'hw3_model.h5'
    train_x, train_y = load_train(sys.argv[1])
    history = train(train_x, train_y, save_model_path, batch_size=200, epoch=1)
    #show_train_history(history, 'acc', 'val_acc')
    
if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser(description="Image Sentiment Classification")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true', default=True,
                        dest='train', help='Input --train to Train')
    parser.add_argument('--save_model_path', type=str,
                        default='hw3_model.h5', dest='save_model_path',
                        help='saved model path')
    args = parser.parse_args()
    '''
    main()

