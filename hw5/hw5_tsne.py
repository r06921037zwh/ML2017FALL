# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 22:11:13 2017

@author: zhewei
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter
from keras.models import load_model

def movie_classify():
    print("Classify the Movies ...")
    with open('movies.csv', 'r', encoding='latin-1') as fin:
        movie_class = np.zeros(3953)
        next(fin)
        for line in fin:
            line = line.strip().split(':')
            ty = line[-1].split('|')
            count = np.zeros(6)
            counter = list(Counter(ty).keys())
            
            if 'Animation' in counter:
                count[0] = count[0] + 1
            if 'Children \'s' in counter:
                count[0] = count[0] + 1            
            if 'Comedy' in counter:
                count[0] = count[0] + 1            
            
            if 'Crime' in counter:
                count[1] = count[1] + 1
            if 'Thriller' in counter:
                count[1] = count[1] + 1
            if 'Horror' in counter:
                count[1] = count[1] + 1 
            if 'Film-Noir' in counter:
                count[1] = count[1] + 1
                
            if 'Mystery' in counter:
                count[2] = count[2] + 1            
            if 'Fantasy' in counter:
                count[2] = count[2] + 1
            if 'Sci-Fi' in counter:
                count[2] = count[2] + 1
                                                                             
            if 'Musical' in counter:
                count[3] = count[3] + 1
            if 'Drama' in counter:
                count[3] = count[3] + 1
                
            if 'Adventure' in counter:
                count[4] = count[4] + 1
            if 'Action' in counter:
                count[4] = count[4] + 1
            if 'Western' in counter:
                count[4] = count[4] + 1    
            
            if 'War' in counter:
                count[5] = count[5] + 1       
            if 'Documentary' in counter:
                count[5] = count[5] + 1
            if 'Romance' in counter:
                count[5] = count[5] + 1
                      
            movie_class[int(line[0]) - 1] = int(np.argmax(count))
        return movie_class
    
def plot_embedding(x, y):
    print("Plotting the embedding Layer T-SNE ...")
    cm = plt.cm.get_cmap('RdYlGn')
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40, c=y, cmap=cm)
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('on')
    ax.axis('tight')
    plt.show()
               
def main():
    model = load_model('model7.h5')
    embeddings = model.get_layer(name='embedding_15').get_weights()
    nsamples, nx, ny = np.array(embeddings).shape
    d2_embeddings = np.array(embeddings).reshape((nx,ny))
    tsne = TSNE(n_components=2, random_state=0, verbose=1)
    transformed_weights = tsne.fit_transform(d2_embeddings)
    movie_class = movie_classify()
    plot_embedding(transformed_weights, movie_class)
    
if __name__ == '__main__':
    main()
    
       






    



       

    
