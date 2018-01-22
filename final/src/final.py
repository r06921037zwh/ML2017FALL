# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 23:38:19 2017

@author: zhewei
"""

from gensim.models import word2vec
from numpy.linalg import norm
from collections import Counter
import multiprocessing 
import numpy as np
import pandas as pd
import pickle
import os
import csv
import jieba
jieba.set_dictionary('dict.txt.big.txt')

DICT_SIZE = 30000
EMBEDDING_SIZE = 1024


def read_file():
    print ('Loading training data ...')
    lines = []
    names = ['1', '2', '3', '4', '5']
    for item in names:
        filename = item + '_train.txt'
        print(filename)
        with open(filename, 'r', encoding='utf8') as train_data:
            for line in train_data:
                line = line.strip()
                lines.append(line)
    
    print("Loading stop words ...")
    stopwords = []
    with open('stopwords.txt', 'r', encoding='utf8') as SW:
        for word in SW:
            stopwords.append(word)
    return lines, stopwords

def preprocess(sentences):
    container_w2v = []
    container_dic = []
    for line in sentences:
        words = jieba.cut(line, cut_all=False)
        temp = []
        for word in words:
            temp.append(word)
        container_w2v.append(temp)
        container_dic.extend(temp)
    return container_w2v, container_dic


def build_w2v_model(container_w2v, model_index):
    print("Building word2vec model ...")
    cell = multiprocessing.cpu_count()
    w2v_model = word2vec.Word2Vec(container_w2v, size=EMBEDDING_SIZE, 
                                  window=7, min_count=1, workers=cell,
                                  iter=50, sg=1)
    
    print("Saving the word2vec model ...")
    save_place = path_pos(model_index,'False', 'True')
    w2v_model.save(save_place)
   
def make_dict(container_dic, stopwords, freq, model_index):
    load_place = path_pos(model_index,'False', 'True')
    model = word2vec.Word2Vec.load(load_place)  
    counter = Counter(container_dic).most_common(50000)
    vocab_list = [w for w, _ in counter]
    # ref https://openreview.net/pdf?id=SyK00v5xx
    occur_freq = [_ for w, _ in counter]
    vocab_prob = np.array(occur_freq) / float(sum(occur_freq))
    alpha = 0.05
    dictionary = {}    
    index = 0
    for word in vocab_list:
        if word in model:
            dictionary[str(word)] = model[str(word)] * (alpha/(alpha + vocab_prob[index]))
            index = index + 1
        if(index > freq):
            break          
    print("Saving the dictionary ...")
    save_place = path_pos(model_index,'True', 'False')
    path = os.path.join(save_place,'dictionary')
    pickle.dump(dictionary, open(path, 'wb'))
    return dictionary


'''
def make_dict(container_dic, stopwords, freq, model_index):
    load_place = path_pos(model_index,'False', 'True')
    model = word2vec.Word2Vec.load(load_place) 
    counter = Counter(container_dic).most_common(freq)
    vocab_list = [w for w, _ in counter]
    dictionary = {}
    for word in vocab_list:
        dictionary[str(word)] = model[str(word)]
        
    print("Saving the dictionary ...")
    save_place = path_pos(model_index,'True', 'False')
    path = os.path.join(save_place,'dictionary')
    pickle.dump(dictionary, open(path, 'wb'))
    return dictionary
'''
    
def read_test():
    print("Loading testing file ... ")
    df = pd.read_csv('testing_data.csv', dtype=str)
    df_question = df['dialogue'].values
    df_answer = df['options'].values
    
    i = 0
    questions = []
    answers = []
    for item in range(5060):
        q_tmp = split_header(str(df_question[i]), 'question')
        questions.append(q_tmp)
        a_tmp = split_header(str(df_answer[i]), 'answer')
        answers.append(a_tmp)
        i = i + 1  
    
    print("Spliting the questions with jieba ...")
    q_split = []    
    for line in questions:
        words = jieba.cut(line, cut_all=False)
        temp = []
        for word in words:
            word = word.strip()
            if word != '':
                temp.append(word)
        q_split.append(temp)
    
    print("Spliting the answers with jieba ...")
    a_split = []
    for lines in answers:
        ans = []
        for line in lines:
            words = jieba.cut(line, cut_all=False)
            temp = []
            for word in words:
                word = word.strip()
                if word != '':
                    temp.append(word)
            ans.append(temp)
        a_split.append(ans)
        
    return q_split, a_split
        
def split_header(sent, sent_type):
    sent = sent.replace('A:','::')
    sent = sent.replace('B:','::')
    sent = sent.replace('C:','::')
    sent = sent.replace('D:','::')
    sent = sent.replace('E:','::')
    sent = sent.replace('F:','::')
    if sent_type == 'question':
        sent = sent.replace('::','')
    else:
        sent = sent.split('::')[1:]
        tmp = []
        for line in sent:
            line = line.strip()
            tmp.append(line)
        sent = tmp
    return sent

def word_to_vec(questions, answers, dictionary):
    print("Transforming questions into vectors ... ")
    q_vec = []
    for line in questions:
        temp = []
        for word in line:
            if word in dictionary:
                temp.append(dictionary[word])
            else:
                s = np.random.uniform(-0.3,0.3, EMBEDDING_SIZE)
                temp.append(s)
        q_vec.append(temp)
    
    print("Transforming answers into vectors ... ")
    a_vec = []
    for lines in answers:
        _ans = []
        for line in lines:
            temp = []
            for word in line:
                if word in dictionary:
                    temp.append(dictionary[word])
                else:
                    s = np.random.uniform(-0.3,0.3, EMBEDDING_SIZE)
                    temp.append(s)
            _ans.append(temp)
        a_vec.append(_ans)
        
    return q_vec, a_vec                 

def cal_dist(q_vec, a_vec):
    print("Calculating mean vectors in questions ... ")
    q_mean_vec = []
    #counter = []
    for line in q_vec:
        count = len(line)
        #counter.append(count)
        vec = np.zeros(EMBEDDING_SIZE)
        if count != 0:                        
            for v in line:
                vec = vec + v
            vec = vec / count
            q_mean_vec.append(vec)
        else: 
            q_mean_vec.append(vec)
    
    print("Calculating mean vectors in answers ... ")  
    a_mean_vec = []
    for lines in a_vec:
        temp = []
        for line in lines:
            count = len(line)
            vec = np.zeros(EMBEDDING_SIZE)
            if count != 0:                        
                for v in line:
                    vec = vec + v
                vec = vec / count
                temp.append(vec)
            else: 
                temp.append(vec)
        a_mean_vec.append(temp)
    
    print("Calculating cos-sim between question & answers ... ")
    dist = []
    i = 0
    for vec_a in q_mean_vec:
        temp = []
        vec = a_mean_vec[i]
        for vec_b in vec:
            if norm(vec_a) != 0 and norm(vec_b) != 0 :
                d = np.inner(vec_a, vec_b)/(norm(vec_a) * norm(vec_b))
            else:
                d = 0
            temp.append(d)
        dist.append(temp)
        i = i + 1
    return dist

def write_ans(dist, outputs_dir):
    print("Get results ...")
    result = []
    for i in range(len(dist)):
        result.append(np.argmax(dist[i]))
        
    with open(outputs_dir, 'w', newline='') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow(['id', 'ans'])
        index = 1
        for element in result:
            writer.writerow([str(index), str(int(element))])
            index += 1

def write_pos(dist, outputs_dir):
    with open(outputs_dir, 'w', newline='') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow(['id', 'option0','option1','option2','option3','option4','option5' ])
        index = 1
        for row in dist:
            writer.writerow([str(index), str(float(row[0])), str(float(row[1])), str(float(row[2])), str(float(row[3])), str(float(row[4])), str(float(row[5]))])
            index += 1
                  
def path_pos(index, directory, model):
    if directory == 'True':
        result = 'w2v_model_' + str(index)
        return result
    
    elif model =='True':
        result = os.path.join(('w2v_model_' + str(index)), ('w2v_model_' + str(index)))
        return result
    
def main():
    # train the word2vec
    sentences, stopwords = read_file()
    container_w2v, container_dic = preprocess(sentences)
    
    # read in testfile
    questions, answers = read_test() 
    #build_w2v_model(container_w2v, model_index)
    
    # model 2 
    model_index = 2
    dictionary = make_dict(container_dic, stopwords, DICT_SIZE, model_index)
    q_vec, a_vec = word_to_vec(questions, answers, dictionary)   
    dist1 = np.array(cal_dist(q_vec, a_vec))
    
    # model 3 
    model_index = 3
    dictionary = make_dict(container_dic, stopwords, DICT_SIZE, model_index)
    q_vec, a_vec = word_to_vec(questions, answers, dictionary)   
    dist2 = np.array(cal_dist(q_vec, a_vec))
    
    # model 4 
    model_index = 4
    dictionary = make_dict(container_dic, stopwords, DICT_SIZE, model_index)
    q_vec, a_vec = word_to_vec(questions, answers, dictionary)   
    dist3 = np.array(cal_dist(q_vec, a_vec))

    dist = dist1 + dist2 + dist3
    write_ans(dist, 'Prediction.csv')
    
if __name__ == '__main__':
    main()
    