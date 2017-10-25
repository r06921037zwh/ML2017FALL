import os, sys
import pandas as pd
import numpy as np
from random import shuffle
import argparse
from math import log, floor

# If you wish to get the same shuffle result
# np.random.seed(2401)

def load_data(train_data_path, train_label_path, test_data_path):
    X_train = pd.read_csv(train_data_path, sep=',', header=0)
    #X_train = np.array(retrieve_Feature(X_train))
    Y_train = pd.read_csv(train_label_path, sep=',', header=0)
    #Y_train = np.array(retrieve_Feature(Y_train))
    X_test = pd.read_csv(test_data_path, sep=',', header=0)
    #X_test = np.array(retrieve_Feature(X_test)) 
    X_train, Y_train, X_test = retrieve_Feature(X_train, Y_train, X_test)
    return (X_train, Y_train, X_test)

def retrieve_Feature(X_train, Y_train, X_test):
      # marriage status
      X_train['Married'] = X_train[' Husband'] + X_train[' Wife']
      X_test['Married'] = X_test[' Husband'] + X_test[' Wife']
      #X_train['Not-In-Marriage'] = X_train[' Divorced']+X_train[' Never-married']+X_train[' Separated']+X_train[' Widowed']
      #X_test['Not-In-Marriage'] = X_test[' Divorced']+X_test[' Never-married']+X_test[' Separated']+X_test[' Widowed']
      del X_train[' Husband'];  del X_test[' Husband']
      del X_train[' Wife'];  del X_test[' Wife']
      del X_train[' Married-spouse-absent'];  del X_test[' Married-spouse-absent']
      del X_train[' Married-civ-spouse'];  del X_test[' Married-civ-spouse'];
      del X_train[' Married-AF-spouse'];  del X_test[' Married-AF-spouse']
      del X_train[' Divorced'];  del X_test[' Divorced']
      del X_train[' Never-married'];  del X_test[' Never-married']
      del X_train[' Separated'];  del X_test[' Separated']
      del X_train[' Widowed'];  del X_test[' Widowed']
      
      
      #workclass
      X_train['No-working'] = X_train[' Without-pay'] + X_train[' Never-worked']
      X_train['Manual-Labor'] = X_train[' Farming-fishing']+X_train[' Handlers-cleaners']+X_train[' Transport-moving']
      #X_train['Service'] = X_train[' Other-service'] + X_train[' Priv-house-serv']
      X_test['No-working'] = X_test[' Without-pay'] + X_test[' Never-worked']
      X_test['Manual-Labor'] = X_test[' Farming-fishing']+X_test[' Handlers-cleaners']+X_test[' Transport-moving']
      #X_test['Service'] = X_test[' Other-service'] + X_test[' Priv-house-serv']
      #del X_train[' Without-pay']; del X_test[' Without-pay']
      #del X_train[' Never-worked']; del X_test[' Never-worked']
      #del X_train[' Farming-fishing'];  del X_test[' Farming-fishing']
      #del X_train[' Handlers-cleaners'];  del X_test[' Handlers-cleaners']
      #del X_train[' Transport-moving'];  del X_test[' Transport-moving']
      #del X_train[' Other-service'];  del X_test[' Other-service']
      #del X_train[' Priv-house-serv'];  del X_test[' Priv-house-serv']

      #country
      X_train['SA'] = X_train[' Columbia']+X_train[' Cuba']+X_train[' Dominican-Republic']+X_train[' Ecuador']+X_train[' El-Salvador']+X_train[' Guatemala']+X_train[' Haiti']+X_train[' Honduras']+X_train[' Jamaica']+X_train[' Mexico']+X_train[' Nicaragua']+X_train[' Peru']+X_train[' Puerto-Rico']+X_train[' Trinadad&Tobago']
      X_test['SA'] = X_test[' Columbia']+X_test[' Cuba']+X_test[' Dominican-Republic']+X_test[' Ecuador']+X_test[' El-Salvador']+X_test[' Guatemala']+X_test[' Haiti']+X_test[' Honduras']+X_test[' Jamaica']+X_test[' Mexico']+X_test[' Nicaragua']+X_test[' Peru']+X_test[' Puerto-Rico']+X_test[' Trinadad&Tobago']
      
      X_train['EU1'] = X_train[' France']+X_train[' Germany']+X_train[' Holand-Netherlands']+X_train[' England']
      X_test['EU1'] = X_test[' France']+X_test[' Germany']+X_test[' Holand-Netherlands']+X_test[' England']
      
      del X_train[' Columbia'];  del X_test[' Columbia']
      del X_train[' Cuba'];  del X_test[' Cuba']
      del X_train[' Dominican-Republic'];  del X_test[' Dominican-Republic']
      del X_train[' Ecuador'];  del X_test[' Ecuador']
      del X_train[' El-Salvador'];  del X_test[' El-Salvador']
      del X_train[' Guatemala'];  del X_test[' Guatemala']
      del X_train[' Haiti'];  del X_test[' Haiti']
      del X_train[' Honduras'];  del X_test[' Honduras']
      del X_train[' Jamaica'];  del X_test[' Jamaica']
      del X_train[' Mexico'];  del X_test[' Mexico']
      del X_train[' Nicaragua'];  del X_test[' Nicaragua']
      del X_train[' Peru'];  del X_test[' Peru']
      del X_train[' Puerto-Rico'];  del X_test[' Puerto-Rico']
      del X_train[' Trinadad&Tobago'];  del X_test[' Trinadad&Tobago']
      del X_train[' France'];  del X_test[' France']
      del X_train[' Germany'];  del X_test[' Germany']
      del X_train[' Holand-Netherlands'];  del X_test[' Holand-Netherlands']
      del X_train[' England'];  del X_test[' England']

      X_train = np.array(X_train)
      Y_train = np.array(Y_train)
      X_test = np.array(X_test)

      return (X_train, Y_train, X_test)

def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def normalize(X_all, X_test):
    # Feature normalization with train and test X
    X_train_test = np.concatenate((X_all, X_test))
    mu = (sum(X_train_test) / X_train_test.shape[0])
    sigma = np.std(X_train_test, axis=0)
    mu = np.tile(mu, (X_train_test.shape[0], 1))
    sigma = np.tile(sigma, (X_train_test.shape[0], 1))
    X_train_test_normed = (X_train_test - mu) / sigma

    # Split to train, test again
    X_all = X_train_test_normed[0:X_all.shape[0]]
    X_test = X_train_test_normed[X_all.shape[0]:]
    return X_all, X_test

def split_valid_set(X_all, Y_all, percentage):
    all_data_size = len(X_all)
    valid_data_size = int(floor(all_data_size * percentage))
    
    X_all, Y_all = _shuffle(X_all, Y_all)
    
    X_valid, Y_valid = X_all[0:valid_data_size], Y_all[0:valid_data_size]
    X_train, Y_train = X_all[valid_data_size:], Y_all[valid_data_size:]
    
    return X_train, Y_train, X_valid, Y_valid

def sigmoid(z):
    res = 1 / (1.0 + np.exp(-z))
    return np.clip(res, 1e-8, 1-(1e-8))

def valid(X_valid, Y_valid, mu1, mu2, shared_sigma, N1, N2):
    sigma_inverse = np.linalg.inv(shared_sigma)
    w = np.dot( (mu1-mu2), sigma_inverse)
    x = X_valid.T
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)
    a = np.dot(w, x) + b
    y = sigmoid(a)
    y_ = np.around(y)
    result = (np.squeeze(Y_valid) == y_)
    print('Valid acc = %f' % (float(result.sum()) / result.shape[0]))
    return

def train(X_all, Y_all, save_dir):
    # Split a 10%-validation set from the training set
    valid_set_percentage = 0.1
    X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, valid_set_percentage)
    
    # Gaussian distribution parameters
    train_data_size = X_train.shape[0]
    cnt1 = 0
    cnt2 = 0

    mu1 = np.zeros((84,))
    mu2 = np.zeros((84,))
    for i in range(train_data_size):
        if Y_train[i] == 1:
            mu1 += X_train[i]
            cnt1 += 1
        else:
            mu2 += X_train[i]
            cnt2 += 1
    mu1 /= cnt1
    mu2 /= cnt2

    sigma1 = np.zeros((84,84))
    sigma2 = np.zeros((84,84))
    for i in range(train_data_size):
        if Y_train[i] == 1:
            sigma1 += np.dot(np.transpose([X_train[i] - mu1]), [(X_train[i] - mu1)])
        else:
            sigma2 += np.dot(np.transpose([X_train[i] - mu2]), [(X_train[i] - mu2)])
    sigma1 /= cnt1
    sigma2 /= cnt2
    shared_sigma = (float(cnt1) / train_data_size) * sigma1 + (float(cnt2) / train_data_size) * sigma2
    N1 = cnt1
    N2 = cnt2

    print('=====Saving Param=====')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    param_dict = {'mu1':mu1, 'mu2':mu2, 'shared_sigma':shared_sigma, 'N1':[N1], 'N2':[N2]}
    for key in sorted(param_dict):
        print('Saving %s' % key)
        np.savetxt(os.path.join(save_dir, ('%s' % key)), param_dict[key])
    
    print('=====Validating=====')
    valid(X_valid, Y_valid, mu1, mu2, shared_sigma, N1, N2)

    return

def infer(X_test, save_dir, output_dir):
    # Load parameters
    print('=====Loading Param from %s=====' % save_dir)
    mu1 = np.loadtxt(os.path.join(save_dir, 'mu1'))
    mu2 = np.loadtxt(os.path.join(save_dir, 'mu2'))
    shared_sigma = np.loadtxt(os.path.join(save_dir, 'shared_sigma'))
    N1 = np.loadtxt(os.path.join(save_dir, 'N1'))
    N2 = np.loadtxt(os.path.join(save_dir, 'N2'))

    # Predict
    sigma_inverse = np.linalg.inv(shared_sigma)
    w = np.dot( (mu1-mu2), sigma_inverse)
    x = X_test.T
    b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)
    a = np.dot(w, x) + b
    y = sigmoid(a)
    y_ = np.around(y)

    print('=====Write output to %s =====' % output_dir)
    # Write output
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = os.path.join(output_dir, sys.argv[6])
    with open(output_path, 'w') as f:
        f.write('id,label\n')
        for i, v in  enumerate(y_):
            f.write('%d,%d\n' %(i+1, v))

    return

def main():
    # Load feature and label
    X_all, Y_all, X_test = load_data(sys.argv[3], sys.argv[4], sys.argv[5])
    # Normalization
    X_all, X_test = normalize(X_all, X_test)
    train(X_all,Y_all,'generative_params/')
    infer(X_test, 'generative_params/', 'generative_output/')
    ''' 
    # To train or to infer
    if opts.train:
        train(X_all, Y_all, opts.save_dir)
    elif opts.infer:
        infer(X_test, opts.save_dir, opts.output_dir)
    else:
        print("Error: Argument --train or --infer not found")
    '''
    return

if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser(
                description='Probabilistic Generative Model for Binary Classification'
             )
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true', default=False,
                        dest='train', help='Input --train to Train')
    group.add_argument('--infer', action='store_true',default=False,
                        dest='infer', help='Input --infer to Infer')
    parser.add_argument('--train_data_path', type=str,
                        default='feature/X_train', dest='train_data_path',
                        help='Path to training data')
    parser.add_argument('--train_label_path', type=str,
                        default='feature/Y_train', dest='train_label_path',
                        help='Path to training data\'s label')
    parser.add_argument('--test_data_path', type=str,
                        default='feature/X_test', dest='test_data_path',
                        help='Path to testing data')
    parser.add_argument('--save_dir', type=str, 
                        default='generative_params/', dest='save_dir',
                        help='Path to save the model parameters')
    parser.add_argument('--output_dir', type=str, 
                        default='generative_output/', dest='output_dir',
                        help='Path to save output')
    opts = parser.parse_args()
    '''
    main()
