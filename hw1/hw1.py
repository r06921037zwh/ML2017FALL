import numpy as np
from numpy.linalg import inv
import csv
import random
import sys
import math

#read model
w = np.load('model.npy')

#reading testing data
test_x = []
n_row = 0
f = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]
with open(sys.argv[1],'r') as testFile:
    testData = csv.reader(testFile, delimiter = ",")
    for r in testData:
        if n_row%18 == 0:
            test_x.append([])
        if n_row%18 in f:
            for i in range(2,11):
                if r[i] != "NR":
                    test_x[n_row//18].append(float(r[i]))
                else:
                    test_x[n_row//18].append(0)
        n_row += 1
test_x = np.array(test_x)

# add square term
test_x = np.concatenate((test_x,test_x**2), axis=1)
# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)

# get ans.csv with my model
ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w, test_x[i])
    ans[i].append(a)

with open(sys.argv[2], 'w+') as out:
    writer = csv.writer(out, delimiter=',', lineterminator='\n')
    writer.writerow(["id","value"])
    for i in range(len(ans)):
        writer.writerow(ans[i])

