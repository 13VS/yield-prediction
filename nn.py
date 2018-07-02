import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import preprocessing,metrics
import itertools
import collections
from statistics import mean,stdev

import matplotlib.pyplot as plt
f = pd.read_csv('FTCT-Chem-YS-UTS 4.csv')
T1 = f.iloc[:, 0:2]
X=f.iloc[:,3:13]
'''
X=preprocessing.scale(X)
T1=preprocessing.scale(T1)
'''
for i in range(0,10):
    X.iloc[:,i]=X.iloc[:,i]-mean(X.iloc[:,i])
    X.iloc[:,i]=X.iloc[:,i]/stdev(X.iloc[:,i])
m=[]
s=[]
for i in range(0,2):
    m.append(mean(T1.iloc[:,i]))
    s.append(stdev(T1.iloc[:,i]))
    T1.iloc[:,i]=T1.iloc[:,i]-m[i]
    T1.iloc[:,i]=T1.iloc[:,i]/s[i]
X_train,X_test,T1_train,T1_test=train_test_split(X,T1,test_size=0.2,random_state=0)
class layer():
    def __init__(self,n_in,n_out):
        self.W=np.random.randn(n_in,n_out)
        self.b=np.zeros(n_out)
    def get_pi(self):
        return itertools.chain(np.nditer(self.W),np.nditer(self.b))
    def get_o(self,X):
        return X.dot(self.W)+self.b
    def get_pg(self,X,og,b):##very doubtful
        JW=X.T.dot(og)
        JW/=b
        Jb=np.sum(og,axis=0)
        Jb/=b
        return [g for g in itertools.chain(np.nditer(JW),np.nditer(Jb))]
    def get_ig(self,og):
        return og.dot(self.W.T)
    def up(self, lbg,lr):
        i=0
        for p, grad in zip(itertools.chain(np.nditer(self.W),np.nditer(self.b)), lbg):
            ab=np.copy(p)
            ab -= lr * grad
            r=self.W.shape[0]
            c=self.W.shape[1]
            if(i<r*c):
                a=(int)(i/c)
                b=i%c
                self.W[a][b]=ab
            else:
                self.b[i-r*c]=ab
            i+=1
h1=6
h2=4
layers=[]
layers.append(layer(X_train.shape[1],h1))
layers.append(layer(h1,h2))
layers.append(layer(h2,T1_train.shape[1]))
def forward(input,layers):
    a=[input]
    X=input
    for l in layers:
        y=l.get_o(X)
        a.append(y)
        X=a[-1]
    return a
def backward(a,t,layers,b):##doubt
    pg=collections.deque()
    og=None
    for l in reversed(layers):
        y=a.pop()
        if og is None:
            og=(np.array(y) - np.array(t))
            ig=l.get_ig(og)
        else :
            ig=l.get_ig(og)
        X=a[-1]
        g=l.get_pg(X,og,b)##very doubtful
        pg.appendleft(g)
        og=ig
    return pg
b=T1_train.shape[0]
nb=X_train.shape[0]/b
XT=zip(
    np.array_split(X_train, 1, axis=0),
    np.array_split(T1_train, 1, axis=0))
maxiter=800
lr=0.01
for i in range(maxiter):
    ##for X,T in XT:
    a=forward(X_train,layers)
    pg=backward(a,T1_train,layers,b)
    for l, lbg in zip(layers, pg):
        l.up(lbg,lr)
yt=T1_test
a=forward(X_test,layers)
yp=a[-1]
for i in range(0,2):
    yp.iloc[:, i] = yp.iloc[:, i] * s[i] + m[i]
    yt.iloc[:, i] = yt.iloc[:, i] * s[i] + m[i]
    plt.figure()
    plt.xlabel('predicted output')
    plt.ylabel('actual output')
    plt.axes().set_xlim(xmin=230,xmax=670)
    plt.axes().set_ylim(ymin=230,ymax=670)
    plt.scatter(yp.iloc[:, i], yt.iloc[:, i], s=5,)
    plt.show()
print(yt)
print(yp)
acc=metrics.r2_score(yt,yp)
print(acc)