import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
import itertools
import collections
from statistics import mean,stdev
import matplotlib.pyplot as plt
f = pd.read_csv('FTCT-Chem-YS-UTS 4.csv')
T1 = f.iloc[:, 0:2]
for j in range(0,2):
    for i in range(3,13):
        X=f.iloc[:,i]
        plt.figure()
        plt.xlabel('input %(i)d'%{'i':i-2})
        plt.ylabel('output %(j)d'%{'j':j+1})
        plt.scatter(X, T1.iloc[:,j],s=5)
        plt.show()
X=f.iloc[:,3:13]
xm=[]
xs=[]
for i in range(0,10):
    xm.append(mean(X.iloc[:, i]))
    xs.append(stdev(X.iloc[:, i]))
    X.iloc[:,i]=X.iloc[:,i]-xm[i]
    X.iloc[:,i]=X.iloc[:,i]/xs[i]
tm=[]
ts=[]
for i in range(0,2):
    tm.append(mean(T1.iloc[:,i]))
    ts.append(stdev(T1.iloc[:,i]))
    T1.iloc[:,i]=T1.iloc[:,i]-tm[i]
    T1.iloc[:,i]=T1.iloc[:,i]/ts[i]
X_train,X_test,T1_train,T1_test=train_test_split(X,T1,test_size=0.2,random_state=8)
class layer():
    def __init__(self,n_in,n_out):
        np.random.seed(11)
        self.W=np.random.randn(n_in,n_out)
        self.b=np.zeros(n_out)
    def get_pi(self):
        return itertools.chain(np.nditer(self.W),np.nditer(self.b))
    def get_o(self,X):
        return X.dot(self.W)+self.b
    def get_pg(self,X,og,b):
        lda=0.045
        JW=X.T.dot(og)
        JW/=b
        JW += lda * self.W
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
h1=7
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
def backward(a,t,layers,b):
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
        g=l.get_pg(X,og,b)
        pg.appendleft(g)
        og=ig
    return pg
b=T1_train.shape[0]
maxiter=920
lr=0.0028
for i in range(maxiter):
    a=forward(X_train,layers)
    pg=backward(a,T1_train,layers,b)
    for l, lbg in zip(layers, pg):
        l.up(lbg,lr)
yt=T1_test
a=forward(X_test,layers)
yp=a[-1]
for i in range(0,2):
    yp.iloc[:, i] = yp.iloc[:, i] * ts[i] + tm[i]
    yt.iloc[:, i] = yt.iloc[:, i] * ts[i] + tm[i]
    tl=[x for x in yp.iloc[:, i]]
    plt.figure()
    plt.xlabel('predicted output')
    plt.ylabel('actual output')
    plt.axes().set_xlim(xmin=300,xmax=600)
    plt.axes().set_ylim(ymin=300,ymax=600)
    plt.scatter(yp.iloc[:, i], yt.iloc[:, i], s=5,)
    plt.plot(yp.iloc[:, i],tl,)
    plt.show()
acc=metrics.r2_score(yt,yp)
acc*=100
print("the percentage accuracy of the model is %(acc)f"%{"acc":acc})
'''
yt=T1_train
a=forward(X_train,layers)
yp=a[-1]
acc=metrics.r2_score(yt,yp)
acc*=100
print("the percentage accuracy of the model on training data is is %(acc)f"%{"acc":acc})
'''
s1=np.array([[881.875,845.625,78.512,0,653.144,90.535,0.045,0.384,1.32,0.047]])
for i in range(0,10):
    s1[0][i]=s1[0][i]-xm[i]
    s1[0][i]=s1[0][i]/xs[i]
a1=forward(s1,layers)
predict1=a1[-1]
predict1[0][0]= predict1[0][0] * ts[0] + tm[0]
predict1[0][1]= predict1[0][1] * ts[1] + tm[1]
print("yield of the given sample is ")
print(predict1)