import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path='data.csv'
data=pd.read_csv(path)
data.drop(data.columns[data.columns.str.contains('unnamed',case = False)],axis = 1)
data.rename({"Unnamed: 32":"a"}, axis="columns", inplace=True)
data.drop(["a"], axis=1, inplace=True)
#print(data)
#data.head()
def get_Xy(data):
    data.insert(2,'ones',1)
    X_=data.iloc[:,2:]
    X=X_.values
    #print(X)
    
    y_=data.iloc[:,1]
    y=y_.values.reshape(len(y_),1)
    #print(y)
    b=0
    while b<=568:
        if y[b,0]=='B':
            y[b,0]=0
        else:
            y[b,0]=1
        b+=1
    #print(y)
    
    return X,y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def costFunction(X,y,theta): 
    A=sigmoid(X@theta)
    #print(A)
    f=y*np.log(A)
    s=(1-y)*np.log(1-A)
    return -np.sum(f+s)/len(X)

def gradientDescent(X,y,theta,iters,alpha):
    m=len(X)
    costs=[]

    for i in range(iters):
        A=sigmoid(X@theta)
        print(A)
        print(X)
        print(theta)
        print(alpha/m)
        theta=theta-(alpha/m)*X.T@(A-y)
        cost=costFunction(X,y,theta)
        costs.append(cost)
        if i%10000==0:
            print(cost)
        return costs,theta

alpha=0.004
X,y=get_Xy(data)
theta=np.zeros((31,1))
print(theta)
iters=200000
costs,theta_final=gradientDescent(X,y,theta,iters,alpha)

#theta.shape
#X,y=get_Xy(data)
#cost_init=costFunction(X,y,theta)
#print(cost_init)