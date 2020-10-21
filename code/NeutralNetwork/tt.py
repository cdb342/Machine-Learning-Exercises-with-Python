import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
"""
导入数据
train_X:训练集特征集合   train_y:训练集标签集合
test_X:测试集特征集合    test_y：测试集标签集合
"""
train_X=np.loadtxt("./Iris/train/x.txt")
train_y=np.loadtxt("./Iris/train/y.txt").reshape(-1,1)
test_X=np.loadtxt("./Iris/test/x.txt")
test_y=np.loadtxt("./Iris/test/y.txt").reshape(-1,1)
print(train_X,'\n',train_y,'\n',test_X,'\n',test_y)#展示数据
def sigmoid(z):
    if np.all(z >= 0):
        return 1.0 / (1 + np.exp(-z))
    else:
        return np.exp(z) / (1 + np.exp(z))
def Standardize(X):
    std = np.std(X)#计算数据的方差
    mean = np.mean(X)#计算数据的均值
    return (X - mean) / std#返回数据标准化的结果
def Adam(v,s,g,i):
    v=0.9*v+0.1*g
    s=0.999*s+0.001*g**2
    v_correct = v/ (1 - 0.9 ** i)
    s_correct = s / (1 - 0.999 ** i)
    g=v_correct/(np.sqrt(s_correct) + 10 ** (-8))
    return v,s,g
class ThreeLayers_NeuralNetworks():
    def __init__(self,yita,times):
        self.yita=yita
        self.times=times
    def feedforward(self,X):
        self.X=X
        Z1=np.dot(self.X,self.W1)+self.b1
        self.A=sigmoid(Z1)
        Z2=np.dot(self.A,self.W2)+self.b2
        h=sigmoid(Z2)
        return h
    def fit(self,X,y):
        self.X=X
        self.y=y
        y=np.eye(3)[self.y]
        np.random.seed(2)
        self.W1 = np.random.randn(2, 2)
        self.W2 = np.random.randn(2, 1)
        self.b1 = np.random.randn(1, 2)
        self.b2=np.random.randn(1)
        self.cost=[]
        self.Y_hat=[]
        self.Accuracy=[]
        s_W1 = v_W1 = np.zeros((2, 2))
        s_W2 = v_W2 = np.zeros((2, 1))
        s_b1 = v_b1 = np.zeros((1, 2))
        s_b2 = v_b2 = np.zeros(1)
        for i in range(self.times):
            h=self.feedforward(self.X)
            y_hat = self.predict(self.X)
            self.Y_hat.append(y_hat)
            J=np.sum((h-y)**2)/2
            self.cost.append(J)
            ac=self.accuracy(y_hat)
            self.Accuracy.append(ac)
            y_mul=np.multiply(np.multiply(h,1-h),h-self.y)
            A_mul=np.multiply(self.A,1-self.A)
            self.g_b2=np.sum(y_mul)
            self.g_W2=np.dot(self.A.T,y_mul)
            self.g_W1=np.dot(self.X.T,np.multiply(A_mul,np.dot(y_mul,self.W2.T)))
            self.g_b1=np.dot(np.dot(y_mul.T,A_mul),np.diag(self.W2.ravel()))
            s_W1,v_W1,self.g_W1=Adam(s_W1,v_W1,self.g_W1,i+1)
            s_W2, v_W2, self.g_W2 = Adam(s_W2, v_W2, self.g_W2, i+1)
            s_b1, v_b1, self.g_b1 = Adam(s_b1, v_b1, self.g_b1, i+1)
            s_b2, v_b2, self.g_b2 = Adam(s_b2, v_b2, self.g_b2, i+1)
            self.W1-=self.yita*self.g_W1
            self.W2-=self.yita*self.g_W2
            self.b1-=self.yita*self.g_b1
            self.b2-=self.yita*self.g_b2
    def predict(self,X):
        h=self.feedforward(self.X)
        y_hat=np.argmax(h,axis=1)
        return y_hat
    def accuracy(self,y_hat):
        return len(np.where(y_hat==self.y)[0])/len(self.y)
train_X_st=Standardize(train_X)
aa=ThreeLayers_NeuralNetworks(0.1,1200)
aa.fit(train_X_st,train_y)
train_y_hat=aa.feedforward(train_X_st)
print(train_y_hat)
print(aa.Accuracy)
print(aa.cost)
