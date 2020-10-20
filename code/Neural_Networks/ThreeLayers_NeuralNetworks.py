import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
"""
导入数据
train_X:训练集特征集合   train_y:训练集标签集合
test_X:测试集特征集合    test_y：测试集标签集合
"""
train_X=np.loadtxt("./Exam/train/x.txt")
train_y=np.loadtxt("./Exam/train/y.txt").reshape(-1,1)
test_X=np.loadtxt("./Exam/test/x.txt")
test_y=np.loadtxt("./Exam/test/y.txt").reshape(-1,1)
print(train_X,'\n',train_y,'\n',test_X,'\n',test_y)#展示数据
def sigmoid(z):
    if np.all(z >= 0):
        return 1.0 / (1 + np.exp(-z))
    else:
        return np.exp(z) / (1 + np.exp(z))
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
        np.random.seed(2)
        self.W1 = np.random.randn(2, 2)
        self.W2 = np.random.randn(2, 1)
        self.b1 = np.random.randn(1, 2)
        self.b2=np.random.randn(1)
        self.cost=[]
        self.Y_hat=[]
        self.Accuracy=[]
        N=len(self.y)
        for i in range(self.times):
            h=self.feedforward(self.X)
            y_hat = self.predict(self.X)
            self.Y_hat.append(y_hat)
            J=np.sum((h-self.y)**2)/2
            self.cost.append(J)
            ac=self.accuracy(y_hat)
            self.Accuracy.append(ac)
            self.y_mul=np.multiply(np.multiply(h,1-h),h-self.y)
            self.A_mul=np.multiply(np.multiply(self.A,1-self.A),self.y_mul)
            self.g_b2=np.mean(self.y_mul)
            self.g_W2=np.dot(self.A.T,self.y_mul)/N
            self.g_W1=np.dot(self.X.T,self.A_mul,np.diag(self.W2.ravel()))/N
            self.g_b1=np.dot(np.mean(self.A_mul,axis=0).reshape(1,-1),np.diag(self.W2.ravel()))
            self.W1-=self.yita*self.g_W1
            self.W2-=self.yita*self.g_W2
            self.b1-=self.yita*self.g_b1
            self.b2-=self.yita*self.g_b2
    def predict(self,X):
        h=self.feedforward(self.X)
        y_hat=np.argmax(np.concatenate([1 - h, h], axis=1),axis=1).reshape(-1,1)
        return y_hat
    def accuracy(self,y_hat):
        self.y_hat = y_hat
        return len(np.where(y_hat==self.y)[0])/len(self.y)

def st(X):
    std = np.std(X)#计算数据的方差
    mean = np.mean(X)#计算数据的均值
    return (X - mean) / std#返回数据标准化的结果
train_X_st=st(train_X)
aa=ThreeLayers_NeuralNetworks(0.5,2000)
aa.fit(train_X_st,train_y)
train_y_hat=aa.feedforward(train_X_st)
print(train_y_hat)
print(aa.Accuracy)
print(aa.cost)





