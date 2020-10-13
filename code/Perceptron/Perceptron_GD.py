import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
train_X=np.loadtxt("./Exam/train/x.txt")
train_y=np.loadtxt("./Exam/train/y.txt")
test_X=np.loadtxt("./Exam/test/x.txt")
test_y=np.loadtxt("./Exam/test/y.txt")
print(train_X,'\n',train_y,'\n',test_X,'\n',test_y)
class Perceptron_Mini_batch_GD():
    def __init__(self,alpha,times):
        self.times=times
        self.alpha=alpha
    def fit(self,X,y):
        np.random.seed(2)
        self.X=np.random.permutation(np.insert(X,0,1,axis=1))
        np.random.seed(2)
        self.y=np.random.permutation(y.reshape(-1,1))
        np.random.seed(2)
        self.omiga=[np.random.randn(self.X.shape[1],1)]
        self.cost=[]
        H_omiga = np.dot(self.X, self.omiga[0]) >= 0
        Num_batch=int(len(self.X)/16)
        for i in range(self.times):
            for j in range(Num_batch):
                H_omiga[j*16:j*16+16]=np.dot(self.X[j*16:j*16+16],self.omiga[i*Num_batch+j])>=0
                error=H_omiga[j*16:j*16+16]-self.y[j*16:j*16+16]
                J=np.sum(np.multiply(error,np.dot(self.X[j*16:j*16+16],self.omiga[i*Num_batch+j])))
                self.cost.append(J)
                self.omiga.append(self.omiga[i*Num_batch+j]-self.alpha*np.dot(self.X[j*16:j*16+16].T,error))
    def predict(self,X):
        self.X=np.insert(X,0,1,axis=1)
        H_omiga=np.dot(self.X, self.omiga) >= 0
        return H_omiga
def Accuracy_list(X,y,omiga):
    omiga=np.array(omiga[:-1]).reshape(-1, 3).T
    X=np.insert(X,0,1,axis=1)
    H_omiga=np.dot(X,omiga)>=0
    return np.sum(np.where(H_omiga==y.reshape(-1,1),1,0),axis=0)/len(y)
aa=Perceptron_Mini_batch_GD(0.00001,200)
aa.fit(train_X,train_y)
train_y_hat=aa.predict(train_X)
test_y_hat=aa.predict(test_X)
print(aa.cost)
print(Accuracy_list(train_X,train_y,aa.omiga))

fig,ax=plt.subplots(1,3,figsize=(15,5))
ax[0].scatter(train_X[:,0],train_X[:,1],c=train_y,cmap='spring',s=10)
ax[0].scatter(test_X[:,0],test_X[:,1],c=test_y,cmap='spring',s=10)
plt.xlabel("Exam1")#设置x轴标签
plt.ylabel("Exam2")#设置y轴标签
plt.title("Classification Line")#设置标题
ax[1]
plt.show()
