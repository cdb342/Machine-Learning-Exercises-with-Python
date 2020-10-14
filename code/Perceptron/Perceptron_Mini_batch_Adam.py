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
        beta1 = 0.9
        beta2=0.999
        v = 0
        s=0
        for i in range(self.times):
            for j in range(Num_batch):
                H_omiga[j*16:j*16+16]=np.dot(self.X[j*16:j*16+16],self.omiga[i*Num_batch+j])>=0
                error=H_omiga[j*16:j*16+16]-self.y[j*16:j*16+16]
                J=np.sum(np.multiply(error,np.dot(self.X[j*16:j*16+16],self.omiga[i*Num_batch+j])))
                self.cost.append(J)
                g = np.dot(self.X[j * 16:j * 16 + 16].T, error) / 16
                v=beta1*v+(1-beta1)*g
                v_correct=v/(1-beta1**(i*Num_batch+j+1))
                s=beta2*s+(1-beta2)*(g**2)
                s_correct=s/(1-beta2**(i*Num_batch+j+1))
                self.omiga.append(self.omiga[i*Num_batch+j]-self.alpha*v_correct/np.sqrt(s_correct+10**(-8)))
    def predict(self,X):
        self.X=np.insert(X,0,1,axis=1)
        H_omiga=np.dot(self.X, self.omiga) >= 0
        return H_omiga
def Accuracy_list(X,y,omiga):
    omiga=np.array(omiga[:-1]).reshape(-1, 3).T
    X=np.insert(X,0,1,axis=1)
    H_omiga=np.dot(X,omiga)>=0
    return np.sum(np.where(H_omiga==y.reshape(-1,1),1,0),axis=0)/len(y)
aa=Perceptron_Mini_batch_GD(0.001,2000)
aa.fit(train_X,train_y)
train_y_hat=aa.predict(train_X)
test_y_hat=aa.predict(test_X)
print(aa.cost)
print(Accuracy_list(train_X,train_y,aa.omiga))

fig,ax=plt.subplots(1,3,figsize=(15,5))
ax[0].scatter(train_X[:,0],train_X[:,1],c=train_y,cmap='spring',s=10)
ax[0].scatter(test_X[:,0],test_X[:,1],c=test_y,cmap='spring',s=10)
ax[0].set_xlabel("Exam1")#设置x轴标签
ax[0].set_ylabel("Exam2")#设置y轴标签
ax[0].set_title("Classification Line")#设置标题
ax[1].set_xlim(0,399)
ax[1].set_ylim(0,1450)
ax[1].set_xlabel("Iteration")#设置x轴标签
ax[1].set_ylabel("Cost Function")#设置y轴标签
ax[1].set_title("Cost Change")#设置标题
ax[2].set_xlim(0,399)
ax[2].set_ylim(0,1)
ax[2].set_xlabel("Iteration")#设置x轴标签
ax[2].set_ylabel("Accuracy")#设置y轴标签
ax[2].set_title("Accuracy Change")#设置标题
exam1=np.linspace(0,65,100)
line0,=ax[0].plot([],[])
def animate(i):
    exam2 =(-aa.omiga[i][0]-aa.omiga[i][1]*exam1)/aa.omiga[i][2]
    line0.set_data(exam1,exam2)
    return line0,
ani=animation.FuncAnimation(fig,animate,frames=8000,interval=10,blit=True)
plt.show()
