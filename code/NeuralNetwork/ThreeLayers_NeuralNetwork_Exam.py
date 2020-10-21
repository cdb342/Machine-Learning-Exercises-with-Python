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
Exam1 = np.linspace(10, 65, 350)#在10~65间均匀生成350个数
Exam2 = np.linspace(38, 90, 350)#在38~90间均匀生成350个数
XX1, XX2 = np.meshgrid(Exam1, Exam2)#生成网格点坐标矩阵
XX=np.c_[XX1.ravel(),XX2.ravel()]#平铺并合并XX1和XX2，方便预测网格点上每个点的分类
def sigmoid(z):
    if np.all(z >= 0):
        return 1.0 / (1 + np.exp(-z))
    else:
        return np.exp(z) / (1 + np.exp(z))
def Standardize(Standard_X,X):
    std = np.std(Standard_X)#计算数据的方差
    mean = np.mean(Standard_X)#计算数据的均值
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
        Z1=np.dot(X,self.W1)+self.b1
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
        self.Accuracy_train = []
        self.Accuracy_test = []
        N=len(self.y)
        s_W1 = v_W1 = np.zeros((2, 2))
        s_W2 = v_W2 = np.zeros((2, 1))
        s_b1 = v_b1 = np.zeros((1, 2))
        s_b2 = v_b2 = np.zeros(1)
        self.ZZ = []
        for i in range(self.times):
            self.ZZ.append(self.predict(XX_st).reshape(np.shape(XX1)))
            h=self.feedforward(self.X)
            y_hat_test = self.predict(test_X_st)
            y_hat = self.predict(self.X)
            ac_test = self.accuracy(y_hat_test, test_y)
            ac_train = self.accuracy(y_hat, self.y)
            self.Accuracy_test.append(ac_test)
            self.Accuracy_train.append(ac_train)
            J=np.sum((h-self.y)**2)/2
            self.cost.append(J)
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
        h=self.feedforward(X)
        y_hat=np.argmax(np.concatenate([1 - h, h], axis=1),axis=1).reshape(-1,1)
        return y_hat
    def accuracy(self,y_hat,y):
        return len(np.where(y_hat==y)[0])/len(y)
train_X_st=Standardize(train_X,train_X)
test_X_st=Standardize(train_X,test_X)
XX_st=Standardize(train_X,XX)
aa=ThreeLayers_NeuralNetworks(0.15,600)
aa.fit(train_X_st,train_y)
train_y_hat=aa.feedforward(train_X_st)
print(train_y_hat)
print(aa.Accuracy_test,'\n',aa.Accuracy_train)
print(aa.cost)
"""
可视化
"""
fig,ax=plt.subplots(1,3,figsize=(15,5))#创建画布和坐标轴
ax[0].scatter(train_X[:,0],train_X[:,1],c=train_y,cmap='Dark2',s=10)#在第一个坐标轴上绘制训练集分布
ax[0].scatter(test_X[:,0],test_X[:,1],c=test_y,cmap='Dark2',s=10)#在第一个坐标轴上绘制测试集分布
ax[0].set_xlabel("Exam1")#设置x轴标签
ax[0].set_ylabel("Exam2")#设置y轴标签
ax[0].set_title("Classification Boundaries")#设置标题
ax[1].set_xlim(0,599)#设置x轴范围
ax[1].set_ylim(2.5,13)#设置y轴范围
ax[1].set_xlabel("Iteration")#设置x轴标签
ax[1].set_ylabel("Cost Function")#设置y轴标签
ax[1].set_title("Cost Change")#设置标题
ax[2].set_xlim(0,599)
ax[2].set_ylim(0,1)
ax[2].set_xlabel("Iteration")#设置x轴标签
ax[2].set_ylabel("Accuracy")#设置y轴标签
ax[2].set_title("Accuracy Change")#设置标题
cont=ax[0].contourf(XX1,XX2,aa.ZZ[0],alpha=0.2,cmap='Set3')#绘制决策界限
line1,=ax[1].plot([],[])#在第二个坐标轴上绘制损失函数下降情况
sca2=ax[2].scatter([],[],label='training set',s=10,c='red')#在第三个坐标轴上绘制训练集预测准确率变化情况
sca3=ax[2].scatter([],[],label='test set',s=10)#在第三个坐标轴上绘制测试集预测准确率变化情况
ite=np.arange(600)#迭代次数数组
def animate(i):#定义动画更新函数
    global cont
    for c in cont.collections:  # 加快动画运行速率
        c.remove()
    cont = ax[0].contourf(XX1, XX2, aa.ZZ[i], alpha=0.2, cmap='Set3')  # 更新决策界限
    line1.set_data(ite[:i],aa.cost[:i])
    sca2.set_offsets(np.stack((ite[:i],aa.Accuracy_train[:i]),axis=1))
    sca3.set_offsets(np.stack((ite[:i],aa.Accuracy_test[:i]),axis=1))
    return ax[0],line1,sca2,sca3
ani=animation.FuncAnimation(fig,animate,frames=600,interval=1)
plt.legend()
plt.show()
#ani.save('ThreeLayers_NeuralNetwork_Exam.gif',writer='imagemagick',fps=60)#保存动态图（需要安装imagemagick）
