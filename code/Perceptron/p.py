import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
"""
导入数据
train_X:训练集特征集合   train_y:训练集标签集合
test_X:测试集特征集合    test_y：测试集标签集合
"""
train_X=np.loadtxt("./Exam/train/x.txt")
train_y=np.loadtxt("./Exam/train/y.txt")
test_X=np.loadtxt("./Exam/test/x.txt")
test_y=np.loadtxt("./Exam/test/y.txt")
print(train_X,'\n',train_y,'\n',test_X,'\n',test_y)#展示数据
"""创建Perceptron_Mini_batch_Adam类
fit():训练数据的方法
Parameters:X,y
predict():预测数据分类情况
Parameter:X
Return:数据的类别
"""
class Perceptron_Mini_batch_Adam():
    def __init__(self,alpha,times):#输入为学习率和迭代次数
        self.times=times
        self.alpha=alpha
    def fit(self,X,y):
        np.random.seed(2)#使特征集和标签集以相同顺序打乱
        self.X=np.random.permutation(np.insert(X,0,1,axis=1))#在特征集第一列插入1，并按行打乱其顺序
        np.random.seed(2)#使特征集和标签集以相同顺序打乱
        self.y=np.random.permutation(y.reshape(-1,1))#将标签集转化为列向量并打乱其顺序
        np.random.seed(2)#使每次运行的初始权值相同
        self.omiga=[np.random.randn(self.X.shape[1],2)]#按照高斯分别随机初始化权值
        self.cost=[]
        H_omiga = []
        Num_batch=int(len(self.X)/16)#每个mini batch长度为16，计算mini batch的数量
        for i in range(self.times):
            for j in range(Num_batch):
                multi=np.dot(self.X[j*16:j*16+16],self.omiga[i*Num_batch+j])
                H_omiga[j*16:j*16+16]=np.argmax(multi,axis=1).reshape(-1,1)#用上一次下降的权值计算每一个mini batch的预测值
                J=np.sum(multi[np.array(H_omiga[j*16:j*16+16]).astype(int)]-multi[self.y[j*16:j*16+16].astype(int)])#在该mini batch内计算损失函数
                self.cost.append(J)
                g0=np.dot(self.X[j*16:j*16+16].T,np.logical_not(np.asarray(H_omiga[j*16:j*16+16])).astype(int)-np.logical_not(self.y[j*16:j*16+16]).astype(int))
                g1 =np.dot(self.X[j * 16:j * 16 + 16].T,(H_omiga[j*16:j*16+16] - self.y[j*16:j*16+16] ))
                self.omiga.append(self.omiga[i*Num_batch+j]-self.alpha*np.hstack((g0,g1)))#用adam算法更新权值
    def predict(self,X):
        self.X=np.insert(X,0,1,axis=1)
        multiply = np.dot(self.X, self.omiga[-1])
        H_omiga = np.argmax(multiply, axis=1)
        return H_omiga
def Accuracy(X,y,omiga):#准确率计算函数（输入迭代过程中生成的每一个权值，用矩阵相乘的方式计算每个权值对应的准确率）
    X=np.insert(X,0,1,axis=1)
    multi = np.dot(X, omiga)
    H_omiga = np.argmax(multi, axis=1)
    return np.sum(np.where(H_omiga==y,1,0))/len(y)
aa=Perceptron_Mini_batch_Adam(0.05,200)
aa.fit(train_X,train_y)#训练权值
train_y_hat=aa.predict(train_X)#预测训练集
test_y_hat=aa.predict(test_X)#预测测试集
print(aa.cost)#输出损失函数
ac_train=[]
for i in range(800):
    ac_train.append(Accuracy(train_X,train_y,aa.omiga[i]))
ac_test=[]
for i in range(800):
    ac_test.append(Accuracy(train_X,train_y,aa.omiga[i]))
print(ac_train,'\n',ac_test)#输出训练集预测准确率
"""
可视化
"""
fig,ax=plt.subplots(1,3,figsize=(15,5))#创建画布和坐标轴
ax[0].scatter(train_X[:,0],train_X[:,1],c=train_y,cmap='spring',s=10)#在第一个坐标轴上绘制训练集分布
ax[0].scatter(test_X[:,0],test_X[:,1],c=test_y,cmap='spring',s=10)#在第一个坐标轴上绘制测试集分布
ax[0].set_xlabel("Exam1")#设置x轴标签
ax[0].set_ylabel("Exam2")#设置y轴标签
ax[0].set_title("Classification Line")#设置标题
ax[1].set_xlim(0,799)
ax[1].set_ylim(0,1450)
ax[1].set_xlabel("Iteration")#设置x轴标签
ax[1].set_ylabel("Cost Function")#设置y轴标签
ax[1].set_title("Cost Change")#设置标题
ax[2].set_xlim(0,799)
ax[2].set_ylim(0,1)
ax[2].set_xlabel("Iteration")#设置x轴标签
ax[2].set_ylabel("Accuracy")#设置y轴标签
ax[2].set_title("Accuracy Change")#设置标题
exam1=np.linspace(0,65,100)#在0-65间均匀生成100个数
line0,=ax[0].plot([],[])#在第一个坐标轴上绘制决策界限
line1,=ax[1].plot([],[])#在第二个坐标轴上绘制损失函数下降情况
sca2=ax[2].scatter([],[],label='training set',s=10)#在第三个坐标轴上绘制训练集预测准确率变化情况
sca3=ax[2].scatter([],[],label='test set',s=10)#在第三个坐标轴上绘制测试集预测准确率变化情况
ite=np.arange(800)#迭代次数
def animate(i):#定义动画更新函数
    exam2 =(-(aa.omiga[i][0][0]-aa.omiga[i][0][1])-(aa.omiga[i][1][0]-aa.omiga[i][1][1])*exam1)/(aa.omiga[i][2][0]-aa.omiga[i][2][1])
    line0.set_data(exam1,exam2)
    line1.set_data(ite[:i],aa.cost[:i])
    sca2.set_offsets(np.stack((ite[:i],ac_train[:i]),axis=1))
    sca3.set_offsets(np.stack((ite[:i],ac_test[:i]),axis=1))
    return line0,line1,sca2,sca3
ani=animation.FuncAnimation(fig,animate,frames=800,interval=10,blit=True)
plt.legend()
plt.show()
