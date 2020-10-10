import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
"""
导入数据
train_X:训练集特征集合   train_y:训练集标签集合
test_X:测试集特征集合    test_y：测试集标签集合
"""
train_X=np.loadtxt("./Iris/train/x.txt")
train_y=np.loadtxt("./Iris/train/y.txt")
test_X=np.loadtxt("./Iris/test/x.txt")
test_y=np.loadtxt("./Iris/test/y.txt")
print(train_X,train_y,test_X,test_y)
"""
创建softmax函数
"""
def softmax(z):
    return np.exp(z)/np.sum(np.exp(z),axis=1).reshape((-1,1))
"""
创建softmax回归类
fit():训练数据的方法
Parameters:X,y
PredictProbability()：预测数据属于标签0，1或2的概率
Parameter:X
Return:属于标签0，1或2的概率
predict():预测数据分类情况
Parameter:X
Return:数据的类别
"""
class SoftmaxRegression_SGD():
    def __init__(self,alpha,times):#参数为学习率与梯度下降迭代次数
        self.alpha=alpha
        self.times=times
    def fit(self,X,y):
        len_class = len(np.unique(y))#计算标签类别数
        len_y = []
        for i in range(len_class):
            len_y.append(len(np.where(y == i)[0]))#计算每个类别中样本的数量
        np.random.seed(2)#使特征集和标签集以相同的顺序打乱
        self.y = np.random.permutation(np.repeat(np.identity(len_class), len_y, axis=0))#将标签转化成向量形式并打乱标签集
        np.random.seed(2)#使特征集和标签集以相同的顺序打乱
        self.X=np.random.permutation(np.insert(X,0,1,axis=1))#在特征集第一列插入一列全1的数据，并打乱特征集
        np.random.seed(2)#使每次生成的随机数相同，确保每次运行有相同的结果
        self.theta=[np.random.rand(self.X.shape[1],len_class)]#初始化特征集的权值
        self.z=np.dot(self.X,self.theta[0])
        self.H_theta =softmax(self.z)#用初始权值计算特征集属于每个标签的概率
        self.cost=[]
        len_X=self.X.shape[0]#样本数
        for i in range(self.times):#运行times次梯度下降
            for j in range(self.X.shape[0]):
                self.index_theta=len_X*i+j#每次迭代的权值索引
                self.z[j]=np.dot(self.X[j],self.theta[self.index_theta])
                self.H_theta[j] = np.exp(self.z[j])/np.sum(np.exp(self.z[j]))#用更新的权值计算本次迭代的样本属于每个标签的概率
                error = self.y[j] - self.H_theta[j]#计算标签与预测概率的差
                J = -np.sum(np.multiply(np.log(self.H_theta), self.y))#计算整体的损失函数
                self.cost.append(J)
                beta=self.alpha/(i*self.index_theta*0.05+0.2)#学习率逐次衰减，加快学习速率
                self.theta.append(self.theta[self.index_theta] + beta * np.dot(self.X[j].reshape(-1, 1), error.reshape(1, -1)))#梯度下降更新权值并将每一次迭代计算出的权值储存在数组中，方便后续可视化
    def PredictProbability(self,X):
        self.X=np.insert(X,0,1,axis=1)
        z=np.dot(self.X,self.theta[self.times*self.X.shape[0]])
        H_theta=softmax(z)#计算特征集属于每个标签的概率
        return H_theta
    def predict(self,X):
        return np.argmax(self.PredictProbability(X),axis=1)#返回特征值属于每个标签的概率的最大值，即预测数据的分类情况
"""
预测准确率计算函数
"""
def accuracy(X,y,theta):
    X=np.insert(X,0,1,axis=1)
    z = np.dot(X, theta)
    H_theta = softmax(z)
    y_hat=np.argmax(H_theta,axis=1)#计算特征集的标签预测集
    temp=np.where(y==y_hat)#计算标签预测集与标签真实集相同的索引
    return len(temp[0])/len(y)#返回准确率

aa = SoftmaxRegression_SGD(0.08,3)
aa.fit(train_X, train_y)#训练数据
train_y_hat = aa.predict(train_X)#用训练好的权值预测训练集
test_y_hat = aa.predict(test_X)#用训练好的权值预测测试集
print(aa.cost)#输出损失函数
"""
#计算每一次迭代后的准确率
"""
ac1=[]
for i in range(aa.index_theta+1):
    ac1.append(accuracy(train_X,train_y,aa.theta[i]))
ac2=[]
for i in range(aa.index_theta+1):
    ac2.append(accuracy(test_X,test_y,aa.theta[i]))
print(ac1,'\n',ac2)
"""
#可视化
"""
fig= plt.figure(figsize=(15,5))#创建画布，大小为15x5
ax0=plt.subplot(131,xlim=(1.5, 5), ylim=(0, 3))#创建第一个坐标轴,绘制数据点分布和算法分类情况
plt.xlabel("Fetures1")#设置x轴标签
plt.ylabel("Feature2")#设置y轴标签
plt.title("Classification Line")#设置标题
ite=len(train_X)*3#迭代次数
a1=ax0.scatter(train_X[:,0], train_X[:,1], c=train_y, cmap='Dark2')#在第一个坐标轴上绘制训练集数据的分布情况
a2=ax0.scatter(test_X[:,0], test_X[:,1], c=test_y, cmap='Dark2')#在第一个坐标轴上绘制测试集数据的分布情况
ax1=plt.subplot(132,xlim=(0, ite), ylim=(26, 195))#创建第二个坐标轴，绘制损失函数下降情况
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Function Change")
ax2=plt.subplot(133,xlim=(0, ite), ylim=(0, 1))#创建第三个坐标轴，绘制准确率变化情况
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.title("Softmax Regression-Stochastic Gradient Descent Accuracy Change")
"""
#绘制动态图，描绘决策界限，损失函数以及预测准确率随时间的变化
"""
feature1 = np.linspace(1.5, 5, 500)#在1.5~5间均匀生成500个数,500比较大，会运行得久一些，但是生成的界限比较光滑
feature2 = np.linspace(0, 3, 500)#在0~3间均匀生成500个数
XX1, XX2 = np.meshgrid(feature1, feature2)#生成网格点坐标矩阵
XX=np.insert(np.c_[XX1.ravel(),XX2.ravel()], 0, 1, axis=1)#平铺并合并XX1和XX2，方便预测网格点上每个点的分类
z = np.dot(XX, aa.theta[0])
H_theta = softmax(z)
yy=np.argmax(H_theta, axis=1).reshape(np.shape(XX1))#预测网格点上每个点的分类
cont=ax0.contourf(XX1,XX2,yy,alpha=0.2,cmap='Set3')#绘制决策界限
iteration = np.arange(0, ite)
line1, = ax1.plot([], [], lw=2)#初始化线条，必须有逗号，,表示得到元组
sca2 = ax2.scatter([], [], label="training set", s=10)
sca3 = ax2.scatter([], [], label="test set",s=10)

def animate(i):#动画更新函数
    global cont
    for c in cont.collections:#加快动画运行速率
        c.remove()
    z = np.dot(XX, aa.theta[i+1])
    H_theta = softmax(z)
    yy=np.argmax(H_theta, axis=1).reshape(np.shape(XX1))#根据权值更新预测情况
    cont=ax0.contourf(XX1,XX2,yy,alpha=0.2,cmap='Set3')#更新决策界限
    line1.set_data(iteration[:i], aa.cost[:i])#更新损失函数随迭代次数变化的值
    sca2.set_offsets(np.stack((iteration[:i],ac1[:i]),axis=1))#更新训练集准确率随迭代次数变化的值
    sca3.set_offsets(np.stack((iteration[:i],ac2[:i]),axis=1))#更新测试集准确率随迭代次数变化的值
    return cont,line1,sca2,sca3
anim = animation.FuncAnimation(fig, animate,frames=ite, interval=1)#创建动画
plt.legend(loc='lower right')#显示图例并设置图例位置
plt.show()
#anim.save('SoftmaxRegression_SGD.gif',writer='imagemagick',fps=30)#保持动态图（需要安装imagemagick）"""
