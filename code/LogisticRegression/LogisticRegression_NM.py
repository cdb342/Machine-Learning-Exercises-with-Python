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
print(train_X,train_y,test_X,test_y)#展示数据
"""
建立sigmoid函数
Parameter: z
Return: sigmoid(-z)
"""
def sigmoid(z):
    if np.all(z >= 0):
        return 1.0 / (1 + np.exp(-z))
    else:
        return np.exp(z) / (1 + np.exp(z))
"""
创建逻辑回归类
fit():训练数据的方法
Parameters:X,y
PredictProbability()：预测数据属于标签0或者标签1的概率
Parameter:X
Return:属于标签0和标签1的概率
predict():预测数据分类情况
Parameter:X
Return:数据的类别
"""
class LogisticRegression_NM():
    def __init__(self,times):#参数为牛顿法迭代次数
        self.times=times
    def fit(self,X,y):
        self.X=np.insert(np.asmatrix(X),0,1,axis=1)#在特征集第一列插入一列全1的数据,并转化为矩阵
        self.y=np.asmatrix(y.reshape(-1,1))#将标签集调整为列向量
        np.random.seed(6)#使每次生成的随机数相同，确保每次运行有相同的结果
        self.theta=[np.asmatrix(np.random.rand(self.X.shape[1],1))]#初始化特征集的权值
        self.cost=[]
        for i in range(self.times):#运行times次牛顿法迭代
            z=self.X*self.theta[i]#计算特征集与权值的乘积
            h_theta=sigmoid(z)#计算特征集属于标签1的概率
            J=-np.sum(self.y.ravel()*np.log(h_theta)+(1-self.y).ravel()*np.log(1-h_theta))#计算损失函数
            self.cost.append(J)
            self.theta.append(self.theta[i]-(self.X.T*np.diag(np.asarray(np.multiply(h_theta,1-h_theta)).ravel())*self.X).I*self.X.T*(h_theta-self.y))#用牛顿法迭代更新权值并将每一次迭代计算出的权值储存在数组中，方便后续可视化
    def PredictProbability(self,X):
        self.X=np.insert(X,0,1,axis=1)
        z=np.dot(self.X,self.theta[self.times])
        h_theta=sigmoid(z)#计算特征集属于标签1的概率
        return np.concatenate([1-h_theta,h_theta],axis=1)#将特征值属于标签0的概率和属于1的概率放到相邻的列中
    def predict(self,X):
        return np.argmax(self.PredictProbability(X),axis=1)#返回特征值属于标签0的概率和属于1的概率的最大值，即预测数据的分类情况
"""
数据标准化函数
标准化数据，防止在计算过程中数据超出计算范围
"""
class Standardize():
    def st(self,X):
        self.X = X
        self.std = np.std(self.X)#计算数据的方差
        self.mean = np.mean(self.X)#计算数据的均值
        return (self.X - self.mean) / self.std#返回数据标准化的结果
"""
预测准确率计算函数
"""
def accuracy(X,y,theta):
    X=np.insert(X,0,1,axis=1)
    z = np.dot(X, theta)
    h_theta = sigmoid(z)
    y_hat=np.argmax(np.concatenate([1-h_theta,h_theta], axis=1),axis=1)#计算特征集的标签预测集
    temp=np.where(y==y_hat.ravel())#计算标签预测集与标签真实集相同的索引
    return len(temp[0])/len(y)#返回准确率

s_train = Standardize()
train_X_st = s_train.st(train_X)#标准化训练集
s_test = Standardize()
test_X_st = s_test.st(test_X)#标准化测试集

aa = LogisticRegression_NM(6)
aa.fit(train_X_st, train_y)#训练数据
train_y_hat = aa.predict(train_X_st)#用训练好的权值预测训练集
test_y_hat = aa.predict(test_X_st)#用训练好的权值预测测试集
print(aa.cost)#输出损失函数
"""
计算每一次迭代后的准确率
"""
ac1=[]#训练集预测准确率
for i in range(aa.times):
    ac1.append(accuracy(train_X_st,train_y,aa.theta[i]))
ac2=[]#测试集预测准确率
for i in range(aa.times):
    ac2.append(accuracy(test_X_st,test_y,aa.theta[i]))
print(ac1,'\n',ac2)
"""
可视化
"""
#将标签为1的数和标签为0的数分离开来，以方便画散点图描绘数据分布
index0_train=np.where(train_y==0)
index0_test=np.where(test_y==0)
index1_train=np.where(train_y==1)
index1_test=np.where(test_y==1)
X0=np.append(train_X[index0_train],test_X[index0_test],axis=0)#将标签为0的数据放在一起
X1=np.append(train_X[index1_train],test_X[index1_test],axis=0)#将标签为1的数据放在一起
jj=np.append(train_X,test_X,axis=0)
kk=np.append(train_y,test_y,axis=0)
fig= plt.figure(figsize=(15,5))#创建画布，大小为5x15
ax0=plt.subplot(131,xlim=(15, 65), ylim=(40, 90))#创建第一个坐标轴,绘制数据分布及决策界限
plt.xlabel("Exam1")#设置x轴标签
plt.ylabel("Exam2")#设置y轴标签
plt.title("Classification Line")#设置标题
ax1=plt.subplot(132,xlim=(0, 5), ylim=(23, 50))#创建第二个坐标轴，绘制损失函数下降情况
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Function Change")
ax2=plt.subplot(133,xlim=(0, 5), ylim=(0, 1))#创建第三个坐标轴，绘制准确率变化情况
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.title("Logistic Regression-Newton's Method Accuracy Change")
a1=ax0.scatter(X1[:,0], X1[:,1], c='green', marker='x',label='Admitted')#在第一个坐标轴上绘制标签为0数据的分布情况
a2=ax0.scatter(X0[:,0], X0[:,1], c='red', marker='o',label="Not admitted")#在第一个坐标轴上绘制标签为1数据的分布情况
ax0.legend(loc='lower center')#显示图例并设置图例位置
"""
绘制动态图，描绘决策界限，损失函数以及预测准确率随时间的变化
"""
line0, = ax0.plot([], [], lw=2)#初始化线条，必须有逗号，,表示得到元组
line1, = ax1.plot([],[], lw=2,marker='o')
sca2= ax2.scatter([],[], label="training set", s=100)
sca3= ax2.scatter([],[], label="test set",s=100)
exam1 = np.linspace(0, 80, 100)#在0~80间均匀生成100个数
iteration = np.arange(0, 6)
def animate(i):#动画更新函数
    exam2 = -(aa.theta[i][0] + aa.theta[i][1] * (exam1 - s_train.mean) / s_train.std) / aa.theta[i][2] * s_train.std + s_train.mean#计算exam2的值随exam1的变化，由于优化权值时用的是标准化后的数据，此时需要反标准化
    line0.set_data(exam1,exam2)#更新exam1与exam2随迭代次数变化的值
    line1.set_data(iteration[:i], aa.cost[:i])#更新损失函数随迭代次数变化的值
    sca2.set_offsets(np.stack((iteration[:i],ac1[:i]),axis=1))#更新训练集准确率随迭代次数变化的值
    sca3.set_offsets(np.stack((iteration[:i],ac2[:i]),axis=1))#更新测试集准确率随迭代次数变化的值
    return line0,line1,sca2,sca3
anim = animation.FuncAnimation(fig,func=animate,frames=7, interval=200,blit=True)#创建动画
plt.legend()
plt.show()
#anim.save('LogisticRegression_NM.gif',writer='imagemagick',fps=5)#保持动态图（需要安装imagemagick）"""