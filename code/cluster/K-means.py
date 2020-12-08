import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_table('../../dataset/GMM/GMM8.txt')#读取数据集
X=np.stack((data['x1'],data['x2']),axis=1)
class K_means():
    def __init__(self,X,class_num=3,epoch_num=5):
        self.X=X
        self.class_num=class_num#类别数
        random_index=np.random.choice(range(self.X.shape[0]),size=class_num,replace=False)#随机选几个点作为初始点
        self.means=self.X[random_index]#初始聚类点
        self.epoch=epoch_num#迭代次数
    def fit(self):
        dd=0#每个类别对应的点
        plt.figure()
        ax=plt.axes()
        for _ in range(self.epoch):#迭代epoch_num次
            dd=[np.zeros((1,self.X.shape[1]))]*len(self.means)#每次聚类开始时清空每个类别对应的点列表
            for i in self.X:#给每个点分配类别
                tem=np.sum((i-self.means)**2,axis=1)#计算点和每个聚类点的距离
                index=int(np.argmin(tem))#找出距离最小的聚类点
                dd[index]=np.append(dd[index],i.reshape(1,-1),axis=0)#把这个点放到距离最小的聚类点列表里
            plt.ion()
            plt.cla()#清空画布
            for i in range(len(dd)):#分配好点之后进行可视化（由于颜色比较少，不同类别的中心点可能是同一颜色，但是不影响观察）
                ax.scatter(dd[i][:,0],dd[i][:,1])
                ax.scatter(self.means[i][0],self.means[i][1],marker='X',s=200)
            plt.pause(0.5)#暂停0.5秒，便于观察
            plt.ioff()
            for i in range(len(self.means)):#更新聚类点
                self.means[i]=np.mean(dd[i],axis=0)
if __name__=='__main__':
  a=K_means(X,8,19)
  X_dic=a.fit()


