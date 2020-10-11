import numpy as np
X=np.loadtxt("./dataset/price/x.txt")
y=np.loadtxt("./dataset/price/y.txt")
class LinearRegression:
    def __init__(self,alpha,times):
        self.alpha=alpha
        self.times=times
    def fit(self,X,y):
        self.X=np.insert(X.reshape((-1,1)),0,1,axis=1)
        self.y=y.reshape((-1,1))
        self.cost=[]
        self.theta=[np.zeros(2).reshape((2,1))]
        for i in range(self.times):
            y_hat=np.dot(self.X,self.theta[i])
            error=y_hat-self.y
            self.cost.append(np.sum(error**2)/2)
            self.theta.append(self.theta[i]-self.alpha*np.dot(self.X.T,error))
    def predict(self,X):
        self.X=np.insert(X.reshape((-1,1)),0,1,axis=1)
        result=np.dot(self.X,self.theta[self.times])
        return result
X_std=np.std(X)
y_std=np.std(y)
X_mean=np.mean(X)
y_mean=np.mean(y)
X_=(X-X_mean)/X_std
y_=(y-y_mean)/y_std
X_test=(2014-X_mean)/X_std
aaa=LinearRegression(0.01,30)
aaa.fit(X_,y_)
y_hat=aaa.predict(X_)*y_std+y_mean
y_test=aaa.predict(X_test)*y_std+y_mean
print(aaa.theta)
print(y_hat,y_test)
print(aaa.cost)

import matplotlib.pyplot as plt
from matplotlib import animation

fig=plt.figure(figsize=(10,10))
ax = plt.axes(xlim=(2000, 2015), ylim=(0, 15))
ax.plot(X,y,"ro-",label="real value")

line, = ax.plot([], [], lw=2)
def init():
    line.set_data([], [])
    return line,
def animate(i):
    xx=np.insert(X_.reshape((-1,1)),0,1,axis=1)
    zz=np.dot(xx,aaa.theta[i])*y_std+y_mean
    line.set_data(X,zz)
    return line,
animation.FuncAnimation(fig,animate,init_func=init,
                             frames=29,interval=200,blit=True)

"""plt.plot(2014,y_test,"o-",label="test value")
plt.xlabel("years")
plt.ylabel("price")
plt.title("Linear Regression-GradientDescend")
plt.legend()"""
plt.show()
