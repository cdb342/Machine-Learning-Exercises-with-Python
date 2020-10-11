```Python
import numpy as np
import matplotlib.pyplot as plt
X=np.loadtxt("./dataset/price/x.txt")
y=np.loadtxt("./dataset/price/y.txt")
X_train=np.asmatrix(np.insert(X.reshape(-1,1),0,1,axis=1))
y_train=np.matrix(y.reshape(-1,1))
theta=(X_train.T*X_train).I*X_train.T*y_train
test_X=np.asmatrix([[1],[2014]])
test_y=np.asarray(theta.T*test_X).ravel()
y_hat=np.array(X_train*theta).ravel()
print(y_hat,test_y)
plt.figure(figsize=(5,5))
plt.plot(X,y,"ro-",label="true value")
plt.plot(X,y_hat,"go--",label="predict value")
plt.plot(2014,test_y,"o-",label="predict for test data")
plt.title("Linear Regression-close-form")
plt.xlabel("years")
plt.ylabel("price")
plt.legend()
plt.show()
```
