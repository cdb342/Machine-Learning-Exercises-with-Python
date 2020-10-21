import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
"""
Import datasets
train_X:feature of training set   train_y:classes of training set
test_X:feature of test set        test_y:classes of test set
"""
train_X=np.loadtxt("./dataset//Exam/train/x.txt")
train_y=np.loadtxt("./dataset/Exam/train/y.txt").reshape(-1,1)#Transfer the classes set into column vector
test_X=np.loadtxt("./dataset/Exam/test/x.txt")
test_y=np.loadtxt("./dataset/Exam/test/y.txt").reshape(-1,1)#Transfer the classes set into column vector
#print(train_X,'\n',train_y,'\n',test_X,'\n',test_y)
"""
Generate some data for drawing the classification boundary
"""
Exam1 = np.linspace(10, 65, 350)#Generate 350 numbers evenly between 10 and 65
Exam2 = np.linspace(38, 90, 350)#Generate 350 numbers evenly between 38 and 90
XX1, XX2 = np.meshgrid(Exam1, Exam2)#Generate grid point coordinate matrix
XX=np.c_[XX1.ravel(),XX2.ravel()]#Tile and merge XX1 and XX2 to facilitate the prediction of the classification of each point on the grid

def sigmoid(z):#Define sigmoid function
    if np.all(z >= 0):
        return 1.0 / (1 + np.exp(-z))
    else:
        return np.exp(z) / (1 + np.exp(z))
"""
This function will use the variance and mean of dataset Standard_X to standardize the dataset X
input:Standard_X,X
output:(X-Standard_X.mean)/Standard_X.std
"""
def Standardize(Standard_X,X):
    std = np.std(Standard_X)#Calculate the variance of the data
    mean = np.mean(Standard_X)#Calculate the mean of the data
    return (X - mean) / std#Return the result of data standardization
"""
Use the Adam Algorithm to optimize the gradient
"""
def Adam(v,s,g,i):
    v=0.9*v+0.1*g
    s=0.999*s+0.001*g**2
    v_correct = v/ (1 - 0.9 ** i)#Momentum
    s_correct = s / (1 - 0.999 ** i)#RMSprop
    g=v_correct/(np.sqrt(s_correct) + 10 ** (-8))
    return v,s,g
"""
Create neural network class
feedforward():Forward transfer feature set
fit()：Method of training weights
predict():Forecast data classification
accuracy：Calculate the accuracy of data classification
"""
class ThreeLayers_NeuralNetworks():
    def __init__(self,yita,times):#The parameters are the learning rate and the number of iterations
        self.yita=yita
        self.times=times
    def feedforward(self,X):
        Z1=np.dot(X,self.W1)+self.b1
        self.A=sigmoid(Z1)#The output of hidden layer
        Z2=np.dot(self.A,self.W2)+self.b2
        h=sigmoid(Z2)#The output of output layer
        return h
    def fit(self,X,y):
        self.X=X
        self.y=y
        """
        Initialize parameters of neural network
        """
        np.random.seed(2)#Make the result of each run the same
        self.W1 = np.random.randn(2, 2)
        self.W2 = np.random.randn(2, 1)
        self.b1 = np.random.randn(1, 2)
        self.b2=np.random.randn(1)
        
        self.cost=[]#Store the cost of each iteration
        self.Accuracy_train = []#Store the prediction accuracy of each iteration of the training set
        self.Accuracy_test = []#Store the prediction accuracy of each iteration of the test set
        """
        Initialize the parameters of Adam algorithm
        """
        s_W1 = v_W1 = np.zeros((2, 2))
        s_W2 = v_W2 = np.zeros((2, 1))
        s_b1 = v_b1 = np.zeros((1, 2))
        s_b2 = v_b2 = np.zeros(1)
        
        self.ZZ = []#Store the prediction of generating data for drawing the classification boundary
        for i in range(self.times):
            self.ZZ.append(self.predict(XX_st).reshape(np.shape(XX1)))
            h=self.feedforward(self.X)
            y_hat_test = self.predict(test_X_st)#Prediction of ith iteration of the test set
            y_hat = self.predict(self.X)#Prediction of ith iteration of the training set
            ac_test = self.accuracy(y_hat_test, test_y)#The prediction accuracy of ith iteration of the test set
            ac_train = self.accuracy(y_hat, self.y)#The prediction accuracy of ith iteration of the training set
            self.Accuracy_test.append(ac_test)
            self.Accuracy_train.append(ac_train)
            J=np.sum((h-self.y)**2)/2#the cost function of training set
            self.cost.append(J)
            y_mul=np.multiply(np.multiply(h,1-h),h-self.y)#Intermediate parameters
            A_mul=np.multiply(self.A,1-self.A)#Intermediate parameters
            """
            Calculate the gradient of each parameter
            """
            self.g_b2=np.sum(y_mul)
            self.g_W2=np.dot(self.A.T,y_mul)
            self.g_W1=np.dot(self.X.T,np.multiply(A_mul,np.dot(y_mul,self.W2.T)))
            self.g_b1=np.dot(np.dot(y_mul.T,A_mul),np.diag(self.W2.ravel()))
            """
            Use Adam algorithm to optimize the gradient
            """
            s_W1,v_W1,self.g_W1=Adam(s_W1,v_W1,self.g_W1,i+1)
            s_W2, v_W2, self.g_W2 = Adam(s_W2, v_W2, self.g_W2, i+1)
            s_b1, v_b1, self.g_b1 = Adam(s_b1, v_b1, self.g_b1, i+1)
            s_b2, v_b2, self.g_b2 = Adam(s_b2, v_b2, self.g_b2, i+1)
            """
            Update parameters
            """
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
"""
Standardize the dataset
"""
train_X_st=Standardize(train_X,train_X)
test_X_st=Standardize(train_X,test_X)
XX_st=Standardize(train_X,XX)

aa=ThreeLayers_NeuralNetworks(0.15,600)
aa.fit(train_X_st,train_y)
print(aa.Accuracy_test,'\n',aa.Accuracy_train)#Print the prediction accuracy of the test set and training set
print(aa.cost)#Print the cost function
"""
Visualization
"""
fig,ax=plt.subplots(1,3,figsize=(15,5))#Create canvas and axis
ax[0].scatter(train_X[:,0],train_X[:,1],c=train_y,cmap='Dark2',s=10)#Plot the training set distribution on the first axis
ax[0].scatter(test_X[:,0],test_X[:,1],c=test_y,cmap='Dark2',s=10)#Plot the test set distribution on the first axis
ax[0].set_xlabel("Exam1")#Set x-axis label
ax[0].set_ylabel("Exam2")#Set y-axis label
ax[0].set_title("Classification Boundaries")#Set title
ax[1].set_xlim(0,599)#Set the x-axis range
ax[1].set_ylim(2.5,13)#Set the y-axis range
ax[1].set_xlabel("Iteration")
ax[1].set_ylabel("Cost Function")
ax[1].set_title("Cost Change")
ax[2].set_xlim(0,599)
ax[2].set_ylim(0,1)
ax[2].set_xlabel("Iteration")
ax[2].set_ylabel("Accuracy")
ax[2].set_title("Accuracy Change")
cont=ax[0].contourf(XX1,XX2,aa.ZZ[0],alpha=0.2,cmap='Set3')#Draw classification boundaries
line1,=ax[1].plot([],[])#Plot the loss function decline on the second axis
sca2=ax[2].scatter([],[],label='training set',s=10,c='red')#Plot the changes in the prediction accuracy of the training set on the third axis
sca3=ax[2].scatter([],[],label='test set',s=10)#Plot the changes in the prediction accuracy of the test set on the third axis
ite=np.arange(600)#Iteration count array
def animate(i):#Define animation update function
    global cont
    for c in cont.collections:#Speed up the animation speed
        c.remove()
    cont = ax[0].contourf(XX1, XX2, aa.ZZ[i], alpha=0.2, cmap='Set3') #Update classification boundaries
    line1.set_data(ite[:i],aa.cost[:i])#Update the cost for each Iteration
    sca2.set_offsets(np.stack((ite[:i],aa.Accuracy_train[:i]),axis=1))#Update the prediction accuracy of the training set for each Iteration
    sca3.set_offsets(np.stack((ite[:i],aa.Accuracy_test[:i]),axis=1))#Update the prediction accuracy of the test set for each Iteration
    return ax[0],line1,sca2,sca3
ani=animation.FuncAnimation(fig,animate,frames=600,interval=1)#Generate animation
plt.legend()#Show legend
plt.show()
#ani.save('ThreeLayers_NeuralNetwork_Exam.gif',writer='imagemagick',fps=60)#Save dynamic images (imagemagick needs to be installed)
