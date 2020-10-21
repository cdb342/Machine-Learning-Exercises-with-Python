# NJUST机器学习课后作业python实现
- ## 神经网络
### I.以两门考试成绩数据集进行二分类
![](./res/ThreeLayers_NeuralNetwork_Exam.gif)
### II.以鸢尾花数据集进行三分类
![](./res/ThreeLayers_NeuralNetwork_Iris.gif)
- ## 感知机
### I.以两门考试成绩数据集进行二分类，Mini_batch梯度下降
### 因为梯度下降会遇到局部极小值点，所以用了Adam算法，因为数据集线性不可分，所以最后不能收敛
![](./res/Perceptron_Mini_batch_Adam.gif)
### II.使用Multiclass感知机以两门考试成绩数据集进行二分类，Mini_batch梯度下降
### 与以上相似
![](./res/Perceptron_Multi_Class_Mini_batch_Adam.gif)
### II.使用Multiclass感知机以鸢尾花数据集进行三分类，Mini_batch梯度下降，使用了Adam算法
![](./res/Perceptron_3Classes_Mini_batch_Adam.gif)
- ## softmax回归
### 用鸢尾花数据集对3种鸢尾花分类
### I.Softmax回归梯度下降可视化结果
![](./res/SoftmaxRegression_GD.gif)
### II.Softmax回归随机梯度下降可视化结果
![](./res/SoftmaxRegression_SGD.gif)
- ## 逻辑回归
### 以两门考试成绩数据集对过和不过的情况进行分类
###  I.逻辑回归梯度下降可视化结果
![](./res/LogisticRegression_GD.gif)
### II.逻辑回归随机梯度下降可视化结果
![](./res/LogisticRegression_SGD.gif)
### III.逻辑回归牛顿法优化可视化结果
![](./res/LogisticRegression_NM.gif)
- ## 线性回归
### 以2010-2013年的房价为训练集，预测2014年房价
### I.线性回归最小二乘法优化可视化结果
![](./res/LinearRegression_close_form_Fitting_Line.jpg)
### II.线性回归梯度下降可视化结果
![](./res/Linear_Regression_GradientDescend.gif)
