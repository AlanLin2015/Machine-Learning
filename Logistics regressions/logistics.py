# -*- coding: utf-8 -*-
from numpy import *


# -*- 导入数据 -*-
def loadDataset():
    datamat=[];labelmat=[]
    fr=open('E:\Users\Alan Lin\Desktop\\machinelearningdata\\testSet.txt')
    for line in fr:
        linearr=line.strip().split()  #把每一行的空格都去掉用\t代替，再用split()把\t转化为逗号
        datamat.append([1.0,float(linearr[0]),float(linearr[1])])  #把每组数据导入datamat数组中，这里加了中括号，导入后如果转成矩阵则为3列
        labelmat.append(int(linearr[2]))   #以行的形式把标签变量导入labelmat数组中，如果转成矩阵则是一个行向量
    return datamat,labelmat
    
# -*- 计算sigmoid _*_
def sigmoid(inX):
    return 1.0/(1+exp(-inX))   #计算sigmoid
    

# _*_ 用矩阵的形式计算w值，计算量大，涉及全部数据 _*_
def gradascent(datamatin,labelmatin):
    datamatrix=mat(datamatin)   #转化为numpy矩阵
    labelmatrix=mat(labelmatin).transpose()  #转化为numpy矩阵，且转置
    alpha=0.001
    m,n=shape(datamatrix)   #取m为行，n为列
    weights=ones((n,1))   #创建一个n行1列的全一矩阵
    maxcycles=500
    for i in range(maxcycles):
        h = sigmoid(datamatrix*weights)   #将所有的W0X0+W1X1+W2X2的结果构成一个100*1的矩阵
        error=(labelmatrix-h)   #错误值为标签量（100*1）-这个矩阵
        weights=weights+alpha*datamatrix.transpose()*error   #重新计算这个weight的值，结果为3*1的矩阵
    return weights


# _*_ 优化后的算法，随机抽取训练集的行数据进行迭代，迭代次数与矩阵算法可能差不多，但是抽样的随机性可能让它很早就收敛了 _*_
def stocgradascent(datamat,classlabels,numiter=150):  #以数值形式运算，减少运算次数
    datamatrix=array(datamat)   #在h计算中datamatrix必须为数组，因为weights为数组
    m,n=shape(datamatrix)   #取datamarix的行和列
    weights=ones(n)   #构建一个全1的数组
    for j in range(numiter):   #迭代输入指定迭代次数
        dataIndex=range(m)   #构建一个0到m-1的数组作为是否已经取样的标记
        for  i in range(m):   #迭代行数
            alpha=4/(1.0+j+i)+0.01   #将alpha的取值设为可逐渐变小的值，但不会为零
            randIndex=int(random.uniform(0,len(dataIndex)))   #取0到m的随机数
            h=sigmoid(sum(datamatrix[randIndex]*weights))   #取随机一行作为sigmoid的计算值
            error=classlabels[randIndex]-h   #计算错误值
            weights=weights+alpha*error*datamatrix[randIndex]   #计算weights值
            del(dataIndex[randIndex])   #删除标记中已经取过样的值，避免重复取值
    return weights
    

# -*- 判断函数 _*_    
def classifyvector(inX,weights):
    prob=sigmoid(sum(inX*weights))
    if prob >0.5:return 1.0
    else:return 0.0

# -*- 导入数据并计算w值，返回w值 _*_
def colictest():
    frtrain=open('E:\Users\Alan Lin\Desktop\\machinelearningdata\\horseColicTraining.txt')
    frtest=open('E:\Users\Alan Lin\Desktop\\machinelearningdata\\horseColicTest.txt')
    trainingset=[];traininglabels=[]
    for line in frtrain.readlines():
        currline=line.strip().split('\t')
        m=len(currline)
        linearr=[]
        for i in range(m-1):
            linearr.append(float(currline[i]))
        trainingset.append(linearr)
        traininglabels.append(float(currline[21]))
    trainweights=stocgradascent(array(trainingset),traininglabels,500)
    return trainweights 



    
# -*- 画图函数 _*_
def plotbestfit(weights):  #输入的weights必须是数组，weights=array(weights)
    import matplotlib.pyplot as plt
    datamat,labelmat=loadDataset()
    dataArr=array(datamat)   #将输入的datamat转化为数组格式，数组与矩阵的区别在于数组是n维的，而矩阵只能是二维的
    n=shape(dataArr)[0]  #取行值
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):   #迭代每一行
        if int(labelmat[i])==1:   #若标签是1
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])  #取每行的第二、三值给xcord1、ycord1
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])  #否则取每行的第二、三值给xcord2，ycord2
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=arange(-3.0,3.0,0.1)   
    y=(-weights[0]-weights[1]*x)/weights[2]  #描拟合分割线，0=W0X0+W1X1+W2Y,XO取1.0，Y=(-W0-W1X1)/W2
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('Y1');
    plt.show