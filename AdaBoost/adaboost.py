# -*- coding: utf-8 -*-
from numpy import *
'''创建测试集'''
def loadsimpdata():
    datamat=matrix([[1.,2.1],[2.,1.1],[1.3,1.],[1.,1.],[2.,1.]])
    classlabels=[1.0,1.0,-1.0,-1.0,1.0]
    return datamat,classlabels
'''单层对比函数，用于构建单层决策树和分类应用时的引用'''
def stumpclassify(datamat,i,threshval,threshineq):
    retarray=ones((shape(datamat)[0],1))   #构建一个同行1列的全一矩阵
    if threshineq=='lt':   #如果判断词为'lt',进行以下操作
        retarray[datamat[:,i] <= threshval] = -1.0   #将第i列的全部值与参照值threshval对比，小于等于它的标签置为-1.0
    else:
        retarray[datamat[:,i] > threshval] = -1.0   #同上操作，大于它的标签置为-1.0，它们的错误率的和不一定是1，只有正负样本十分平均的情况下。
    return retarray   #返回判断后的标签向量


'''构建决策树函数，其中引用了stumpclassify函数'''
def buildstump(dataarr,classlabels,D):
    datamat=mat(dataarr);labelmat=mat(classlabels).T   #矩阵化数据集和标签集
    m,n=shape(datamat)
    bestclasseat = mat(zeros((m,1)))  #创建最好的分类标签集，初始化为全零矩阵
    numsteps = 10.0 ; beststump = {}   #设置总步长为10，并初始化最可靠的单层决策树字典形式
    minerror=inf   #错误率初始化为无穷大
    for i in range(n):   #第一层循环，迭代数据集的每一列
        datamin=datamat[:,i].min();datamax=datamat[:,i].max()   #取这列中数据集的最大值和最小值
        stepsize=(datamax-datamin)/numsteps   #每一步的步长为最大值与最小值的差除以总步长
        for j in range(-1,int(numsteps)+1):   #第二层循环，循环总步数
            for threshineq in ['lt','gt']:   #第三层循环，循环是大于还是小于等于
                threshval=(datamin+float(stepsize)*float(j))   #计算参照值，它是逐步递增的
                prediction=stumpclassify(datamat,i,threshval,threshineq)   #计算预测标签集
                errarr=mat(ones((m,1)))   #初始化错误标签向量
                errarr[prediction == labelmat] = 0   #将预测标签与标签集一致的样本变成非错误标签
                weighterror=D.T*errarr   #计算错误率
                '''错误率并不是简单地把未正确的样本数除以所有的样本数的这样一个定义算法。它实际计算有加权的，
                一旦上一次某个样本分类错误，就赋予超高权重，在第二次分类时肯定是正确的，而此时的α的值也会
                严重偏向那个样本，使得整体α向那个样本靠近，从而修正那个样本。'''
                if weighterror < minerror:   #如果这次的错误率小于最小的错误率
                    minerror=weighterror    #则把这次的错误率当成最小的错误率
                    bestclasseat=prediction.copy()   #将预测标签给最佳标签集
                    beststump['dim']=i   #将列数给字典显示
                    beststump['thresh']=threshval   #参照值
                    beststump['ineq']=threshineq   #大于还是小于等于的标志
    return beststump,minerror,bestclasseat   #返回最佳决策树字典，最小错误率，最佳标签集
    
'''用单层决策树构建adaboost算法，用多个决策树构建多个弱分类器，再计算每个弱分类器的权重'''
def adaboosttrainDS(dataarr,classlabels,numiter=40):   
    weakclassarr=[]   #初始化决策树集合列表
    m=shape(dataarr)[0]
    D=mat(ones((m,1))/m)   #D的初始值为总体均值
    aggclasseat=mat(zeros((m,1)))   #初始化加权后的标签集
    for i in range(numiter):
        beststump,minerror,bestclasseat=buildstump(dataarr,classlabels,D)
        print 'D:',D.T
        alpha=float(0.5*log((1-minerror)/max(minerror,1e-16)))   #计算alpha
        beststump['alpha']=alpha   #给字典添加alpha值
        weakclassarr.append(beststump)   #把这个决策树添加到列表中
        print 'classest:',bestclasseat.T
        expon=multiply(-1*alpha*mat(classlabels).T,bestclasseat)
        D=multiply(D,exp(expon))
        D=D/D.sum()   #计算D
        aggclasseat += alpha*bestclasseat  #计算总alpha加权值，得到的和就是估计标签值
        print 'aggclasseat:',aggclasseat.T
        aggerrors=multiply(sign(aggclasseat) != mat(classlabels).T,ones((m,1)))   #将估计标签值经过函数转化为标签值
        errorrate=aggerrors.sum()/m
        print 'total error:',errorrate,'\n'
        if errorrate == 0.0:break
    return weakclassarr
    
'''分类应用函数，调用每个弱分类器计算每个分类指标，再通过alpha计算总的分类标志'''
def adaclassify(datatest,classarrall):   
    datatest=mat(datatest)
    m,n=shape(datatest)
    aggclasseat=zeros((m,1))
    for i in range(len(classarrall)):   #迭代每个分类器
        classeat=stumpclassify(datatest,classarrall[i]['dim'],classarrall[i]['thresh'],classarrall[i]['ineq'])   #计算这个分类器得到的标签
        aggclasseat += classarrall[i]['alpha']*classeat
    return sign(aggclasseat)
    

'''导入训练集'''
def loaddatasettrain():
    m=len(open('E:\Users\Alan Lin\Desktop\machinelearningdata\\horseColicTraining2.txt').readline().strip().split('\t'))
    datamat=[];labelmat=[]
    fr=open('E:\Users\Alan Lin\Desktop\machinelearningdata\\horseColicTraining2.txt')
    for line in fr.readlines():
        linearr=[]
        dataarr=line.strip().split('\t')
        for i in range(0,m-1):
            linearr.append(float(dataarr[i]))
        datamat.append(linearr)
        labelmat.append(float(dataarr[-1]))
    return datamat,labelmat

'''导入测试集'''
def loaddatasettest():
    m=len(open('E:\Users\Alan Lin\Desktop\machinelearningdata\\horseColicTest2.txt').readline().strip().split('\t'))
    datamat=[];labelmat=[]
    fr=open('E:\Users\Alan Lin\Desktop\machinelearningdata\\horseColicTest2.txt')
    for line in fr.readlines():
        linearr=[]
        dataarr=line.strip().split('\t')
        for i in range(0,m-1):
            linearr.append(float(dataarr[i]))
        datamat.append(linearr)
        labelmat.append(float(dataarr[-1]))
    return datamat,labelmat
'''解释说明：主要用adaboosttrainDS(dataarr,classlabels,numiter=40)函数构建分类器，这里输入训练集，获得分类器和
   alpha值后，用adaclassify(datatest,classarrall)应用分类器，这里输入测试集。'''
























