# -*- coding: utf-8 -*-
from numpy import *
import operator

from os import listdir

'''k近邻算法中心程序'''
def classify0(inX,dataset,labels,k): #inX输入量 dataset训练样本量 labels标签向量 k选择近邻数目
    datasetsize=dataset.shape[0] #shape[0]表示取dataset的行数
    diffmat = tile(inX,(datasetsize,1)) - dataset #先把输入量的行数转换成与dataset一样，再求差，矩阵求差
    sqdiffmat=diffmat**2
    sqdistance=sqdiffmat.sum(axis=1) #列向量相加
    distance=sqdistance**0.5  #取开方，这里用的是欧式距离
    sortdistance=distance.argsort() #增序排序并取标签
    classcount={}
    for i in range(k):    #迭代前K个样本
        vlabels=labels[sortdistance[i]]   #取前K个样本的标签
        classcount[vlabels]=classcount.get(vlabels,0)+1  #将每个标签对应的值（这里是频率）取出来并构成一个字典
    sortedclasscount=sorted(classcount.iteritems(),key=operator.itemgetter(1),reverse=True) 
    '''iteritems是输出字典的键值对，对字典的第二个域（即字典的值）进行排序，reverse降序排序'''
    return sortedclasscount[0][0]
    
def creatDataset():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels


# -*-初始化导入文件，使导入数据切割为一个二维矩阵和一个标签数组-*-
def file2matrix(filename):
    fr=open(filename)   #打开文件filename
    arrayoflines = fr.readlines()   #读取行
    numberoflines=len(arrayoflines)   #计算行数
    returnmat=zeros((numberoflines,3))   #创造一个同行数，3列的二维0矩阵
    classlabelsvector = []   #创造一个数组作为标签的储存地
    index = 0   #作为行数叠加的标志
    for line in arrayoflines:   #迭代每一行
        line=line.strip()
        lineformline = line.split(',')
        returnmat[index,:] = lineformline[0:3]
        classlabelsvector.append(int(lineformline[-1])) 
        index += 1
    return returnmat,classlabelsvector   
 
    
       
# -*-标准化矩阵里的数字-*-   

def autoNorm(dataset):
    minvals=dataset.min(0)   #axis=0，取每列的最小值，表现在行上
    maxvals=dataset.max(0)   #axis=1，取每列的最大值，表现在行上
    ranges=maxvals-minvals   #计算每列的最大值减最小值
    normDataSet = zeros(shape(dataset))   #shape(dataset)取得dataset的行与列数，这里为（1000，）
    m = dataset.shape[0]   #取得dataset的行数，这里为1000
    normDataSet=dataset - tile(minvals,(m,1))   #计算原始数据减去最小值，这里最小值的行数需要与原始数据相同、
    normDataSet=normDataSet/tile(ranges,(m,1)) #计算标准化的值
    return normDataSet,ranges,minvals
    
# -*- 取前10%的数据作为测试数据-*-
def datingclasstest():
    hoRatio=0.10             #用10%的数据作为测试量
    dD,dL=file2matrix('E:\Users\Alan Lin\Desktop\\d.txt') #得到样本量，标签
    normMat,ranges,minvals=autoNorm(dD)   #得到标准化的值，极差，最小值
    m=normMat.shape[0]   #得到标准化的值的行数
    numTestVecs=int(m*hoRatio)   #计算前10%的具体数目
    errorCount = 0.0   #错误数目累加器
    for i in range(numTestVecs):   #前10%的数据开始迭代
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],dL[numTestVecs:m],3)
        print 'the classifier came back with: %d,the real answer is %d' % (classifierResult,dL[i])
        if     classifierResult != dL[i]:
            errorCount += 1.0
    print 'the total error rate is: %f' % (errorCount/float(numTestVecs))
    
# -*- 输入目标数据量，并初始化数据，标准化数据，完成k近邻算法，输出结果-*-
def classfying():
    outputlist=['very 喜欢','small like','didntlike']
    flytime=float(raw_input('how long time did you fly in a year?'))
    gametime=float(raw_input('how long time did you spend on playing games?'))
    icecreams=float(raw_input('how much icecream did you eat in a year?'))
    arr=array([flytime,gametime,icecreams])
    dD,dL=file2matrix('E:\Users\Alan Lin\Desktop\\d.txt')
    normMat,ranges,minvals=autoNorm(dD)
    classfying=classify0((arr-minvals)/ranges,normMat,dL,3)
    print 'the most is:',outputlist[classfying-1]  
    
# _*_ 将文件中的32*32矩阵转换成1*1024列 _*_
def to_32(filename):
    returnos=zeros((1,1024))   #构造一个1行1024列的零矩阵
    ma=open(filename)   
    for i in range(32):   #从0迭代到31
        lintr=ma.readline()   #readline()为阅读每行函数，这里根据迭代一行一行阅读
        for j in range(32):   #迭代列
            returnos[0,i*32+j]=lintr[j]   #将每一行的数据给returnos矩阵中
    return returnos   #转换完毕，返回returnos
    
    
    
""" 作用同to_32
def to_322(filename):
    returnoss=zeros((1,1024))
    maa=open(filename)    
    i=int(0)
    for line in maa.readlines():   #这里用的是readlines()函数，即一次性阅读所有行的数据
        for j in range(32):
            returnoss[0,i*32+j]=line[j]
        i += 1
    return returnoss
"""


# _*_识别手写数字的程序 _*_

def recognationos():
    trainingtation=listdir('C:\Users\Alan Lin\Desktop\machinelearningdata\digits\\trainingDigits')#用lsitdir函数获取试验向量文件里的每个文件名
    m=len(trainingtation)   #计算所有文件的数量总和
    trainingclocks=zeros((m,1024))   #创造m行1024列的矩阵
    traininglabels=[]   #创造一个空列表用于储存试验向量的标签
    for i in range(m):   #迭代文件数量
        fN=trainingtation[i]   #迭代选取文件名
        fS=fN.split('.')[0]   #通过‘.’切割文件名，并取第一个域
        fS=int(fS.split('_')[0])   #通过‘_’切割剩余文件名，并去第一个域，即表示数字的值
        traininglabels.append(fS)   #把数字标签加入标签向量列表中
        trainingclocks[i,:]=to_32('C:\Users\Alan Lin\Desktop\machinelearningdata\digits\\trainingDigits/%s' % fN)   #把每个文件中的32*32矩阵转换成1*1024的矩阵
    
    
    errornum=0.0   #初始化判断错误数量
    testtation=listdir('C:\Users\Alan Lin\Desktop\machinelearningdata\digits\\testDigits')   #获取测试向量文件的每个文件
    mt=len(testtation)   #计算所有文件的数量总和
   # testclocks=zeros((mt,1024))   #创造mt行1024列的矩阵
    testlabels=[]   #创造一个空列表用于储存测试向量的标签
    for i in range(mt):   
        tfN=testtation[i]   
        tfS=tfN.split('.')[0]
        tfS=int(tfS.split('_')[0])
        #testlabels.append(tfS)
        testclocks=to_32('C:\Users\Alan Lin\Desktop\machinelearningdata\digits\\testDigits/%s' % tfN)   #将每一个文件中的32*32矩阵转换为1*1024矩阵
        classifyresult=classify0(testclocks,trainingclocks,traininglabels,3)   #将得到的测试向量1*1024矩阵用k近邻算法与试验向量进行距离计算并得到预测类别
        if classifyresult !=tfS:   #将得到的预测的类别与测试向量的标签进行对比，若不相等，则错误参数加1
            errornum += 1.0
    print 'the errornum percent is: %f' % (errornum/float(mt))   #输出判断错误率
        

    
    



    
