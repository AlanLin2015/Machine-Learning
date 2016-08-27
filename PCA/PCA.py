# -*- coding: utf-8 -*-
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

def loaddataset(filename,spam='\t'):
    fr=open(filename)
    dataarr=[line.strip().split(spam) for line in fr.readlines()]
    floatarr=[map(float,line) for line in dataarr]
    return mat(floatarr)
    
filename1='C:\Users\Alan Lin\Desktop\machinelearningdata\\testSet.txt'
filename2='C:\Users\Alan Lin\Desktop\machinelearningdata\\secom.data'
    
    
def pca(datamat,topNfeat=9999999):
    meanvals=mean(datamat,axis=0)   #计算平均值
    meanremoved=datamat-meanvals   #所有数据减去平均值
    covmat=cov(meanremoved,rowvar=0)   #计算协方差
    eigvals,eigvects=linalg.eig(mat(covmat))   #计算特征值，特征向量，一般有n列就有n*n的特征向量矩阵
    eigvalind=argsort(eigvals)   #给特征值排序
    eigvalind=eigvalind[:-(topNfeat+1):-1]   #逆序找到前topNfeat最大的特征值的位置
    redeigvects=eigvects[:,eigvalind]   #找到对应的前topNfeat个特征向量
    print 'shape(redeigvects):',shape(redeigvects)
    print 'shape(meanremoved):',shape(meanremoved)
    lowddatamat=meanremoved*redeigvects   #通过这个topNfeat特征向量，将数据降到topNfeat维度
    print 'shape(lowddatamat):',shape(lowddatamat)
    reconmat=(lowddatamat*redeigvects.T)+meanvals   #再重构数据，返回到原来的维度，此时已经删除噪点了,这里重构后的数据有些不明白
    print 'shape(reconmat):',shape(reconmat)
    return lowddatamat,reconmat

def disp():
    data=loaddataset(filename1)
    lowdmat,reconmat=pca(data,1)
    fig=plt.figure()
    ax1=fig.add_subplot(111)
    ax1.scatter(data[:,0].flatten().A[0],data[:,1].flatten().A[0],marker='^',s=90)
    ax1.scatter(reconmat[:,0].flatten().A[0],reconmat[:,1].flatten().A[0],marker='o',s=50,c='red')

    
    