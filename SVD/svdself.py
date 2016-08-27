# -*- coding: utf-8 -*-
from numpy import *
from numpy import linalg as la


def loaddata():
    return [[2, 2, 0, 2, 2],[2, 0, 0, 3, 3],[2, 0, 0, 1, 1],[1, 1, 1, 0, 0],[2, 2, 2, 0, 0],[5, 5, 5, 0, 0],[1, 1, 1, 0, 0]]
    
def loaddata2():
    return [[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]



def ecludSim(inA,inB):   #欧式相关度，1.0/（1.0+欧氏距离）
    similiar=1.0/(1.0+la.norm(inA,inB))   #假定inA和inB都是列向量
    return similiar
    
def pearsSim(inA,inB):   #皮尔逊相关度
    if len(inA) < 3: return 1.0   #如果inA的长度小于3，则是完全相关
    similiar=0.5+0.5*corrcoef(inA,inB,rowvar=0)[0][1]   #皮尔逊相关系数取值为[-1,1]，所以需要0.5来归一化

def cosSim(inA,inB):   #余弦相关度
    num=float(inA.T*inB)
    denom=la.norm(inA)*la.norm(inB)
    similiar=0.5+0.5*(num/denom)
    return similiar
    
def standest(datamat,user,similiarmeans,item):   #非svd推荐方法
    n=shape(datamat)[1]   #取列数
    similiartotal=0.0;ratsimiliartotal=0.0   #初始化相关度值，相关度分数值
    for j in range(n):
        userrating=datamat[user,j]   #取user行中有数值的列
        if userrating == 0: continue
        lop=nonzero(logical_and(datamat[:,item].A>0,datamat[:,j].A>0))[0]   #并且其他行的这一列也有数值
        if len(lop) == 0: similiar=0.0
        else:similiar=similiarmeans(datamat[lop,item],datamat[lop,j])   #计算相关度，这里采用余弦相关度
        print "the item %d with the j %d is similiar %f:" % (item,j,similiar)
        similiartotal += similiar   #累加所有相关度
        ratsimiliartotal += similiar*userrating   #累加所有相关度分数值
    if similiartotal == 0:return 0
    return ratsimiliartotal/similiartotal

def svdest(datamat,user,similiarmeans,item):   #svd推荐方法
    n=shape(datamat)[1]
    similiartotal=0.0;ratsimiliartotal=0.0
    U,sigma,VT=la.svd(datamat)   #svd分解
    sig4=mat(eye(4)*sigma[:4])   #取前n个特征值
    xform=datamat.T*U[:,:4]*sig4.I   #重构n个特征值的矩阵，.I为矩阵的逆
    for j in range(n):
        userrating=datamat[user,j]
        if userrating == 0 or j == item :continue
        similiar=similiarmeans(xform[item,:].T,xform[j,:].T)   #重构后的矩阵计算方法不一样，因为cosSim是列向量，所以需要转置
        print "the item %d with the j %d is %f:" % (item,j,similiar)
        similiartotal += similiar
        ratsimiliartotal += similiar*userrating
    if similiartotal == 0:return 0
    return ratsimiliartotal/similiartotal
    
    
def recommend(datamat,user,N=3,similiarmeans=cosSim,method=svdest): #数据集，用户（行数），前N个推荐，相关度方法选择，推荐方法选择
    noratingitems=nonzero(datamat[user,:].A == 0)[1]   #输出这一行中为0的列位置
    if len(noratingitems) == 0: print "all rates full"   #如果这一行都没有为0，说明这些菜品这个用户都吃过，就没有推荐的意义了。
    scorelist=[]
    for item in noratingitems:
        scores=method(datamat,user,similiarmeans,item)   #调用推荐方法得到分数
        scorelist.append([item,scores])
    return sorted(scorelist,key=lambda jj:jj[1],reverse=True)[:N]   #按照第二列逆序排序并取前N个

#求能量 sig2=sigma**2  总能量=sum(sig2)  90%总能量=sum(sig2)*0.9 前n个能量=sum(sig2[:n])