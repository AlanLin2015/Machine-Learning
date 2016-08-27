# -*- coding: utf-8 -*-
from numpy import *
import numpy as np

# _*_ 导入训练集并处理数据生成数据集datamat和标签集labelsmat
def loadDataset():   
    datamat=[];labelsmat=[]
    frtrain=open('E:\Users\Alan Lin\Desktop\machinelearningdata\\testSet.txt')
    for i in frtrain.readlines():
        linearr=i.strip().split('\t')
        datamat.append([float(linearr[0]),float(linearr[1])])
        labelsmat.append(float(linearr[2]))
    return datamat,labelsmat
    
# _*_ 随机选择函数，通过输入i和m，输出在0到m中随机抽取的数j
def selectJrand(i,m):
    j=i
    while(j==i):
        j=int(random.uniform(0,m))
    return j

# _*_ 自定义函数clip
def clip(aj,H,L):
    if aj>H:
        aj=H
    if aj<L:
        aj=L
    return aj

# _*_ 自定义数据结构体，使用时应当用这样的方式：name=optstruct(datamin,classlabels,C,toler)
class optstruct:
    def __init__(selfs,datamatin,classlabels,C,toler):
        selfs.X=datamatin
        selfs.labelmat=classlabels
        selfs.C=C
        selfs.tol=toler
        selfs.m=shape(datamatin)[0]
        selfs.alphas=mat(zeros((selfs.m,1)))
        selfs.b=0
        selfs.ecache=mat(zeros((selfs.m,2)))
# _*_ 计算错误率Ek
def calcek(os,k):
    fxk=float(multiply(os.alphas,os.labelmat).T*(os.X*os.X[k,:].T))+os.b  #仍有问题，这里是f(x)关于alpha的决策分类函数，求解alpha(i)和b(i)
    Ek=fxk-float(os.labelmat[k])   #计算Ek,就是用fxk减去k的标签值
    return Ek
    
# _*_ 用最大步长法获取j的值(计算当时的os.ecache第一列的所有有值的位置对应的错误率Ek的值，与Ei对比找出步长最大的k)
def selectJ(i,os,Ei):
    maxk=-1;maxdeltae=0;Ej=0
    os.ecache[i]=[i,Ei]
    validecachelist=nonzero(os.ecache[:,0].A)[0]  #取ecache的第一列的所有行有值的数对应的位置，因为ecache本身是二维数组，所以nonzero返回的第一列是描述ecache的第一列的，这里去第一列
    if (len(validecachelist)) > 1:
        for k in validecachelist:
            if k==i: continue   #当取得k等于i的时候，重新迭代取下一个k，这里是除了k==i的其他值都计算了一遍
            Ek=calcek(os,k)
            deltaE=abs(Ei-Ek)
            if(deltaE>maxdeltae):
                maxk=k;maxdeltae=deltaE;Ej=Ek
        return maxk,Ej
    else:
        j=selectJrand(i,os.m)
        Ej=calcek(os,j)
    return j,Ej
    
# _*_ 将k和错误率上传到os.ecache全局变量保存
    
def updateEk(os,k):
    Ek=calcek(os,k)
    os.ecache[k]=[1,Ek]
    
    
# _*_ 主函数 _*_
def smop(datamatin,classlabels,C,toler,maxiter,ktup=('lin',0)):
    os=optstruct(mat(datamatin),mat(classlabels).transpose(),C,toler)   #创建数据结构体os保存全局变量
    iters=0   #初始化迭代次数
    entireset=True;alphapairschanged=0   #初始化迭代条件entireset为 True，alphapairschanged为0
    while (iters<maxiter) and ((alphapairschanged>0) or (entireset)):   #退出循环条件，以下任何一个条件满足：1.迭代次数达到了指定次数；2.alphapairschanged=0和entireset为False同时满足；
        alphapairschanged=0
        if entireset:   #假如entireset为True的话，开始遍历所有数据样本，以检验整个集合是否满足KTT条件
            for i in range(os.m): #外循环
                alphapairschanged += innerL(i,os)  #当标签*错误率和alpha[i]同时满足条件时进行计算并返回1，否则返回0
            print 'fullset,iter: %d i:%d,pairs changed %d' %(iters,i,alphapairschanged)  #迭代完毕，打印while迭代次数，i的值，满足条件的次数alphapairschanged
            iters+=1   #while迭代次数加1
        else:
            #print '(os.alphas.A>0):',(os.alphas.A>0)   #返回一个布尔值向量，大于0的为True，小于零的为False
            #print '(os.alphas.A<C):',(os.alphas.A<C)
            #print '((os.alphas.A>0)*(os.alphas.A<C)):',((os.alphas.A>0)*(os.alphas.A<C))  #布尔向量乘布尔向量，对对则对，这里其实是一个取都为对的结果，数学意义上是交集
            nonBoundIs=nonzero((os.alphas.A>0)*(os.alphas.A<C))[0]   #返回非边界样本的i值，就是返回alpha的值在[0,C]之间的alpha的位置，即是alpha对应的i值
            #print 'nonBoundIs:', nonBoundIs
            for i in nonBoundIs:  #外循环
                alphapairschanged += innerL(i,os)   #开始内循环
                print 'non-bound,iter:%d i:%d,pairs changed %d' %(iters,C,alphapairschanged)  #每遍历一个i值，就打印出当时的while迭代次数，i的值，满足条件的次数alphapairschanged
            iters+=1   #while迭代次数加1
        if entireset:entireset=False   #如果entireset为True的话，则令其为False以便进入非边界样本进行内循环
        elif (alphapairschanged==0):entireset=True   #这里有问题，不知道其在遍历整个集合都未对alpha对进行修改时是如何退出循环的
        print 'iteration number: %d' %iters   #打印while迭代次数
    return os.b,os.alphas
   
# _*_内循环函数
def innerL(i,os):
    Ei=calcek(os,i)
    if ((os.labelmat[i]*Ei<-os.tol) and (os.alphas[i]<os.C)) or ((os.labelmat[i]*Ei > os.tol)and(os.alphas[i]>0)): #当标签*错误率和alpha[i]同时满足条件时进行计算并返回1，否则返回0
        j,Ej=selectJ(i,os,Ei)   #用最长步长法选择αj
        alphaIold=os.alphas[i].copy()
        alphaJold=os.alphas[j].copy()
        if (os.labelmat[i] != os.labelmat[j]):   #如果yi不等于yj
            L=max(0,os.alphas[j]-os.alphas[i])
            H=min(os.C,os.C+os.alphas[j]-os.alphas[i])
        else:                                    #如果yi等于yj
            L=max(0,os.alphas[j]+os.alphas[i]-os.C)
            H=min(os.C,os.alphas[j]+os.alphas[i])
        if L==H:print 'L==H';return 0
        n=2.0*os.X[i,:]*os.X[j,:].T-os.X[i,:]*os.X[i,:].T-os.X[j,:]*os.X[j,:].T   #计算η值，η是小于0的
        if n >=0 :print 'n>=0';return 0
        os.alphas[j] -= os.labelmat[j]*(Ei-Ej)/n   #计算αj
        os.alphas[j] = clip(os.alphas[j],H,L)   #筛选αj
        updateEk(os,j)   #上传错误率
        if (abs(os.alphas[j]-alphaJold)<0.00001):   #如果αj的变化量小于阈值，则重新循环
            print 'j not moving enough';return 0
        os.alphas[i] +=os.labelmat[j]*os.labelmat[i]*(alphaJold-os.alphas[j])   #计算αi
        updateEk(os,i)   #上传错误率
        b1=os.b-Ei-os.labelmat[i]*(os.alphas[i]-alphaIold)*os.X[i,:]*os.X[i,:].T-os.labelmat[j]*(os.alphas[j]-alphaJold)*os.X[i,:]*os.X[j,:].T   #计算b1
        b2=os.b-Ej-os.labelmat[i]*(os.alphas[i]-alphaIold)*os.X[i,:]*os.X[j,:].T-os.labelmat[j]*(os.alphas[j]-alphaJold)*os.X[j,:]*os.X[j,:].T   #计算b2
        if (0 < os.alphas[i]) and (os.C>os.alphas[i]): os.b=b1
        elif (0<os.alphas[j]) and (os.C>os.alphas[j]): os.b=b2
        else:os.b=(b1+b2)/2.0   #最后计算b
        return 1
    else: return 0

def calcWs(alphas,dataarr,classlabels):
    X=mat(dataarr);labelmat=mat(classlabels).transpose()
    m,n=shape(X)
    w=zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelmat[i],X[i,:].T)  #X[i,:]为2行1列，这里定了w的格式
    return w
    


    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def smosimple(datamatin,classlabels,C,toler,maxiter):
    datamatrix=mat(datamatin);labelmat=mat(classlabels).transpose()
    b=0;m,n=shape(datamatrix)
    alphas=mat(zeros((m,1)))
    iters=0
    while(iters<maxiter):
        alphapairschanged=0
        for i in range(m):
            fxi=float(multiply(alphas,labelmat).T*(datamatrix*datamatrix[i,:].T))+b
            Ei=fxi-float(labelmat[i])
            if((labelmat[i]*Ei<-toler) and (alphas[i]<C)) or ((labelmat[i]*Ei>toler) and (alphas[i]>0)):
                j=selectjrand(i,m)
                fxj=float(multiply(alphas,labelmat).T*(datamatrix*datamatrix[i,:].T))+b
                Ej=fxj-float(labelmat[j])
                alphaIold=alphas[i].copy()
                alphaJold=alphas[j].copy()
                if (labelmat[i]!=labelmat[j]):
                    L=max(0,alphas[j]-alphas[i])
                    H=min(C,C+alphas[j]-alphas[i])
                else:
                    L=max(0,alphas[i]+alphas[j]-C)
                    H=min(C,alphas[i]+alphas[j])
                if L==H:
                    print 'L==H';continue
                n=2.0*datamatrix[i,:]*datamatrix[j,:].T-datamatrix[i,:]*datamatrix[i,:].T-datamatrix[j,:]*datamatrix[j,:].T
                if n>= 0:
                    print 'n>=0';continue
                alphas[j] -= labelmat[j]*(Ei-Ej)/n
                alphas[j] = clip(alphas[j],H,L)
                if(abs(alphas[j]-alphaJold)<0.00001):print 'j not moving enough';continue
                alphas[i] += labelmat[j]*labelmat[i]*(alphaJold-alphas[j])
                b1=b-Ei-labelmat[i]*(alphas[i]*alphaIold)*datamatrix[i,:]*datamatrix[i,:].T-labelmat[j]*(alphas[j]-alphaJold)*\
                datamatrix[i,:]*datamatrix[j,:].T
                b2=b-Ej-labelmat[i]*(alphas[i]-alphaIold)*datamatrix[i,:]*datamatrix[j,:].T-labelmat[j]*(alphas[j]-alphaJold)*\
                datamatrix[j,:]*datamatrix[j,:].T
                if(0<alphas[i]) and (C>alphas[i]):b=b1
                elif (i<alphas[j]) and (C>alphas[j]): b=b2
                else: b=(b1+b2)/2.0
                alphapairschanged +=1
                print 'iter: %d i:%d,pairs changed %d' %(iters,i,alphapairschanged)
        if (alphapairschanged == 0):
            iters += 1
        else: iters = 0
        print 'iteration number: %d' %iters
    return b,alphas
    
            
            
            