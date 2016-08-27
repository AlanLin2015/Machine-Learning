# -*- coding: cp936 -*-
'''####################核函数部分##############'''

# -*- coding: utf-8 -*-
from numpy import *
import numpy as np

# _*_ 导入训练集并处理数据生成数据集datamat和标签集labelsmat
def loadDataset(filename):   
    datamat=[];labelsmat=[]
    frtrain=open(filename)
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

# _*_ 开始构建核函数
def kerneltrans(X,A,ktup):
    m,n=shape(X)   #取X的行与列
    k=mat(zeros((m,1)))   #构建一个m行1列的零矩阵
    if ktup[0]=='lin':k=X*A.T   #如果元祖ktup的第一个是字符串'lin'，则k等于X*A.T，结果是一个m行1列的矩阵
    elif ktup[0]=='rbf':   #如果ktup的第一个是字符串'rbf'
        for j in range(m):   #迭代每一行
            deltarow=X[j,:]-A   #X的第j行的数据-X的第i行数据
            k[j]=deltarow*deltarow.T    #获得这行的k值   
        k=exp(k/(-1*ktup[1]**2))
    else:raise NameError('Houston we have a problem that kernel is not recognized')   #假如ktup的第一个值不是以上的两个字符串，则打印错误信息
    return k
    

# _*_ 自定义数据结构体，使用时应当用这样的方式：name=optstruct(datamin,classlabels,C,toler)
class optstruct:
    def __init__(selfs,datamatin,classlabels,C,toler,ktup):
        selfs.X=datamatin
        selfs.labelmat=classlabels
        selfs.C=C
        selfs.tol=toler
        selfs.m=shape(datamatin)[0]
        selfs.alphas=mat(zeros((selfs.m,1)))
        selfs.b=0
        selfs.ecache=mat(zeros((selfs.m,2)))
        selfs.k=mat(zeros((selfs.m,selfs.m)))   #构建一个m行m列的零矩阵
        for i in range(selfs.m):
            selfs.k[:,i]=kerneltrans(selfs.X,selfs.X[i,:],ktup)   #这个k矩阵的每一列用核函数得到的结果代替
# _*_ 计算错误率Ek
def calcek(os,k):
    fxk=float(multiply(os.alphas,os.labelmat).T*os.k[:,k]+os.b)  #仍有问题，这里是f(x)关于alpha的决策分类函数，求解alpha(i)和b(i)
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
    os=optstruct(mat(datamatin),mat(classlabels).transpose(),C,toler,ktup)   #创建数据结构体os保存全局变量
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
    if ((os.labelmat[i]*Ei<-os.tol) and (os.alphas[i]<os.C)) or ((os.labelmat[i]*Ei > os.tol)and(os.alphas[i]>0)):
        j,Ej=selectJ(i,os,Ei)
        alphaIold=os.alphas[i].copy()
        alphaJold=os.alphas[j].copy()
        if (os.labelmat[i] != os.labelmat[j]):
            L=max(0,os.alphas[j]-os.alphas[i])
            H=min(os.C,os.C+os.alphas[j]-os.alphas[i])
        else:
            L=max(0,os.alphas[j]+os.alphas[i]-os.C)
            H=min(os.C,os.alphas[j]+os.alphas[i])
        if L==H:print 'L==H';return 0
        n=2.0*os.k[i,j]-os.k[i,i]-os.k[j,j]
        if n >=0 :print 'n>=0';return 0
        os.alphas[j] -= os.labelmat[j]*(Ei-Ej)/n
        os.alphas[j] = clip(os.alphas[j],H,L)
        updateEk(os,j)
        if (abs(os.alphas[j]-alphaJold)<0.00001):
            print 'j not moving enough';return 0
        os.alphas[i] +=os.labelmat[j]*os.labelmat[i]*(alphaJold-os.alphas[j])
        updateEk(os,i)
        b1=os.b-Ei-os.labelmat[i]*(os.alphas[i]-alphaIold)*os.k[i,i]-os.labelmat[j]*(os.alphas[j]-alphaJold)*os.k[i,j]
        b2=os.b-Ej-os.labelmat[i]*(os.alphas[i]-alphaIold)*os.k[i,j]-os.labelmat[j]*(os.alphas[j]-alphaJold)*os.k[j,j]
        if (0 < os.alphas[i]) and (os.C>os.alphas[i]): os.b=b1
        elif (0<os.alphas[j]) and (os.C>os.alphas[j]): os.b=b2
        else:os.b=(b1+b2)/2.0
        return 1
    else: return 0

# _*_ 计算w函数
def calcWs(alphas,dataarr,classlabels):
    X=mat(dataarr);labelmat=mat(classlabels).transpose()
    m,n=shape(X)
    w=zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelmat[i],X[i,:].T)  #X[i,:]为2行1列，这里定了w的格式
    return w
    


def testrbf(k1=1.3):
    dataarr,labelarr=loadDataset('E:\Users\Alan Lin\Desktop\machinelearningdata\\testSetRBF.txt')   #导入测试集
    b,alphas=smop(dataarr,labelarr,200,0.0001,10000,('rbf',k1))   #将主要参数输入，k1调整k值，得到b，alphas值
    datamat=mat(dataarr);labelmat=mat(labelarr).transpose()
    svInd=nonzero(alphas.A>0)[0]   #得到alphas大于0的位置，即是有用的支持向量的位置
    #print 'alphas.A:',alphas.A
    sVs=datamat[svInd]   #截取这些有用的向量值
    labelSV=labelmat[svInd]   #获得这些向量的对应的标签值
    print 'there are %d support vectors' % shape(sVs)[0]   #计算有用的向量的数量
    m,n=shape(datamat)
    errorcount=0
    for i in range(m):
        kerneleval=kerneltrans(sVs,datamat[i,:],('rbf',k1))   #计算K值
        predict=kerneleval.T*multiply(labelSV,alphas[svInd])+b #这里直接用k的关于训练集的决策函数做分类，理解这里十分重要
        if sign(predict)!=sign(labelarr[i]):errorcount +=1
    print 'the training error rate is: %f'%(float(errorcount)/m)
    dataarr,labelarr=loadDataset('E:\Users\Alan Lin\Desktop\machinelearningdata\\testSetRBF2.txt')
    errorcount=0
    datamat=mat(dataarr);labelmat=mat(labelarr).transpose()
    m,n=shape(datamat)
    for i in range(m):
        kernereval=kerneltrans(sVs,datamat[i,:],('rbf',k1))
        predict=kerneleval.T*multiply(labelSV,alphas[svInd])+b
        if sign(predict)!=sign(labelarr[i]):errorcount +=1
    print 'the test error rate is: %f' % (float(errorcount)/m)
    
    
    
    
# _*_ 将文件中的32*32矩阵转换成1*1024列 _*_
def to_32(filename):
    returnos=zeros((1,1024))   #构造一个1行1024列的零矩阵
    ma=open(filename)   
    for i in range(32):   #从0迭代到31
        lintr=ma.readline()   #readline()为阅读每行函数，这里根据迭代一行一行阅读
        for j in range(32):   #迭代列
            returnos[0,i*32+j]=lintr[j]   #将每一行的数据给returnos矩阵中
    return returnos   #转换完毕，返回returnos
    
# _*_导入图片文件函数
def loadimages(dirname):
    from os import listdir
    trainingtation=listdir(dirname)#用lsitdir函数获取试验向量文件里的每个文件名
    m=len(trainingtation)   #计算所有文件的数量总和
    trainingclocks=zeros((m,1024))   #创造m行1024列的矩阵
    hwlabels=[]   #创造一个空列表用于储存试验向量的标签
    for i in range(m):   #迭代文件数量
        fN=trainingtation[i]   #迭代选取文件名
        fS=fN.split('.')[0]   #通过‘.’切割文件名，并取第一个域
        fS=int(fS.split('_')[0])   #通过‘_’切割剩余文件名，并去第一个域，即表示数字的值
        if fS==9:hwlabels.append(-1)
        else:hwlabels.append(1)   #把数字标签加入标签向量列表中
        trainingclocks[i,:]=to_32('E:\Users\Alan Lin\Desktop\\trainingDigits/%s' % fN)   #把每个文件中的32*32矩阵转换成1*1024的矩阵
    return trainingclocks,hwlabels

# _*_识别手写数字函数
def testdigits(ktup=('rbf',10)):
    dataarr,labelarr=loadimages('E:\Users\Alan Lin\Desktop\\trainingDigits')   #导入测试集
    b,alphas=smop(dataarr,labelarr,200,0.0001,10000,ktup)   #将主要参数输入，k1调整k值，得到b，alphas值
    datamat=mat(dataarr);labelmat=mat(labelarr).transpose()
    svInd=nonzero(alphas.A>0)[0]   #得到alphas大于0的位置，即是有用的支持向量的位置
    #print 'alphas.A:',alphas.A
    sVs=datamat[svInd]   #截取这些有用的向量值
    labelSV=labelmat[svInd]   #获得这些向量的对应的标签值
    print 'there are %d support vectors' % shape(sVs)[0]   #计算有用的向量的数量
    m,n=shape(datamat)
    errorcount=0
    for i in range(m):
        kerneleval=kerneltrans(sVs,datamat[i,:],ktup)   #计算K值
        predict=kerneleval.T*multiply(labelSV,alphas[svInd])+b #这里直接用k的关于训练集的决策函数做分类，理解这里十分重要
        if sign(predict)!=sign(labelarr[i]):errorcount +=1
    print 'the training error rate is: %f'%(float(errorcount)/m)
    dataarr,labelarr=loadimages('E:\Users\Alan Lin\Desktop\\testDigits')
    errorcount=0
    datamat=mat(dataarr);labelmat=mat(labelarr).transpose()
    m,n=shape(datamat)
    for i in range(m):
        kernereval=kerneltrans(sVs,datamat[i,:],ktup)
        predict=kerneleval.T*multiply(labelSV,alphas[svInd])+b
        if sign(predict)!=sign(labelarr[i]):errorcount +=1
    print 'the test error rate is: %f' % (float(errorcount)/m)
    
    
    

    
    
    
    
    
    
    
    
    