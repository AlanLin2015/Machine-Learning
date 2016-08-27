# -*- coding: utf-8 -*-
from numpy import *
import matplotlib.pyplot as plt

# _*_ 导入数据文件并将数据文件转化为数据量集和标签量集 _*_
def loaddataset(filename):   
    frtrain=open(filename)   #打开指定文件
    datatrain=[]   #创建空列表用于储存训练数据量集
    labeltrain=[]   #创建空列表用于储存训练标签量集
    for i in frtrain.readlines():   #遍历每一行数据
        linearr=i.strip().split('\t')
        m=len(linearr)   #取列数
        ma=[]   #创建空列表用于储存每一行的数据量
        for j in range(m-1):   #将第1到m-1列的数据作为数据集导入
            ma.append(float(linearr[j]))
        datatrain.append(ma)   #将每一行的数据量添加至总数据量集
        labeltrain.append(float(linearr[-1]))   #将最后一列的数据作为标签集导入
    return datatrain,labeltrain   #返回数据集，标签集

#filename='E:\Users\Alan Lin\Desktop\machinelearningdata\\ex0.txt'
#xarr,yarr=loaddataset(filename)

def standregressions(xarr,yarr):   #输入数据集和标签集，求解W值
    xmat=mat(xarr);ymat=mat(yarr).T   #将数据集转化为矩阵，将标签集转化为矩阵且转置
    xTx=xmat.T*xmat   #计算xmat的转置与xmat的乘机
    if linalg.det(xTx) == 0.0:    #如果xTx不满足求逆的条件，即行列式为0，linalg.det()返回行列式，     则返回不满足条件
        print 'This matrix is singular,cannot do inverse'
        return
    #print shape(xTx.I)
    #print shape(xmat)
    #print shape(ymat)
    #print shape(mat(labeltrain))
    ws=(xTx.I)*(xmat.T*ymat)  #计算系数矩阵ws
    return xmat,ymat,ws
    
#xmat,ymat,ws=standregressions(xarr,yarr)

# _*_ 画图函数
            
def image(xmat,ymat,ws):
    fig=plt.figure()   #建立一个新窗口
    ax=fig.add_subplot(111)   #在新窗口里建立一个一行一列的坐标系并选定这个坐标系
    ax.scatter(xmat[:,1].flatten().A[0],ymat[:,0].flatten().A[0])  #将xmat第二列作为x轴，ymat的第一列作为y轴，描点，不知道flatten的作用
    xcopy=xmat.copy()   #将xmat复制给xcopy
    xcopy.sort(0)   #将xcopy按升序重新排序
    yhat=xcopy*ws   #将xcopy和ws的内积yhat作为y轴
    ax.plot(xcopy[:,1],yhat)   #取数据集的第2列作为x轴，绘线图。
    plt.show()

# _*_ 加权线性回归，返回回归后的y值，重点调用函数
def lwlrtest(testxarr,xarr,yarr,k=1.0):
    m=shape(testxarr)[0]   #取数据量集的行数
    yhat=zeros(m)   #将yhat初始化为m行零向量
    for i in range(m):   #遍历每一行
        yhat[i]=lwlr(testxarr[i],xarr,yarr,k)   #利用局部加权线性回归得到每一个yhat的值
    return yhat


def lwlr(testxarr,xarr,yarr,k=1.0):
    xmat=mat(xarr);ymat=mat(yarr).T   #将xarr转化为矩阵，将yarr转化为矩阵并转置成列
    m=shape(xmat)[0]  #取xmat的行数
    weights=mat(eye(m))   #设置一个对角线为1的矩阵weights
    for j in range(m):   #迭代每一行
        diffmat=testxarr-xmat[j,:]   #diffmat为输入向量与每一行的向量的差,testpoint为每一行的xarr，即xarr[i]
        weights[j,j]=exp(diffmat*diffmat.T/(-2.0*k**2))   #计算用于加权的weights值
    xTx=xmat.T*(weights*xmat)   #xmat用weights加权再带入计算公式
    if linalg.det(xTx)==0.0:
        print 'this matrix is singular,cannot do inverse'
        return
    ws=xTx.I*(xmat.T*(weights*ymat))   #ymat也要用weights加权
    return testxarr*ws
    

#yhat=lwlrtest(xarr,xarr,yarr,k=0.003)

# _*_加权画图部分
    
def imagex(xarr,yarr):
    #xarr,yarr=loaddataset(filename)   
    yhat=lwlrtest(xarr,xarr,yarr,0.003)
    xmat=mat(xarr)
    sortInd=xmat[:,1].argsort(0)
    xsort=xmat[sortInd][:,0,:]
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(xsort[:,1],yhat[sortInd])
    ax.scatter(xmat[:,1].flatten().A[0],mat(yarr).T.flatten().A[0],s=2,c='red')
    plt.show()
    
# _*_计算错误率
def rssError(yarr,yhatarr):
    return ((yarr-yhatarr)**2).sum()
    
    
# _*_岭回归
def ridgeregres(xmat,ymat,lam=0.2):  
    xTx=xmat.T*xmat
    denom=xTx+eye(shape(xmat)[1])*lam #加一个λI（对角矩阵即岭回归矩阵）使得矩阵可逆
    if linalg.det(denom) == 0.0:   #如果lam为0的话仍会产生错误，所以要检查一下
        print "This matrix is singular, cannot do inverse"
        return
    ws=denom.I*(xmat.T*ymat)
    return ws
    
# _*_参数标准化，运用岭回归得到ws
    
def ridgetest(xarr,yarr):
    xmat=mat(xarr);ymat=mat(yarr).T
    ymean=mean(ymat,0)   #计算y均值
    ymat=ymat-ymean
    xmean=mean(xmat,0)   #计算x均值
    xvar=var(xmat,0)   #计算x方差
    xmat=(xmat-xmean)/xvar   #计算标准化
    numtestpts=30
    wmat=zeros((numtestpts,shape(xmat)[1]))
    for i in range(numtestpts):   #便利30个指数级变化的不同的lam来看ws的输出结果 
        ws=ridgeregres(xmat,ymat,exp(i-10))
        wmat[i,:]=ws.T
    return wmat
    
    
    
# _*_ 标准化函数
def regularize(xMat):#regularize by columns
    inMat = xMat.copy()
    inMeans = mean(inMat,0)   #calc mean then subtract it off
    inVar = var(inMat,0)      #calc variance of Xi then divide by it
    inMat = (inMat - inMeans)/inVar
    return inMat

# _*_前向逐步线性回归
def stagewise(xarr,yarr,eps=0.01,numit=100):
 #   fig=plt.figure(figsize=(12,6))
 #   ax1=fig.add_subplot(111)
 #   ax1.set_xlabel('numit')
 #   ax1.set_ylabel('num')
 #   ax1.set_title('num')
    xmat=mat(xarr);ymat=mat(yarr).T   
    ymean=mean(ymat,0)   #计算标签的平均值
    ymat=ymat-ymean
    xmat=regularize(xmat)   #将数据进行均值为0方差1的标准化处理
    m,n=shape(xmat)
    returnmat=zeros((numit,n))   #构建一个100行n列的零矩阵作为ws结果输出
    ws=zeros((n,1));wstest=ws.copy();wsmax=ws.copy()   #构建一个n行1列的ws作为每次的计算的初始值
    for i in range(numit):   #遍历每一组ws的计算
        print ws.T   #打印上次的ws计算结果
        lowesterror=inf;   #初始化lowesterror为最大值
        for j in range(n):   #遍历每一个特征
            for sign in [-1,1]:   #遍历两个方向，分别是正方向和反方向
                wstest=ws.copy()   #第二次遍历重置wstest的值，以免正负抵消
                wstest[j]+=eps*sign  #计算这个特征的wstest值
                ytest=xmat*wstest   #计算ytest值
                rsse=rssError(ymat.A,ytest.A)   #计算误差率
                if rsse<lowesterror:   #若误差率小于当前最小误差率
                    lowesterror=rsse   #置当前误差为最小误差
                    wsmax=wstest   #将这组wstest的值置为最佳wsmax值
        ws=wsmax.copy()   #将wsmax的值置于ws
      #  for i in range(n):
      #      ax1.scatter(ws[i],numit)
        returnmat[i,:]=ws.T   #将这组的ws的值给这一组遍历的矩阵中去
    return returnmat   #返回总矩阵，包含100组ws值



    
    