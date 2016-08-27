# -*- coding: utf-8 -*-
from numpy import *
import urllib
import json
from time import sleep   #sleep为延时函数
import matplotlib
import matplotlib.pyplot as plt

def loaddataset():
    datamat=[]
    fr=open('E:\Users\Alan Lin\Desktop\\machinelearningdata\\testSet2.txt')
    for line in fr.readlines():
        linearr=line.strip().split('\t')
        floatlinearr=map(float,linearr)   #map函数是一种自定义函数的应用，可以应用到列表的每一个元素中
        datamat.append(floatlinearr)
    return datamat

def diseclud(A,B):
    return sqrt(sum(power(A-B,2)))   #计算欧式距离，是矩阵的运算

'''构建k行n列的质心'''
def randcent(dataset,k):
    dataset=mat(dataset)   
    m,n=shape(dataset)
    centroids=mat(zeros((k,n)))   #构建质心矩阵，它是有k行n列组成的，k行表示k个质心
    for i in range(n):   #迭代每一列
        mini=float(min(dataset[:,i]))   #计算整列的最小值
        maxi=float(max(dataset[:,i]))   #计算整列的最大值
        centroids[:,i]=(mini+(maxi-mini)*random.rand(k,1))   #每一列随机生成k个处于最大值与最小值之间的随机值
    return centroids   
    
    
'''k均值主程序，返回分类的质心和分类后的每行的类别和距质心的距离'''
def kmeans(dataset,k,dismeas=diseclud,createcent=randcent):
    datamat=mat(dataset)
    m,n=shape(datamat)
    clusterassment=mat(zeros((m,2)))   #构建一个m行2列的矩阵用于储存每个样本的类别和距质心距离的全局信息
    centroids=createcent(dataset,k)   #生成k个质心
    clusterchanges=True   #循环继续标志
    while clusterchanges:   #若标志为True
        clusterchanges=False   #置标志为False
        for i in range(m):   #迭代每一个样本
            clusterdistance=inf;index=-1    #初始化最小质心的距离为无穷大，类别为-1
            for j in range(k):   #迭代每一类
                distance=diseclud(datamat[i,:],centroids[j,:])   #计算每个样本和每一类的质心样本的距离
                if distance < clusterdistance:   #若距离小于最小质心距离
                    index = j;clusterdistance = distance   #将类别设置为这个质心的类别，将最小距离置为现在的距离
            if clusterassment[i,0] != index:clusterchanges=True   #循环完每一类后，若全局信息中的类别与现在计算的类别不一样，则需要再循环计算一遍
            clusterassment[i,:]=index,clusterdistance**2   #将这次的类别信息，距质心距离信息给全局信息
        print centroids   #打印现在的质心
        for cent in range(k):   #迭代每一类
            ptsinlust=datamat[nonzero(clusterassment[:,0].A==cent)[0]]   #提取训练集中属于这一类的所有样本
            centroids[cent,:]=mean(ptsinlust,axis=0)   #计算样本均值，重新计算质心
    return centroids,clusterassment


'''二分k均值算法，避免局部最小值，比较优化的k均值算法''' #重点在于列表和矩阵的转换
def biKmeans(dataset,k,distMeas=diseclud):
    datamat=mat(dataset)    #将输入的数据转化为矩阵
    m,n=shape(datamat)
    clusterassment=mat(zeros((m,2)))   #创建一个m行2列的零矩阵C用于储存全局标签和平方误差信息
    centroid0=mean(datamat,axis=0).tolist()[0]   #计算数据集每列的平均值作为质心，由于下面需要用到列表，所以需要将矩阵转化为列表，此时是二维的列表，需要取第一个列表达到降维的目的
    centList=[centroid0]   #将降维后的列表重新组成二维列表
    for j in range(m):   #迭代每一行，于C的第二列中保存所有数据与平均值的平方误差信息
        clusterassment[j,1]=distMeas(mat(centroid0),mat(datamat[j,:]))**2   #centroid0降维的作用在这里体现，它需要与一行两列的datamat[j,:]完成计算，它们必须是同维
    while(len(centList)<k):    #直到质心的数量达到k，否则循环不结束
        lowestSSE=inf   #将最小平方误差设为无穷大
        for i in range(len(centList)):   #迭代现在的每一个类别
            ptsincurrcluster=datamat[nonzero(clusterassment[:,0].A==i)[0],:]   #将数据集中属于i类的数据提出来
            centroids,splitclustass=kmeans(ptsincurrcluster,2,distMeas)   #将这个提出来的i类数据重新进行kmeans二分类
            ssesplit=sum(splitclustass[:,1])   #计算i类的最小平方误差（已经被分成两类）
            ssenotsplit=sum(clusterassment[nonzero(clusterassment[:,0].A != i)[0],1])   
           #计算除了i类的其他的最小平方误差,A在这里的作用是将bool值转化为array的形式
            print 'ssesplit,ssenotsplit:',ssesplit,ssenotsplit   
            if (ssesplit+ssenotsplit)<lowestSSE:   #如果全局平方误差小于历史最小平方误差
                bestcentsplit=i   #标志现在这个被分类的i类
                bestnewcents=centroids   #标志分类后的质心（是i类被分类后的两个质心）
                bestclustass=splitclustass.copy()   #标志分类后的标签和平方误差信息
                lowestSSE=ssesplit+ssenotsplit   #重新标志历史最小平方误差
        bestclustass[nonzero(bestclustass[:,0].A == 1)[0],0]=len(centList)   #新类别中的1应为此时的最高类
        bestclustass[nonzero(bestclustass[:,0].A == 0)[0],0]=bestcentsplit   #新类别中的0应为原来的类别
        print 'bestcentsplit',bestcentsplit   #打印被分类的类别i
        print 'len of bestclustass',len(bestclustass)   #打印分类后的标签和平方误差信息的长度
        print 'bestnewcents[0,:]',bestnewcents[0,:]
        centList[bestcentsplit]=bestnewcents[0,:].tolist()[0]   #因为这个类在centList已存在，不用添加，而用更新
        centList.append(bestnewcents[1,:].tolist()[0])
        clusterassment[nonzero(clusterassment[:,0].A == bestcentsplit)[0],:] = bestclustass 
        #因为bestclustass只能储存被计算的那个簇的全局信息，而clusterassment储存所有的全局信息，需要时用bestclustass更新即可
    return mat(centList),clusterassment          


'''输入地址信息，返回包含信息的字典'''
def geograb(staddress,city):   #获得该地理位置对应的位置信息，以json形式返回
    apistem='http://where.yahooapis.com/geocode?'   #目标网址
    params={}   #创建即将被转化为网址具体信息的字典
    params['flags']='J'   #以json格式返回结果
    params['appid']='FZ8GRMTWBC4PYDX3G4NR'   #用于访问yahoo的个人API KEY
    params['location']='%s %s' % (staddress,city)
    url_params = urllib.urlencode(params)   #将字典信息转化为url的字符串格式
    yahooapi=apistem+url_params   #汇总总网址，得到一个访问的url
    print yahooapi
    c=urllib.urlopen(yahooapi)   #打开url读取返回值
    return json.loads(c.read())   #返回值是json格式的，所以要用json模块解码成字典

'''将每个城市的位置信息返回到一个文件中'''
def massplacefind():
    fw=open('E:\Users\Alan Lin\Desktop\machinelearningdata\\places.txt','w')   #打开文件并写入
    fr=open('E:\Users\Alan Lin\Desktop\machinelearningdata\\portlandClubs.txt')
    for line in fr.readlines():
        line=line.strip()
        linearr=line.split('\t')
        redict=geograb(linearr[1],linearr[2])
        if redict['ResultSet']['Error'] == 0:
            lat=float(redict['ResultSet']['Results'][0]['latitude'])
            lng=float(redict['ResultSet']['Results'][0]['longitude'])
            print '%s\t%f\t%f'   % (line,lat,lng)
            fw.write('%s\t%f\t%f\n' % (line,lat,lng))
        else:print 'error fetching'
        sleep(1)
    fw.close()
                
def distslc(A,B):   #球面距离计算公式
    a=sin(A[0,1]*pi/180)*sin(B[0,1]*pi/180)   #A的纬度*B的纬度
    b=cos(A[0,1]*pi/180)*cos(B[0,1]*pi/180)*cos(pi*(B[0,0]-A[0,0])/180)
    return arccos(a+b)*6371.0
    
    
    
    
def clusterclubs(numiter=5):
    datList=[]
    for line in open('E:\Users\Alan Lin\Desktop\machinelearningdata\\places.txt').readlines():
        linearr=line.split('\t')
        datList.append([float(linearr[4]),float(linearr[3])])
    datamat=mat(datList)
    mycentroids,clusterassing=biKmeans(datamat,numiter,distMeas=distslc)
    fig = plt.figure()
    rect=[0.1,0.1,0.8,0.8]
    scattermarkers=['s','o','^','8','p','d','v','h','>','<']
    axprops=dict(xticks=[],yticks=[])
    ax0=fig.add_axes(rect,label='ax1',frameon=False)
    imgP=plt.imread('E:\Users\Alan Lin\Desktop\machinelearningdata\\Portland.png')
    ax0.imshow(imgP)
    ax1=fig.add_axes(rect,label='ax1',frameon=False)
    for i in range(numiter):
        ptsincurrcluster=datamat[nonzero(clusterassing[:,0].A==i)[0],:]
        markerstyle=scattermarkers[i%len(scattermarkers)]
        ax1.scatter(ptsincurrcluster[:,0].flatten().A[0],ptsincurrcluster[:,1].flatten().A[0],marker=markerstyle,s=90)
    ax1.scatter(mycentroids[:,0].flatten().A[0],mycentroids[:,1].flatten().A[0],marker='+',s=300)
    plt.show()
