# -*- coding: utf-8 -*-
from numpy import *

def loaddataset():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

def loaddataset1():
    return [[1,3,4,6],[2,3,5],[1,2,3,5],[2,5,6],[1,2,3,4,6],[1,2,3,4,5,6],[2,3,4,5,6],[1,3,4,5,6]]
    
def createC1(dataset):
    C1=[]
    for line in dataset:
        for j in line:
            if [j] not in C1:   #因为集合不能直接填数字，必须填[]或者‘’形式的
                C1.append([j])   #这里是通过转化为列表，才能使下一步转化为集合
    C1.sort()
    return map(frozenset,C1)   #frozenset是不可变集合

dataset=loaddataset()
D=map(set,dataset) 

'''计算每个类别组合的支持度，并筛选不符合要求的类别组合'''
def scanD(D,Ck,minsupport):
    sscnt={}   #新建一个空字典用于储存该物品出现的次数
    for tid in D:   #遍历每一个购物单，这里D是集合形式
        for can in Ck:   #遍历每一种物品，这里的物品是不变集合形式，can的形式是frozenset([1])
            if can.issubset(tid):   #如果该物品在该购物单中出现,假如是在集合中判断是否存在集合数字，则用issubset()
                if can not in sscnt:sscnt[can]=1   #如果空字典中没有该物品，则增加一个
                else:sscnt[can] += 1   #如果字典中有该物品，则数量加1
    num=float(len(D))   #标志购物单的数量
    L1=[]   #空列表用于储存支持度过滤后的种类
    supportData={}   #空字典用于储存每个类别对应的支持度
    for key in sscnt:   #对字典进行遍历
        support=sscnt[key]/num   #计算该类别的支持度
        if support >= minsupport:   #如果该支持度大于最小支持度
            L1.insert(0,key)   #则用头插法在空列表中加入该物品种类
            supportData[key]=support   #储存对应的支持度信息
    return L1,supportData   #因为key的形式也就是can的形式是frozenset([1])，所以L1是由不变集合组成的列表


'''将符合支持度的类别组合再加一个类别'''
def apriorigen(Lk,k):
    retList=[]   #创建一个空列表用于储存物品搭配的集合
    lenlk=len(Lk)   #取目前物品搭配集合的长度
    for i in range(lenlk):    #i与j的组合遍历每一个可能的两两组合
        for j in range(i+1,lenlk):
            L1=list(Lk[i])[:k-2];L2=list(Lk[j])[:k-2]
            print 'L1',L1;print  'L2',L2
            if L1 == L2:   #假如是单个数字，则L1与L2都是空集
                retList.append(Lk[i]|Lk[j])
    return retList


'''主函数，输入数据和支持度，输出符合支持度的类别组合，和对应的支持度信息'''
def apriori0(datamat,minsupport):
    C1=createC1(datamat)
    #print 'C1:',C1
    D=map(set,datamat)
    #print 'D:',D
    L1,supportData=scanD(D,C1,minsupport)
    #print 'L1:',L1
    #print 'supportData:',supportData
    L=[L1]   #将类别组合转化为列表的方式十分重要
    #print 'L:',L
    k=2   #这里k的设置并无特殊要求，只是与下面的k-2=0要形成对应
    while(len(L[k-2])>0):   #如果第k-2个列表是空的话，则循环结束
        Ck=apriorigen(L[k-2],k)   #根据第n个列表的类别组合，生成第n+1个列表的类别组合
        #print 'Ck:',Ck
        Lk,sup=scanD(D,Ck,minsupport)   #用最小支持度筛选符合要求的第n+1列表里的类别组合
        supportData.update(sup)   #将第n+1的列表的支持度信息更新到全局支持度字典里，字典加字典用update()更新
        L.append(Lk)   #将最新的n+1列表的类别组合添加到总列表里
        k += 1   #类别组合的列表数加1
    return L,supportData

'''关联分析'''

def generateRules(L,supportData,minconf=0.7):  #minconf为可信度
    bigrulelist=[]   #新建列表用于储存关联信息
    for i in range(1,len(L)):   #从第二个开始遍历每一个由频繁项集组成的列表
        for freqset in L[i]:   #从列表里遍历每一个频繁项集
            H1=[frozenset([item]) for item in freqset]   #对频繁项集里的每个项提出来化为frozenset的形式储存在列表中，如[frozenset([1]),frozenset([2])]
            print 'H1:',H1
            if (i > 1):   #因为第二行的频繁项集里的项都只有2个，所以选择大于二行的进行迭代求解，第一行只有一个直接忽略
                H1=clacconf(freqset,H1,supportData,bigrulelist,minconf)   #先算第二层匹配
                rulesfromconseq(freqset,H1,supportData,bigrulelist,minconf)              
            else:
                clacconf(freqset,H1,supportData,bigrulelist,minconf)   #直接求每个频繁项作为后项的可信度，并保留可信度符合要求的项
    return bigrulelist

def clacconf(freqset,H,supportData,bigrulelist,minconf):   #输入频繁项集如frozenset([0,1])，H值作为后项，形式如[frozenset([0]),frozenset([1])]
    returnlist=[]
    for conseq in H:   #对频繁项集里的每个项都假设是后项，计算该可信度
        a=supportData[freqset]/supportData[freqset-conseq]
        if a >= minconf:   #若该可信度符合要求，则输出该后项
            print freqset-conseq,'-->',conseq, 'conf:',a
            bigrulelist.append((freqset-conseq,conseq,a))
            returnlist.append(conseq)
    return returnlist
    
def rulesfromconseq(freqset,H,supportData,bigrulelist,minconf):   #当频繁项集的内容大于1时，如frozenset([0,1,2,3]),其H值为[frozenset([0]),frozenset([1]),...frozenset([3])]
    if len(H) == 0:   #如果上一层没有匹配上则H为空集
        pass
    else:
        m=len(H[0])   #计算H值的第一个值的长度
        if (len(freqset) > (m+1)):   #若freqset的长度大于m+1的长度，则继续迭代
            hmp=apriorigen(H,m+1)   #将单类别加类别，如{0,1,2}转化为{0,1},{1,2}等
            print 'hmp:',hmp
            hmp=clacconf(freqset,hmp,supportData,bigrulelist,minconf)   #计算可信度
            if (len(hmp) > 1):   #如果后项的数量大于1，则还有合并的可能，继续递归
                rulesfromconseq(freqset,hmp,supportData,bigrulelist,minconf)
            
    
            
    
            


    

                
    