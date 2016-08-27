# -*- coding: utf-8 -*-
from math import log
import operator

'''根据导入的数据矩阵计算香农熵'''
def clacshannonEnt(dataset):   #定义香农熵函数
    num=len(dataset)   #取得输入数据的长度，即行数
    dic = {}   #定义一个空字典
    for i in dataset:   #迭代数据矩阵中的行
        labels=i[-1]   #选取每一行的最后一列的数据，看成数据的标签
        if labels not in dic.keys():   #如果字典里的键里没有这个标签
            dic[labels]=0   #添加这个标签为字典里的键，对应的值为0
        dic[labels] += 1   #若字典中已有这个标签，则对应的值加1
    shannon=0.0   #初始化香农熵的值为0
    for key in dic:   #迭代字典的键
        prob=float(dic[key])/num   #将每个键对应的值除以总行数，即得到这个键的概率
        shannon -= prob * log(prob,2)   #计算香农熵
    return shannon   #返回香农熵

    
# _*_ 创建一个数据矩阵 _*_

def creatDataset():
    dataset = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]   #定义一个数据列表
    labels=['no surfacing','flippers']   #数据的标签
    return dataset,labels

# _*_ 按照给定特征划分数据集 _*_
def splitdataset(dataset,axis,value):
    retDataset=[]   #初始化一个空列表
    for valueset in dataset:   #迭代数据的每一行
        if valueset[axis] == value:   #如果每一行的第axis列的值等于value
            reg=valueset[:axis]   #取axis之前的数据
            reg.extend(valueset[axis+1:])   #把axis之前的数据与axis之后的数据相加
            retDataset.append(reg)  #把（值为value的axis列去掉后）的每一行加到retDataset列表里
    return retDataset   #返回retDataset列表
    
# _*_ 选择当前矩阵中香农熵最高的特征划分方法，重点在香农熵加权累加，当列中的值不同时，需要把不同的值分别计算香农熵，在根据不同的值所占的行数加权 _*_   
def choosebestfeature(dataset):   
    numfeature=len(dataset[0])-1   #第一行的列数
    baseEntropy=clacshannonEnt(dataset)   #计算初始化香农熵
    bestInfoGain=0.0;bestFeature=-1   #初始化香农熵对比值，初始化最好的方法的值
    for i in range(numfeature):   #迭代每一列
        baselist=[example[i] for example in dataset]   #把每一行的第i列的所有数集合到一个列表中
        baseset=set(baselist)   #把列表转换成集合，目的是把重复的数值去掉，集合里只能有不重复的值
        newEntropy=0.0   #初始化加权后的香农熵
        for m in baseset:   #迭代集合里的数
            regg=splitdataset(dataset,i,m)   #按每个数的特征分别划分数据集，即去掉值为m的第i列，所得到的行，这里的regg是一个由行组成的列表
            probs=len(regg)/float(len(dataset))   #取每个数据集的行数占总行数的多少，这里有一个加权
            newEntropy += probs * clacshannonEnt(regg)  #每个m值的加权香农熵相加得到一个总的加权香农熵   H（D|A）
        infoGain = baseEntropy - newEntropy   #计算信息增益，当newEntropy信息熵越小时，信息增益越大，则把这列作为决策点时是合适的
        if(infoGain>bestInfoGain):   #假如对比结果大于零
            bestInfoGain=infoGain   #把这次相减的结果给判断值
            bestFeature=i   #将这个列作为最好的特征划分数据集
    return bestFeature

# _*_ 创建字典导入个值列表，用多数表决法选择出最好的类别 _*_
def majorityCnt(classlist):
    majority={}   #初始化用于储存键和出现次数的字典
    for i in classlist:
        if i not in majority.keys():majority[i]=0   
        majority[i] += 1
    sortedmajority=sorted(majority.iteritems,key=operator.itergetter(1),reverse=True)   #将字典的第二个域用于关键字，逆序排序
    return sortedmajority[0][0]
    
'''决策树字典中心创建程序，主要用到了递归方法 _*_'''
def createtree(dataset,labels):
    classlist=[example[-1] for example in dataset]   #把数据矩阵中的最后一列数据即标签栏拿来初始化标签列表
    if classlist.count(classlist[0])==len(classlist):   #如果列表中第一个值的数量等于整个列表的数量
        return classlist[0]   #那么这个列表中的标签都是一样的，返回这个标签
    if len(dataset[0])==1:   #如果这个数据矩阵的列数等于1，说明没有用于分类的数据了
        return majorityCnt(classlist)   #那么用多数表决的方法来返回该分类
    bestfeature=choosebestfeature(dataset)   #返回香农熵最大的列
    bestlabels=labels[bestfeature]   #返回该列的标签
    mytrees={bestlabels:{}}   #初始化当前最优的标签的字典
    del(labels[bestfeature])   #把当前最优的标签从labels里删除，便于递归时labels已经是删了前一个递归最优值的labels
    bestdata=[example[bestfeature] for example in dataset]   #从每行里获取最优列的值并构成列表
    bestvalue=set(bestdata)   #为保证值得唯一性，将列表转换成集合
    for value in bestvalue:   #迭代所有集合里的值
        sublabels=labels[:]   #标识下一个递归时labels的值
        mytrees[bestlabels][value]=createtree(splitdataset(dataset,bestfeature,value),sublabels)   #构架决策树字典，递归下一个删了当前最优列的数据矩阵
    return mytrees

'''决策树调用应用程序，检测被测量所被分类的类别'''
def classify(inputtree,featlabels,testvec):
    firstStr=inputtree.keys()[0]   #取决策树字典的首标签
    seconddict=inputtree[firstStr]   #取决策树第二层字典
    featindex=featlabels.index(firstStr)   #用index函数得到首标签字符串在标签列表中的排位featindex
    for key in seconddict.keys():   #迭代第二层字典的所有键
        if testvec[featindex]==key:   #若被测数组中的第featindex个值等于第二层字典的键
            if type(seconddict[key]).__name__=='dict':   #检测该字典的键对应的值是否仍为字典
                classlabels=classify(seconddict[key],featlabels,testvec)   #若仍为字典，则递归计算下一层的值
            else:classlabels=seconddict[key]   #若不是，说明到了叶节点，把这个节点的值给最后的结果参数
    return classlabels

#_*_储存决策树到文件函数_*_
def storeTree(inputTree,filename):
    import pickle   #声明pickle
    fw=open(filename,'w')   #以写入的形式新建文件名为filename的文件
    pickle.dump(inputTree,fw)   #将inputTree写入文件
    fw.close()   #关闭文件
    
#_*_调用文件里的决策树函数_*_
def grabTree(filename):
    import pickle
    fr=open(filename)   #打开文件名为filename的文件
    return pickle.load(fr)   #导入文件中的决策树
            
        