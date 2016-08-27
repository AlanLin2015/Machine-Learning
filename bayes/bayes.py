# -*- coding: utf-8 -*-
import numpy as np
from numpy import *


# _*_ 构建包含六个内容为单词的数组的数组，和一个用于分类的标签数组 _*_
def loadDataSet():    
    postingList=[['my','dog','has','has','flea','problems','help','please'],\
                 ['maybe','not','take','him','to','dog','park','stupid'],\
                 ['my','dalmation','is','so','cute','I','love','him'],\
                 ['stop','posting','stupid','worthless','garbage'],\
                 ['mr','licks','ate','my','steak','how','to','stop','him'],\
                 ['quit','buying','worthless','dog','food','stupid']]
    classlabels=[0,1,0,1,0,1]
    return postingList,classlabels


# _*_ 以集合形式去除列表中的重复值并以列表形式返回 _*_
def createVocabList(dataset):
    vocabList=set([])   #初始化set集合
    for line in dataset:   #迭代数据集中的每一行
        vocabList=vocabList | set(line)   #set集合出去重复值，与之前并集
    return list(vocabList)   #将集合已列表形式返回
    
# _*_ 将删去重复值的列表集作为参照，对输入的测试集进行标记，出现一次就加1，最后导出标记向量 _*_
def setofwords2vec(vocabList,inputset):   #vocabList为删去重复值的列表集，inputset为输入测试列表
    returnlist=[0]*len(vocabList)   #创建含有训练列表集个数的零向量列表
    for word in inputset:   #将测试列表里的每个字符串迭代
        if word in vocabList:   #如果这个字符串在训练列表集中出现
            returnlist[vocabList.index(word)] += 1   #则在零列表的第该个字符串出现的位置加1
        else: print 'the word:%s is not in vocabList' % word   #如果没有则显示无
    return returnlist   #返回值为0或1的列表


# _*_ 输入由各个标记向量组成的总矩阵，和每一行的标签向量，输出P1、P0概率和P（Ci） _*_
def trainNBO(trainMatrix,trainCategory):   #trainMatrix为已转化为向量集，traninCategory为原始列表标签
    numtrain=len(trainMatrix)   #取输入列表集的行数
    numaxis=len(trainMatrix[0])   #取列表集的每行的长度
    numcate=sum(trainCategory)   #将标签里的值相加取和作为敏感词的出现总次数
    pobsive=numcate/float(numtrain)   #用标签的和值除以列表的行数
    p1num=ones(numaxis);p0num=ones(numaxis)   #这里的数组：组合各个词为一个向量
    '''初始化为1是为了降低有些词没有出现结果是0的情况的影响'''
    p1ov=2.0;p0ov=2.0
    '''将分母初始化为2'''
    for i in range(numtrain):   #迭代每一行
        if trainCategory[i]==1:   #如果标签栏为1，说明是带有敏感词的行
            p1num += trainMatrix[i]   #累加敏感词的行，这里其实是一个向量的加法，数组中每个词各加各的
            p1ov += sum(trainMatrix[i])   #这里把所有向量都横向加起来，就是全加起来的意思，得到的值是一个标量值而不是向量值
        else:
            p0num += trainMatrix[i]   #如果不是的话，累加非敏感值的行的向量
            p0ov +=sum(trainMatrix[i])   #累加非敏感词的行的向量值，这里的值是一个标量值
            
    p1v=log(p1num/p1ov)   #向量值除以标量值，结果 还是向量值
    p0v=log(p0num/p0ov)   
    '''这里取对数，是为了减小多个向量相乘导致结果非常小的影响，取对数可以直接相加'''
    return p1v,p0v,pobsive   #返回敏感行的的概率值向量，即每个词作为一个敏感词的概率是多少，非敏感行的概率值，敏感行的概率P（Ci）
            
# _*_ 测试函数，输入被测的已经标记好的标记向量，输出是属于P1还是P0的判断 _*_
def classifyNB(vec2Classify,p0v,p1v,pobsive):
    p1=sum(vec2Classify*p1v)+log(pobsive)
    p0=sum(vec2Classify*p0v)+log(1.0-pobsive)
    if p1 > p0:
        return 1
    else:
        return 0
        
def test(testentry):   #总测试函数
    postinglist,classlabels=loadDataSet()  #将分别包含敏感与非敏感词的六个数组和对应的标签数组导出来
    vocablist=createVocabList(postinglist)   #将六个数组里的词都提出来构成一个不重复的集合，再构成列表
    trainmat=[]   #构建一个空数组，作为储存六个数组中每个词的出现在集合中的次数
    for i in postinglist:   #迭代每行数组
        trainmat.append(setofwords2vec(vocablist,i))   #将每行的词都转化为集合中的出现次数
    p1v,p0v,pobsive=trainNBO(trainmat,classlabels)   #返回每个词在集合中出现次数除以集合中总的所有词的出现次数，即P（Wi|Ci）
    thisdoc=array(setofwords2vec(vocablist,testentry))
    print 'the herbs you put in classified as:',classifyNB(thisdoc,p0v,p1v,pobsive)
    
    
# _*_ 解析文本函数，将长度大于2的文字筛选出来 _*_
def textParse(bigstring):
    import re
    listoftokens=re.split(r'\w*',bigstring)   #用正则表达式提取单词
    return [tok.lower() for tok in listoftokens if len(tok)>2]
    
    
# _*_ 导入多个文本文件，并从中抽取测试集，剩余的作为训练集（留存交叉验证），得到贝叶斯函数的错误率 _*_
    
def testpode():
    doclist=[];classlist=[];fulltext=[]   
    for i in range(1,26):
        worldlist1=textParse(open('E:\Users\Alan Lin\Desktop\email\spam\\%d.txt' % i).read())
        doclist.append(worldlist1)   #单词列表添加单词数组
        fulltext.extend(doclist)
        classlist.append(1)   #标签列表添加标签1
        worldlist2=textParse(open('E:\Users\Alan Lin\Desktop\email\ham\\%d.txt' % i).read())
        doclist.append(worldlist2)   #单词列表添加单词数组
        fulltext.extend(doclist)
        classlist.append(0)   #标签列表添加标签0
    vocablist=createVocabList(doclist)   #生成总单词列表
    trainingset=range(50);testset=[]
    for i in range(10):
        randindex=int(random.uniform(0,len(trainingset)))   #随机抽取一个数
        testset.append(trainingset[randindex])   #将这个数添加到testset里，作为测试集的索引
        del(trainingset[randindex])   #删除该数，得到删了测试集索引的训练集索引
    trainmat=[];trainlabels=[]
    for j in trainingset:
        trainmat.append(setofwords2vec(vocablist,doclist[j]))   #将训练集的每个数组都转化为向量
        trainlabels.append(classlist[j])   #对应的标签也提取过来
    p1v,p0v,pobsive = trainNBO(array(trainmat),array(trainlabels))   #用训练集得到P1，P0，和P（Ci）
    error=0  #初始化错误数为0
    for m in testset:
        wordvector=setofwords2vec(vocablist,doclist[m])   #将测试集转化为向量
        if classifyNB(array(wordvector),p0v,p1v,pobsive) != classlist[m]:   #如果测试结果不等于测试集的标签，则错误数加1
            error +=1
    print 'the error rate is:',float(error/len(testset))   #将错误数除以总测试集的数量，得到贝叶斯分类错误率。
            
        
    
        
        
