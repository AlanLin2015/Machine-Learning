# -*- coding: utf-8 -*-
from numpy import *

def loadDataSet():      #general function to parse tab -delimited floats
    dataMat = []                #assume last column is target value
    fr = open("C:\Users\Alan Lin\Desktop\machinelearningdata\\ex00.txt")
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = map(float,curLine) #map all elements to float()
        dataMat.append(fltLine)
    return mat(dataMat)

def binSplitDataSet(dataSet, feature, value):
    mat0 = dataSet[nonzero(dataSet[:,feature] > value)[0],:]   #使该特征点的值大于阈值的所有行归于左节点
    mat1 = dataSet[nonzero(dataSet[:,feature] <= value)[0],:]   #使该特征点的值小于等于阈值的所以行归于右节点
    return mat0,mat1

def regLeaf(dataSet):   #当不再对数据进行切分时，使用该平均值函数生成节点
    return mean(dataSet[:,-1])

def regErr(dataSet):   #计算总方差函数，均方差乘以个数
    return var(dataSet[:,-1]) * shape(dataSet)[0]

def linearSolve(dataSet):   #helper function used in two places
    m,n = shape(dataSet)
    X = mat(ones((m,n))); Y = mat(ones((m,1)))#create a copy of data with 1 in 0th postion
    X[:,1:n] = dataSet[:,0:n-1]; Y = dataSet[:,-1]#and strip out Y
    xTx = X.T*X
    if linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
        try increasing the second value of ops')
    ws = xTx.I * (X.T * Y)
    return ws,X,Y

def modelLeaf(dataSet):#create linear model and return coeficients
    ws,X,Y = linearSolve(dataSet)
    return ws

def modelErr(dataSet):
    ws,X,Y = linearSolve(dataSet)
    yHat = X * ws
    return sum(power(Y - yHat,2))


#切分函数，输入矩阵数据集，叶节点模型函数（均值），误差估计函数（总方差），总方差差值可接受阈值，每个节点最小样本数
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    tolS = ops[0]; tolN = ops[1]   #tosS为最小总方差减少值，假如总方差差值比这个小，切分就没有意义。tolN为切分最小样本数，小于这个样本数就不切了
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:   #如果该数据的剩余特征值只有一种，则返回这个数据的均值作为叶节点
        return None, leafType(dataSet)
    m,n = shape(dataSet)   #取数据集的行数m和列数n
    S = errType(dataSet)   #计算总数据集的总方差，RSS
    bestS = inf; bestIndex = 0; bestValue = 0   #初始化最小总方差，最好的特征，最好的特征值
    for featIndex in range(n):  #遍历除了最后一列的所以列
        for splitVal in set(dataSet[:,featIndex].T.tolist()[0]):   #遍历每列的所有值
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)   #以每个值为标准切割数据集，直到找到总方差最小的
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN): continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS: 
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S - bestS) < tolS:    #如果总方差的差值仍小于阈值
        return None, leafType(dataSet) #切分没有意义，直接返回均值做为叶节点
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)   #否则返回左节点和右节点
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):  #如果左右节点有一个样本数小于阈值
        return None, leafType(dataSet)   #说明切分也没有意义，直接返回均值作为叶节点
    return bestIndex,bestValue   #返回最好的特征，最好的切分值

#建树函数，调用chooseBestSplit和binSplitDataSet函数
def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1,4)):
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops) #返回当前最佳特征，最佳特征值
    if feat == None: return val   #如果最佳特征为空，则返回数据集的平均值作为叶节点
    retTree = {}   #创建空树
    retTree['spInd'] = feat   #特征
    retTree['spVal'] = val   #特征值
    lSet, rSet = binSplitDataSet(dataSet, feat, val)   #返回左右节点
    retTree['left'] = createTree(lSet, leafType, errType, ops)   #对左节点递归
    retTree['right'] = createTree(rSet, leafType, errType, ops)   #对右节点递归
    return retTree  

def isTree(obj):
    return (type(obj).__name__=='dict')

def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0
    
def prune(tree, testData):
    if shape(testData)[0] == 0: return getMean(tree) #if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):#if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] =  prune(tree['right'], rSet)
    #if they are now both leafs, see if we can merge them
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        errorNoMerge = sum(power(lSet[:,-1] - tree['left'],2)) +\
            sum(power(rSet[:,-1] - tree['right'],2))
        treeMean = (tree['left']+tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1] - treeMean,2))
        if errorMerge < errorNoMerge: 
            print "merging"
            return treeMean
        else: return tree
    else: return tree
    
def regTreeEval(model, inDat):
    return float(model)

def modelTreeEval(model, inDat):
    n = shape(inDat)[1]
    X = mat(ones((1,n+1)))
    X[:,1:n+1]=inDat
    return float(X*model)

def treeForeCast(tree, inData, modelEval=regTreeEval):
    if not isTree(tree): return modelEval(tree, inData)
    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']): return treeForeCast(tree['left'], inData, modelEval)
        else: return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']): return treeForeCast(tree['right'], inData, modelEval)
        else: return modelEval(tree['right'], inData)
        
def createForeCast(tree, testData, modelEval=regTreeEval):
    m=len(testData)
    yHat = mat(zeros((m,1)))
    for i in range(m):
        yHat[i,0] = treeForeCast(tree, mat(testData[i]), modelEval)
    return yHat