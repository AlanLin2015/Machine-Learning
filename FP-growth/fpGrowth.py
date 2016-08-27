# -*- coding: utf-8 -*-
class treeNode:
    def __init__(self,namevalue,numoccur,parentnode):
        self.name=namevalue
        self.count=numoccur
        self.nodeLink=None
        self.parent=parentnode
        self.children={}
        
    def inc(self,numoccur):
        self.count += numoccur
    
    def disp(self,ind=1):
        print ' '*ind,self.name,' ',self.count
        for child in self.children.values():
            child.disp(ind+1)

# _*_ 创建新数据集 _*_
def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

# _*_ 将数据集转化为frozenset键的字典
def createinitset():
    dataSet=loadSimpDat()
    dataset = {}
    for trans in dataSet:
        dataset[frozenset(trans)] = 1
    return dataset

'''创建新FP树，输入由字典格式的数据，返回FP树和由样本和频率组成的字典'''
def createtree(dataset,minsup=1):   #输入由字典组成的数据，最小支持频率
    headertable={}   #字典用于储存每一个样本和它们的出现频率
    '''第一次遍历完成所有样本的频率计算'''
    for trans in dataset:   #遍历每一行
        for item in trans:   #遍历每行的每个样本
            headertable[item]=headertable.get(item,0)+dataset[trans]   #get用法：若存在则用字典的值，若不存在则用零替代
    for k in headertable.keys():   #遍历字典的键
        if headertable[k]<minsup:   #若字典的键值小于最小支持频率
            del headertable[k]   #则删除这个键
    freqitemset=set(headertable.keys())   #将符合要求的字典的键转化为集合储存
    if len(freqitemset)==0: return None,None   #若没有一个键值符合要求，则返回无
    for k in headertable.keys():   #遍历每一个键
        headertable[k]=[headertable[k],None]   #将键值改为由键值和指针组成的列表
    retTree=treeNode('Null',1,None)   #新建一个空FP树
    for trans,count in dataset.items():   #遍历数据的每一行（集合）和对应的值，items()将字典转化为列表形式
        '''第二次遍历，完成FP树的建立'''
        localD={}   #字典用于储存每一行符合要求的样本和它们的出现频率
        for item in trans:   #迭代每一行样本（集合）
            if item in freqitemset:   #如果该样本在符合要求的样本集合里
                localD[item]=headertable[item][0]   #则将它的出现频率作为键值储存
        if len(localD)>0:   #如果这行的样本符合要求的数量大于0
            orderitems=[v[0] for v in sorted(localD.items(),key=lambda p:p[1],reverse=True)]   #字典的items()是由键和值组成的列表，所以这里的v[0]是指所有键
            updateTree(orderitems,retTree,headertable,count)   #更新这行符合要求的键在FP树上的表现
    return retTree,headertable   #返回FP树，符合要求的样本和出现频率组成的字典

#更新FP树函数
def updateTree(items,retTree,headertable,count):
    if items[0] in retTree.children:   #如果这个样本在目前所处的节点有出现的，则数量更新
        retTree.children[items[0]].inc(count)
    else:
        retTree.children[items[0]]=treeNode(items[0],count,retTree)   #否则新建一个节点
        if headertable[items[0]][1]==None:   #如果这个样本在字典中的指针还没有指向，则将字典的指针指向这个节点
            headertable[items[0]][1]=retTree.children[items[0]]
        else:
            updateheader(headertable[items[0]][1],retTree.children[items[0]])   #如果这个样本在字典中的指针有指向了，则找到字典中的指针指向的样本，直到那个样本的指针没有指向，把这个节点的指针给那个样本
    if len(items)>1:
        updateTree(items[1::],retTree.children[items[0]],headertable,count)   #items[1::]是指取除了目前的第一个数，这样不断地取下一个数
        
#指针跳跃函数     
def updateheader(nodetotest,targetnode):
    while(nodetotest.nodeLink != None):   #直到最后一个节点指针无指向，停止循环
        nodetotest=nodetotest.nodeLink
    nodetotest.nodeLink=targetnode

#找出该节点的所有的前溯节点
def ascendtree(leafnode,prefixpath):   #找出节点的所有前溯节点，直到第一个节点没有父母节点
    if leafnode.parent != None:
        prefixpath.append(leafnode.name)
        ascendtree(leafnode.parent,prefixpath)
    
#找出每一个样本的所有关联节点的所有前溯节点        
def findprefixpath(basepat,treenode):   #找到每一个样本的所有关联的前溯节点集，basepat是样本名（如x），treenode是节点字典中的指针（它指向第一个节点）
    condpats={}
    while(treenode != None):
        prefixpath=[]
        ascendtree(treenode,prefixpath)
        if len(prefixpath)>1:
            condpats[frozenset(prefixpath[1:])]=treenode.count   #treenode.count是指针指向的节点的计数值
        treenode=treenode.nodeLink
    return condpats
    
'''通过生成条件FP树和频繁项集的对比，证明条件树FP可以取代频繁项集的作用，优化了运算次数'''
def minetree(intree,headertable,minsup,prefix,freqitemlist):
    bigl=[v[0] for v in sorted(headertable.items(),key=lambda p:p[1])]   #将所有的样本放到一个列表里
    for basepat in bigl:   #迭代每一个样本
        newfreqset=prefix.copy()   #重置newfreqset为输入的prefix集合，刚开始时是一个空白集合
        newfreqset.add(basepat)   #将该样本添加到该集合
        freqitemlist.append(newfreqset)   #将该集合添加到频繁项集列表里
        #print 'newfreqset0 is :',newfreqset
        condpattbases=findprefixpath(basepat,headertable[basepat][1])   #找出该样本在FP树中的所有关联的前溯节点集，如果该样本在FP树中处于第一层，则没有父母节点也就没有前溯节点集
        #print 'condpattbases:',condpattbases
        mycondtree,myhead=createtree(condpattbases,minsup)   #根据前溯节点集创建树
        #print 'myhead is :',myhead
        if myhead != None:   #如果condpattbases为空的话，即该节点在目前的FP条件树中已经是最高节点了，则以该basepat为始的频繁项已经全部找好了
            print 'newfreqset is :',newfreqset
            mycondtree.disp()
            minetree(mycondtree,myhead,minsup,newfreqset,freqitemlist)   #否则再次迭代minetree运算，找到以该basespat为始的所有条件树

    
