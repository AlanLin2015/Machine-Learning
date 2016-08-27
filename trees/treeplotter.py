# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

decisionNode = dict(boxstyle='sawtooth',fc='0.8')   #决策节点框，锯齿形，0.8宽
leafNode=dict(boxstyle='round4',fc='0.8')   #叶节点框，圆弧形，0.8宽
arrow_args=dict(arrowstyle='<-')   #箭头类型
# _*_形成箭头函数_*_
def plotNode(nodeTxt,centerPt,parentPt,nodeType):  
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,\
                            textcoords='axes fraction',va='center',ha='center',bbox=nodeType,\
                            arrowprops=arrow_args)
                          #  '''nodeTxt：子节点的值，xy：父节点的坐标，xytext：子节点的坐标，bbox：子节点的框类型\
                          #  arrowprops：箭头类型'''
'''                            
def createPlot1():   #创建绘图函数
    fig=plt.figure(1,facecolor='white')
    fig.clf()  #清空当前图像
    createPlot.ax1=plt.subplot(111,frameon=False)   #创建一个图像
    plotNode(U'决策节点',(0.5,0.1),(0.1,0.5),decisionNode)   #画箭头节点图
    plotNode(U'叶节点',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()
'''


#_*_计算叶子节点数量函数_*_

def getNumleafs(myTree):   
    numleafs = 0   #初始化叶子节点为零
    firstStr=myTree.keys()[0]   #取myTree的首个标签的第一个值作为下一个字典的导引值
    seconddict=myTree[firstStr]   #构建下个字典
    for key in seconddict.keys():   #迭代这个字典的键
        if type(seconddict[key]).__name__=='dict':   #若每个键对应的字典对的形式的都是字典
            numleafs +=getNumleafs(seconddict[key])   #递归字典，重复以上的运算，直到所有的叶节点对应的不是字典
        else: numleafs +=1   #否则的话，叶节点数量加1
    return numleafs
    
#_*_计算决策树深度函数_*_
def getTreeDepth(myTree):   
    maxDepth=0.0   #初始化最大深度值为零
    firstStr=myTree.keys()[0]
    seconddict=myTree[firstStr]
    for key in seconddict.keys():   #迭代每个键
        if type(seconddict[key]).__name__=='dict':   #若每个键对应的字典对的形式都是字典
            newdepth=1+getTreeDepth(seconddict[key])   #则深度加1且继续递归计算深度值，直到所有的叶节点不是字典对
        else: newdepth=1   #否则深度值为1
        if newdepth>maxDepth:maxDepth=newdepth   #对比深度值与最大深度值的大小，若大于最大深度值，则取代最大深度值
    return maxDepth


def retrieveTree(i):   #构建简单决策树函数
    listOfTrees=[{'no surfacing':{0:'no',1:{'flippers':{0:'no',1:'yes'}}}},{'no surfacing':\
                 {0:'no',1:{'flippers':{0:{'head':{0:'no',1:'yes'}},1:'no'}}}}]
    return listOfTrees[i]
    
#_*_#取父节点和子节点中间点且把key命名其名称_*_
def plotMidText(cntrPt,parentPt,txtString):   
    Xmid=(parentPt[0]-cntrPt[0])/2.0+cntrPt[0]   #取x轴上的父节点和子节点的中间点
    Ymid=(parentPt[1]-cntrPt[1])/2.0+cntrPt[1]   #取y轴上的父节点和子节点的中间点
    createPlot.ax1.text(Xmid,Ymid,txtString)   #命名中间点
    
    
'''构建绘图函数，调用的重点程序'''
def createPlot(inTree):   
    fig=plt.figure(1,facecolor='white')   #开辟一个新图像，颜色为白色
    fig.clf()   #清空当前图像
    axprops=dict(xticks=[],yticks=[])   #x,y轴的刻度都默认无
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops)   #创建一个空白图像，frameon，**axprops含义不明
    plotTree.totalW=float(getNumleafs(inTree))   #设置全局宽度变量，其值为决策树的叶节点数
    plotTree.totalD=float(getTreeDepth(inTree))   #设置全局深度变量，其值为决策树的深度值
    plotTree.xoff= -0.5/plotTree.totalW;plotTree.yoff=1.0;   
    '''初始化追踪变量plotTree.xoff，其初始值为全局宽度变量的-1/2，意为第一个格子的前半个格子点为初始点，\
       其作用就是在下次绘图时自动追踪上次绘图的坐标并作为其值，plotTree.yoff的作用同上，初始值为格子高度1。
       注：格子的参数为高为1，宽为1。'''
    plotTree(inTree, (0.5,1.0), '')   #引用真正绘图函数，inTree为决策树，(0.5,1.0)为父节点坐标，中间点命名名称
    plt.show()

'''真正的绘图函数,理解整段程序的重点'''
def plotTree(myTree,parentPt,nodeTxt):   
    numleafs=getNumleafs(myTree)   #叶节点数为导入递归后（首次无递归）的决策树的叶节点数，影响决策节点的x轴
    Depth=getTreeDepth(myTree)   #深度值也是导入递归后（首次无递归）的决策树深度值
    firstStr=myTree.keys()[0]   #决策树字典的第一个标签
    cntrPt=(plotTree.xoff+(1.0+float(numleafs))/2.0/plotTree.totalW,plotTree.yoff)   #计算子节点的x轴坐标，这里计算结果为1/2
    plotMidText(cntrPt,parentPt,nodeTxt)   #中间点命名
    plotNode(firstStr,cntrPt,parentPt,decisionNode)  
    '''画箭头，这里第一个箭头的子节点与父节点的坐标是一样的，而且中间点的命名为空，\
    所以这里可看作是画了一个父节点框。'''
    seconddict=myTree[firstStr]   #取第二层字典
    plotTree.yoff=plotTree.yoff-1/plotTree.totalD   #y轴全局变量减去一个格子（一个格子长度为1/plotTree.totalD）
    for key in seconddict.keys():  #迭代第二层字典的所有标签
        if type(seconddict[key]).__name__=='dict':   #若第二层字典的标签对应的值仍为字典
            plotTree(seconddict[key],cntrPt,str(key))   
            '''则把第二层字典的每个标签对应的字典作为输入决策树，重新递归，这里是整个函数的重点，\
            plotTree.xoff只有在递归时重新绘制图时才会变成上次绘图的坐标，因而可以使cntrPt的值产生变化，\
            使其移动到应该的位置上，str(key)的值为0或1作为中间点的命名名称。'''
        else:
            plotTree.xoff=plotTree.xoff+1.0/plotTree.totalW   #否则plotTree.xoff还是初始值，它要加一个格子
            plotNode(seconddict[key],(plotTree.xoff,plotTree.yoff),cntrPt,leafNode)   #画叶子节点的箭头图
            plotMidText((plotTree.xoff,plotTree.yoff),cntrPt,str(key))   #命名中间点
    plotTree.yoff=plotTree.yoff+1/plotTree.totalD   #plot.yoff恢复其值
    
    
