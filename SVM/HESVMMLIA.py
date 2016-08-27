# -*- coding: cp936 -*-
'''####################�˺�������##############'''

# -*- coding: utf-8 -*-
from numpy import *
import numpy as np

# _*_ ����ѵ���������������������ݼ�datamat�ͱ�ǩ��labelsmat
def loadDataset(filename):   
    datamat=[];labelsmat=[]
    frtrain=open(filename)
    for i in frtrain.readlines():
        linearr=i.strip().split('\t')
        datamat.append([float(linearr[0]),float(linearr[1])])
        labelsmat.append(float(linearr[2]))
    return datamat,labelsmat
    
# _*_ ���ѡ������ͨ������i��m�������0��m�������ȡ����j
def selectJrand(i,m):
    j=i
    while(j==i):
        j=int(random.uniform(0,m))
    return j

# _*_ �Զ��庯��clip
def clip(aj,H,L):
    if aj>H:
        aj=H
    if aj<L:
        aj=L
    return aj

# _*_ ��ʼ�����˺���
def kerneltrans(X,A,ktup):
    m,n=shape(X)   #ȡX��������
    k=mat(zeros((m,1)))   #����һ��m��1�е������
    if ktup[0]=='lin':k=X*A.T   #���Ԫ��ktup�ĵ�һ�����ַ���'lin'����k����X*A.T�������һ��m��1�еľ���
    elif ktup[0]=='rbf':   #���ktup�ĵ�һ�����ַ���'rbf'
        for j in range(m):   #����ÿһ��
            deltarow=X[j,:]-A   #X�ĵ�j�е�����-X�ĵ�i������
            k[j]=deltarow*deltarow.T    #������е�kֵ   
        k=exp(k/(-1*ktup[1]**2))
    else:raise NameError('Houston we have a problem that kernel is not recognized')   #����ktup�ĵ�һ��ֵ�������ϵ������ַ��������ӡ������Ϣ
    return k
    

# _*_ �Զ������ݽṹ�壬ʹ��ʱӦ���������ķ�ʽ��name=optstruct(datamin,classlabels,C,toler)
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
        selfs.k=mat(zeros((selfs.m,selfs.m)))   #����һ��m��m�е������
        for i in range(selfs.m):
            selfs.k[:,i]=kerneltrans(selfs.X,selfs.X[i,:],ktup)   #���k�����ÿһ���ú˺����õ��Ľ������
# _*_ ���������Ek
def calcek(os,k):
    fxk=float(multiply(os.alphas,os.labelmat).T*os.k[:,k]+os.b)  #�������⣬������f(x)����alpha�ľ��߷��ຯ�������alpha(i)��b(i)
    Ek=fxk-float(os.labelmat[k])   #����Ek,������fxk��ȥk�ı�ǩֵ
    return Ek
    
# _*_ ����󲽳�����ȡj��ֵ(���㵱ʱ��os.ecache��һ�е�������ֵ��λ�ö�Ӧ�Ĵ�����Ek��ֵ����Ei�Ա��ҳ���������k)
def selectJ(i,os,Ei):
    maxk=-1;maxdeltae=0;Ej=0
    os.ecache[i]=[i,Ei]
    validecachelist=nonzero(os.ecache[:,0].A)[0]  #ȡecache�ĵ�һ�е���������ֵ������Ӧ��λ�ã���Ϊecache�����Ƕ�ά���飬����nonzero���صĵ�һ��������ecache�ĵ�һ�еģ�����ȥ��һ��
    if (len(validecachelist)) > 1:
        for k in validecachelist:
            if k==i: continue   #��ȡ��k����i��ʱ�����µ���ȡ��һ��k�������ǳ���k==i������ֵ��������һ��
            Ek=calcek(os,k)
            deltaE=abs(Ei-Ek)
            if(deltaE>maxdeltae):
                maxk=k;maxdeltae=deltaE;Ej=Ek
        return maxk,Ej
    else:
        j=selectJrand(i,os.m)
        Ej=calcek(os,j)
    return j,Ej
    
# _*_ ��k�ʹ������ϴ���os.ecacheȫ�ֱ�������
    
def updateEk(os,k):
    Ek=calcek(os,k)
    os.ecache[k]=[1,Ek]
    
    
# _*_ ������ _*_
def smop(datamatin,classlabels,C,toler,maxiter,ktup=('lin',0)):
    os=optstruct(mat(datamatin),mat(classlabels).transpose(),C,toler,ktup)   #�������ݽṹ��os����ȫ�ֱ���
    iters=0   #��ʼ����������
    entireset=True;alphapairschanged=0   #��ʼ����������entiresetΪ True��alphapairschangedΪ0
    while (iters<maxiter) and ((alphapairschanged>0) or (entireset)):   #�˳�ѭ�������������κ�һ���������㣺1.���������ﵽ��ָ��������2.alphapairschanged=0��entiresetΪFalseͬʱ���㣻
        alphapairschanged=0
        if entireset:   #����entiresetΪTrue�Ļ�����ʼ�������������������Լ������������Ƿ�����KTT����
            for i in range(os.m): #��ѭ��
                alphapairschanged += innerL(i,os)  #����ǩ*�����ʺ�alpha[i]ͬʱ��������ʱ���м��㲢����1�����򷵻�0
            print 'fullset,iter: %d i:%d,pairs changed %d' %(iters,i,alphapairschanged)  #������ϣ���ӡwhile����������i��ֵ�����������Ĵ���alphapairschanged
            iters+=1   #while����������1
        else:
            #print '(os.alphas.A>0):',(os.alphas.A>0)   #����һ������ֵ����������0��ΪTrue��С�����ΪFalse
            #print '(os.alphas.A<C):',(os.alphas.A<C)
            #print '((os.alphas.A>0)*(os.alphas.A<C)):',((os.alphas.A>0)*(os.alphas.A<C))  #���������˲����������Զ���ԣ�������ʵ��һ��ȡ��Ϊ�ԵĽ������ѧ�������ǽ���
            nonBoundIs=nonzero((os.alphas.A>0)*(os.alphas.A<C))[0]   #���طǱ߽�������iֵ�����Ƿ���alpha��ֵ��[0,C]֮���alpha��λ�ã�����alpha��Ӧ��iֵ
            #print 'nonBoundIs:', nonBoundIs
            for i in nonBoundIs:  #��ѭ��
                alphapairschanged += innerL(i,os)   #��ʼ��ѭ��
                print 'non-bound,iter:%d i:%d,pairs changed %d' %(iters,C,alphapairschanged)  #ÿ����һ��iֵ���ʹ�ӡ����ʱ��while����������i��ֵ�����������Ĵ���alphapairschanged
            iters+=1   #while����������1
        if entireset:entireset=False   #���entiresetΪTrue�Ļ���������ΪFalse�Ա����Ǳ߽�����������ѭ��
        elif (alphapairschanged==0):entireset=True   #���������⣬��֪�����ڱ����������϶�δ��alpha�Խ����޸�ʱ������˳�ѭ����
        print 'iteration number: %d' %iters   #��ӡwhile��������
    return os.b,os.alphas
   
# _*_��ѭ������
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

# _*_ ����w����
def calcWs(alphas,dataarr,classlabels):
    X=mat(dataarr);labelmat=mat(classlabels).transpose()
    m,n=shape(X)
    w=zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelmat[i],X[i,:].T)  #X[i,:]Ϊ2��1�У����ﶨ��w�ĸ�ʽ
    return w
    


def testrbf(k1=1.3):
    dataarr,labelarr=loadDataset('E:\Users\Alan Lin\Desktop\machinelearningdata\\testSetRBF.txt')   #������Լ�
    b,alphas=smop(dataarr,labelarr,200,0.0001,10000,('rbf',k1))   #����Ҫ�������룬k1����kֵ���õ�b��alphasֵ
    datamat=mat(dataarr);labelmat=mat(labelarr).transpose()
    svInd=nonzero(alphas.A>0)[0]   #�õ�alphas����0��λ�ã��������õ�֧��������λ��
    #print 'alphas.A:',alphas.A
    sVs=datamat[svInd]   #��ȡ��Щ���õ�����ֵ
    labelSV=labelmat[svInd]   #�����Щ�����Ķ�Ӧ�ı�ǩֵ
    print 'there are %d support vectors' % shape(sVs)[0]   #�������õ�����������
    m,n=shape(datamat)
    errorcount=0
    for i in range(m):
        kerneleval=kerneltrans(sVs,datamat[i,:],('rbf',k1))   #����Kֵ
        predict=kerneleval.T*multiply(labelSV,alphas[svInd])+b #����ֱ����k�Ĺ���ѵ�����ľ��ߺ��������࣬�������ʮ����Ҫ
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
    
    
    
    
# _*_ ���ļ��е�32*32����ת����1*1024�� _*_
def to_32(filename):
    returnos=zeros((1,1024))   #����һ��1��1024�е������
    ma=open(filename)   
    for i in range(32):   #��0������31
        lintr=ma.readline()   #readline()Ϊ�Ķ�ÿ�к�����������ݵ���һ��һ���Ķ�
        for j in range(32):   #������
            returnos[0,i*32+j]=lintr[j]   #��ÿһ�е����ݸ�returnos������
    return returnos   #ת����ϣ�����returnos
    
# _*_����ͼƬ�ļ�����
def loadimages(dirname):
    from os import listdir
    trainingtation=listdir(dirname)#��lsitdir������ȡ���������ļ����ÿ���ļ���
    m=len(trainingtation)   #���������ļ��������ܺ�
    trainingclocks=zeros((m,1024))   #����m��1024�еľ���
    hwlabels=[]   #����һ�����б����ڴ������������ı�ǩ
    for i in range(m):   #�����ļ�����
        fN=trainingtation[i]   #����ѡȡ�ļ���
        fS=fN.split('.')[0]   #ͨ����.���и��ļ�������ȡ��һ����
        fS=int(fS.split('_')[0])   #ͨ����_���и�ʣ���ļ�������ȥ��һ���򣬼���ʾ���ֵ�ֵ
        if fS==9:hwlabels.append(-1)
        else:hwlabels.append(1)   #�����ֱ�ǩ�����ǩ�����б���
        trainingclocks[i,:]=to_32('E:\Users\Alan Lin\Desktop\\trainingDigits/%s' % fN)   #��ÿ���ļ��е�32*32����ת����1*1024�ľ���
    return trainingclocks,hwlabels

# _*_ʶ����д���ֺ���
def testdigits(ktup=('rbf',10)):
    dataarr,labelarr=loadimages('E:\Users\Alan Lin\Desktop\\trainingDigits')   #������Լ�
    b,alphas=smop(dataarr,labelarr,200,0.0001,10000,ktup)   #����Ҫ�������룬k1����kֵ���õ�b��alphasֵ
    datamat=mat(dataarr);labelmat=mat(labelarr).transpose()
    svInd=nonzero(alphas.A>0)[0]   #�õ�alphas����0��λ�ã��������õ�֧��������λ��
    #print 'alphas.A:',alphas.A
    sVs=datamat[svInd]   #��ȡ��Щ���õ�����ֵ
    labelSV=labelmat[svInd]   #�����Щ�����Ķ�Ӧ�ı�ǩֵ
    print 'there are %d support vectors' % shape(sVs)[0]   #�������õ�����������
    m,n=shape(datamat)
    errorcount=0
    for i in range(m):
        kerneleval=kerneltrans(sVs,datamat[i,:],ktup)   #����Kֵ
        predict=kerneleval.T*multiply(labelSV,alphas[svInd])+b #����ֱ����k�Ĺ���ѵ�����ľ��ߺ��������࣬�������ʮ����Ҫ
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
    
    
    

    
    
    
    
    
    
    
    
    