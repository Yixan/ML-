import numpy as np
from functools import reduce
numberOfLines=0
def file2matrix(filename):
	#打开文件,此次应指定编码，
	fr = open(filename,'r',encoding = 'utf-8')
	#读取文件所有内容
	arrayOLines = fr.readlines()
	#针对有BOM的UTF-8文本，应该去掉BOM，否则后面会引发错误。
	arrayOLines[0]=arrayOLines[0].lstrip('\ufeff')
	#得到文件行数
	global numberOfLines
	numberOfLines = len(arrayOLines)
	#返回的NumPy矩阵,解析完成的数据:numberOfLines行,3列
	returnMat=[]
	classVec=list()
	for line in arrayOLines:
		#s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
		line = line.strip()
		#使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
		listFromLine = line.split(',')
		numberOfColumn=len(listFromLine)
		returnMat.append(listFromLine[0:numberOfColumn-1])
		if listFromLine[-1]!='\s+' and listFromLine[-1]!=None:
			classVec.append(listFromLine[-1].strip())
	return returnMat[200:numberOfLines],returnMat[0:200],classVec


def createVocabList(dataSet):
	vocabSet = set([])  # 创建一个空的不重复列表
	for document in dataSet:
		vocabSet = vocabSet | set(document)  # 取并集
	return list(vocabSet)

def setOfWords2Vec(vocabList, inputSet):
	returnVec = [0] * len(vocabList)									#创建一个其中所含元素都为0的向量
	for word in inputSet:												#遍历每个词条
		if word in vocabList:											#如果词条存在于词汇表中，则置1
			returnVec[vocabList.index(word)] = 1
		else: print("the word: %s is not in my Vocabulary!" % word)
	return returnVec													#返回文档向量

def trainNB0(trainMatrix,trainCategory):					#(集合，标签）
	numTrainDocs = len(trainMatrix)							#计算训练的文档行数
	classSet=set(trainCategory)
	classList=[]
	for val in classSet:
		classList.append(val)
	pAbusive=[0 for i in range(len(classList))]
	s = [0 for x in range(len(classList))]
	for i in range(numTrainDocs):
		for j in range(len(classList)):
			if trainCategory[i]==classList[j]:
				s[j]=s[j]+1
	for i in range(len(classList)):
		pAbusive[i] = s[i]/float(numTrainDocs)		#每一类的概率
	pNum=[[0 for j in range(len(trainMatrix[0]))] for i in range(len(classList))]
	pDenom=[0 for i in range(len(classList))]
	for i in range(numTrainDocs):#行数
		for j in range(len(classList)):
			if trainCategory[i]==classList[j]:
				for k in range(len(trainMatrix[0])):
					pNum[j][k]=trainMatrix[i][k]+pNum[j][k]
				pDenom[j] += sum(trainMatrix[i])
	pVect=[0 for i in range(len(classList))]
	for j in range(len(classList)):
		pVect[j] = pNum[j]/pDenom[j]									#相除
	return pVect,pAbusive,classList						#返回属于侮辱类的条件概率数组，属于非侮辱类的条件概率数组，文档属于侮辱类的概率

def classifyNB(vec2Classify, pVec, pClass):
	p=[0 for i in range(len(pClass))]
	pMax=0
	textClass=-1

	for i in range(len(pVec)):
		pItem = 1
		for j in range(len(vec2Classify)):
			if vec2Classify[j]!=0 :
				pItem *=vec2Classify[j]*pVec[i][j]
		p[i] =pItem*pClass[i]
		if p[i]>pMax:
			pMax=p[i]
			textClass=i
	return textClass


if __name__ == '__main__':
	# 创建训练集和测试集
	returnMat,testSet,classVec=file2matrix("nursery.txt")
	#创建词汇表
	myVocabList = createVocabList(returnMat)
	print(myVocabList)
	#训练样本向量化
	trainMat=[]
	testMat=[]
	for postinDoc in returnMat:
		trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
	#测试样本向量化
	for postinDoc in testSet:
		testMat.append(setOfWords2Vec(myVocabList, postinDoc))
	#训练朴素贝叶斯分类器
	pV,pAb,classList= trainNB0(np.array(trainMat), np.array(classVec[200:numberOfLines]))
	e=0.0
	for i in range(len(testMat)):
		realClass=classVec[i]
		textNum =classifyNB(testMat[i],pV,pAb)
		if textNum !=-1:
			textClass=classList[textNum]
			print("分类结果：%s,真实类别：%s"%(textClass,realClass))
			if textClass!=realClass:
				e+=1
		else:
			print("ErrorClass")
	print("错误率：",e/200)
