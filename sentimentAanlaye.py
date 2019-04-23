#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/12/12 0012 上午 10:34
# @Author : XXX
# @Function : 情感分析
# @File : sentimentAanlaye.py
# @IDE ：python 3.7
import re
import os
import jieba
jieba.load_userdict(r'D:\python\Lib\site-packages\jieba\dict2.txt')
import time
from numpy import *
t_start = time.time()
os.chdir(r'E:\guthub\gensim_w2v\知网Hownet情感词典')
#-----------------训练原始文本数据----------
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
try:
    import cPickle as pickle
except ImportError:
    import pickle
'''
sentences = LineSentence(r'E:\guthub\w2v\trunk\mt20181209.txt')
model = Word2Vec(sentences, size=200, window=8, min_count=4)
model.save(r'E:\guthub\gensim_w2v\mt20181213_model')

#--------------加载并求出最接近的词汇---------------
model = Word2Vec.load(r'E:\guthub\gensim_w2v\mt20181213_model')
rootWord = open('E:\\guthub\\gensim_w2v\\rootWord.txt','r',encoding = 'utf_8').read().split('\n')
file = open('E:\\guthub\\gensim_w2v\\levelbroadWord.txt','w',encoding = 'utf_8')
print(len(rootWord))
for j in range(len(rootWord)):
    if rootWord[j] not in model:
        continue
    else:
        print(rootWord[j])
        most_similar = model.most_similar(rootWord[j],topn = 20)#找最接近的50个词
        for Word in most_similar:
            file.write(rootWord[j] + ' '+Word[0] + ' '+ str(Word[1])+'\n')
file.close()

'''
#---------------------情感分析------------------
add_punc = "!？｡\\ # ＄％＆＇（）*＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗\"〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
def readTestText(filepath):#读入测试语料
    testText = [line.strip().split('\t')[0] for line in open(filepath, 'r', encoding = 'utf_8')]

    return testText
def stopWordList(filepath):#停用词表
    stopWords = [line.strip() for line in open(filepath,'r',encoding = 'utf_8').readlines()]

    return stopWords
def positivedict(filepath):#褒义词表
    positiveDict = [line.strip() for line in open(filepath,'r',encoding = 'utf_8').readlines()]
    return positiveDict
def negetivedict(filepath):#贬义词表
    negetiveDict = [line.strip() for line in open(filepath,'r',encoding = 'utf_8').readlines()]
    return negetiveDict
def leveldict(filepath):#程度副词表
    levelList =[]
    levelDict = []
    Dict = [line.strip() for line in open(filepath,'r',encoding = 'utf_8').readlines()]
    for line in Dict:
        element = line.split(',')
        levelList.append(float(element[1]))
        levelDict.append(element[0])
    return levelDict,levelList
def denydict(filepath):#否定词表
    denyDict = [line.strip() for line in open(filepath,'r',encoding = 'utf_8').readlines()]
    return denyDict
def themedict(filepath):#主题词表
    themeList =[]
    themeDict = []
    Dict = [line.strip() for line in open(filepath,'r',encoding = 'utf_8').readlines()]
    for line in Dict:
        element = line.split(',')
        themeList.append(float(element[1]))
        themeDict.append(element[0])
    return themeDict,themeList
themeDict,themeList = themedict('themedict.txt')
testText = readTestText('add_text.txt')
positiveDict = positivedict('positiveDict.txt')
negetiveDict = negetivedict('negetiveDict.txt')
levelDict,levelList = leveldict('levelDict.txt')
denyDict = denydict('denyDict.txt')
stopWords = stopWordList('stop_Words.txt')
print('-------所有字典读取完毕------')
lengthOfText = len(testText)
label = open('testTextlabel.txt','w',encoding ="utf_8")
for i in range(lengthOfText):#分句
    content = testText[i].strip()
    regex = "，|。|！|？|、|\\s+"
    content = re.split(regex,content)#testText[i] =[' ……'，'……']
    len_sentence = len(content)
    sentimentList = zeros(len_sentence)#设置与总句等长度的0向量
    themeWeightlist = []
    for k in range(len_sentence):#对每个分句进行分词
        #content = re.sub(add_punc, '',testText[k])#去除无效符号
        seg_list = list(jieba.cut(content[k], cut_all=True))
        print(seg_list)
        lenPartSentence = len(seg_list)
        partSentimentList = zeros(lenPartSentence)  # 设置与分句等长度的0向量
        now_senti_index = 0
        last_senti_index = 0
        themeWeight =1
        deny_num = 0
        numofpoint = 0  # 感叹号的个数
        levelweight = 1
        
        for ind,word in enumerate(seg_list):#遍历分句中的每个词,用item可以同时获取索引
            if word in stopWords:#去停用用词
                continue
            else:
                if word in themeDict:
                    themeind = themeDict.index(word)
                    themeWeight = themeList[themeind]
                    print('主题权重：',themeWeight)
                if word == '！':
                    numofpoint += 1
                    print('感叹号个数：',numofpoint)
                if word in positiveDict:#褒义词判定
                    partSentimentList[ind] = 1
                    now_senti_index = ind
                    for level_ind in range(last_senti_index,now_senti_index):
                        #上个情感词与当前情感词之间搜索程度副词
                        if seg_list[level_ind] in levelDict:
                            w_ind = levelDict.index(seg_list[level_ind])
                            levelweight = levelList[w_ind]
                            print("程度副词权重：",levelweight)
                            partSentimentList[now_senti_index] = partSentimentList[now_senti_index]*levelweight#(该程度的权重)
                        else:
                            partSentimentList[now_senti_index] = partSentimentList[now_senti_index]* 1
                    for deny_ind in range(last_senti_index,now_senti_index):#上个情感词与当前情感词之间搜索否定词
                        if seg_list[deny_ind ] in denyDict:
                            deny_num += 1
                    print('否定词个数：',deny_num)
                    if deny_num % 2 == 1:#否定词个数为偶数则整体乘以1
                        partSentimentList[now_senti_index] = partSentimentList[now_senti_index]* (-1)
                    else:#否定词个数为奇数则整体乘以-1
                        partSentimentList[now_senti_index] = partSentimentList[now_senti_index]* (1)
                    last_senti_index = now_senti_index+1   # 将当前情感词索引赋值给上一个情感词索引
                elif word in negetiveDict:#贬义词判定
                    partSentimentList[ind] = -1
                    now_senti_index = ind
                    for level_ind in range(last_senti_index,now_senti_index):
                        # 上个情感词与当前情感词之间搜索程度副词
                        if seg_list[level_ind] in levelDict:
                            w_ind = levelDict.index(seg_list[level_ind])
                            levelweight = levelList[w_ind]
                            partSentimentList[now_senti_index] = partSentimentList[now_senti_index] * levelweight  # (该程度的权重)
                        else:
                            partSentimentList[now_senti_index] = partSentimentList[now_senti_index] * 1
                    for deny_ind in range(last_senti_index,now_senti_index):
                        # 上个情感词与当前情感词之间搜索否定词
                        if seg_list[deny_ind] in denyDict:
                            deny_num += 1
                    print('否定词个数：', deny_num)
                    if deny_num % 2 == 0:
                        partSentimentList[now_senti_index] = partSentimentList[now_senti_index]*(1)
                    else:  # 否定词个数为奇数则整体乘以-1
                        partSentimentList[now_senti_index] = partSentimentList[now_senti_index]*(-1)
                    last_senti_index = now_senti_index+1  # 将当前情感词索引赋值给上一个情感词索引

        print('情感向量：',partSentimentList)
        parttotalsentiment = sum(partSentimentList)*themeWeight
        themeWeightlist.append(themeWeight)
        partSentimentScore = parttotalsentiment
        print('分句得分：',partSentimentScore)
        sentimentList[k] = partSentimentScore
    sentimentScore = sum(sentimentList)#计算总句的情感极性值
    print('总句得分：', sentimentScore)
    if sentimentScore > 0:
        label.write(str(testText[i]) + '@@@' + str(sentimentScore) +'@@@'+str(themeWeightlist)+'@@@'+str(sentimentList) +'\n')
    elif sentimentScore == 0:
        label.write(str(testText[i])+ '@@@'  + str(sentimentScore) +'@@@'+str(themeWeightlist)+'@@@'+str(sentimentList) +'\n')
    else:
        label.write(str(testText[i]) + '@@@'  + str(sentimentScore) +'@@@'+str(themeWeightlist)+'@@@'+str(sentimentList) +'\n')

 