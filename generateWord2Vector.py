#-*- coding: utf-8 -*-
import io
import numpy as np
import os

from itertools import islice

from data_preprocessor import parse_dialog


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False

def generateVector(lexicon):


    fw = io.open(os.path.join(preprocessWordVector_path, "reducedW2V.txt"), 'w', encoding='utf-8')
    f = io.open(os.path.join(preprocessWordVector_path, preprocessWordVector_files), 'r', encoding='UTF-8')
    i=0
    for line in islice(f, 1, None):
        values = line.split()

        try:
            coefs = np.asarray(values[1:], dtype='float32')
            word = values[0]
            #print(word)
        except:
            continue
        #print(is_chinese(word))
        if(word in lexicon):
            fw.write(line)
            i=i+1
    f.close()
    fw.close()
    print("总共词向量数：")
    print(i)
    return

def loadWord(preprocessWordVector_path,preprocessWordVector_files):
    # 建立词表。词表就是文本中所有出现过的单词组成的词表。

    fw = io.open(os.path.join(preprocessWordVector_path, "reducedW2V.txt"), 'w', encoding='utf-8')
    f = io.open(os.path.join(preprocessWordVector_path, preprocessWordVector_files), 'r', encoding='UTF-8')
    i=0
    for line in islice(f, 1, None):
        values = line.split()

        try:
            coefs = np.asarray(values[1:], dtype='float32')
            word = values[0]
            #print(word)
        except:
            continue
        #print(is_chinese(word))
        if(len(word)<=10 and is_chinese(word)):
            fw.write(line)

            i=i+1
    f.close()
    fw.close()
    print("词表总共单词数：")
    print(i)
    return



def selectHighFrequencyWord(preprocessWordVector_path,preprocessWordVector_files):
    # 建立词表。词表就是文本中所有出现过的单词组成的词表。
    temp = []

    f = io.open(os.path.join(preprocessWordVector_path, preprocessWordVector_files), 'r', encoding='UTF-8')

    for line in f:
        values = line.split("\t")
        word = values[0]
        frequency = values[1]
        if(frequency>0):
            temp.append(word)
    f.close()

    return temp

def selectSimilarWord(preprocessWordVector_path,preprocessWordVector_files):


    return


def loadDatasetWord(train_rela_files,train_ques_file,train_label_file,test_rela_files,test_ques_file,test_label_file):
    train_data = parse_dialog(train_ques_file, train_rela_files, train_label_file)
    test_data = parse_dialog(test_ques_file, test_rela_files, test_label_file)
    print("train_data")
    print(np.array(train_data).shape)
    print("test_data")
    print(np.array(test_data).shape)
    # 建立词表。词表就是文本中所有出现过的单词组成的词表。
    lexicon = set()
    for ques, rela, labe in train_data + test_data:
        lexicon |= set(ques + rela)

    return list(lexicon)


train_rela_files="train_case_rela.txt"
train_ques_file="train_case_ques.txt"
train_label_file="train_case_label.txt"
test_rela_files="test_case_rela.txt"
test_ques_file="test_case_ques.txt"
test_label_file="test_case_label.txt"
preprocessWordVector_path="/data/ylx/"
preprocessWordVector_files="Tencent_AILab_ChineseEmbedding.txt"
chineseWordFile="HighFrequencyWords.txt"
temp=selectHighFrequencyWord(preprocessWordVector_path,chineseWordFile)
words=loadDatasetWord(train_rela_files,train_ques_file,train_label_file,test_rela_files,test_ques_file,test_label_file)
words+=temp
lexicon = set(words)
lexicon_size = len(lexicon) + 1
print("lexicon_size")
print(lexicon_size)
generateVector(lexicon)