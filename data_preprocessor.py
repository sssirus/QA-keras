#coding=utf-8
import os
import re
import numpy as np
import globalvar as gl
from keras.preprocessing.sequence import pad_sequences
MAX_NB_WORDS = gl.get_value('MAX_NB_WORDS')
EMBEDDING_DIM = gl.get_value('EMBEDDING_DIM')
#将每个单词按空格分割来
def tokenize(data):
   # \s 匹配任何空白字符，包括空格、制表符、换页符等等。等价于 [ \f\n\r\t\v]。
   return [x.strip() for x in re.split(r'[;,\s]\s*', data) if x.strip()]

# 用——将问题和谓词切开

def parse_lines(lines):
    ques = []
    rela = []
    for line in lines:
        line = line.strip()
        relationship, question = line.split('——',1)
        ques.append(tokenize(question))
        rela.append(tokenize(relationship))
    return ques, rela

#将（问题，答案）组成相对应的tuple存储。
#这里的maxlen是控制文本最大长度的，可以利用分位数找出覆盖90%数据的长度，令其为maxlen。
# 否则序列长度太长，训练时内存不够。
def get_lines(files, max_length = None):
    data = parse_lines(files.readlines())
    data = [(ques, rela) for (ques, rela) in data if not max_length ]
    return data

#数据长度归一化。找出问题文本的最大单词长度，对所有的问题进行padding，将长度归一化。关系集同此。
def vectorize_dialog(data,wd_idx, relation_maxlen, ques_maxlen):
#向量化,返回对应词表的索引号
    ques_vec = []
    rela_vec = []
    for ques, rela in data:
        rela_idx = [wd_idx[w] for w in rela]
        ques_idx = [wd_idx[w] for w in ques]
        ques_vec.append(ques_idx)
        rela_vec.append(rela_idx)

    #序列长度归一化，分别找出对话，问题和答案的最长长度，然后相对应的对数据进行padding。
    return pad_sequences(ques_vec, maxlen=ques_maxlen), \
       pad_sequences(rela_vec, maxlen=relation_maxlen)

#从Tecent AI Lab文件中解析出每个词和它所对应的词向量，并用字典的方式存储。
# 然后根据得到的字典生成上文所定义的词向量矩阵
def generateWord2VectorMatrix(path,filename,wd_idx):
    embeddings_index = {}
    f = open(os.path.join(path, filename))
    for line in f:
        values = line.split()
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(wd_idx) + 1, EMBEDDING_DIM))
    for word, i in wd_idx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

def preprocess(train_files,test_files):

    # 准备数据
    train = get_lines(train_files)
    test = get_lines(test_files)
    # 建立词表。词表就是文本中所有出现过的单词组成的词表。
    lexicon = set()
    for ques, rela in train + test:
        lexicon |= set(ques + rela)
    lexicon = sorted(lexicon)
    # word2vec，并求出对话集和问题集的最大长度，padding时用。
    wd_idx = dict((wd, idx+1) for idx, wd in enumerate(lexicon))
    relation_maxlen = max(map(len, (x for x, _, _ in train + test)))
    ques_maxlen = max(map(len, (x for _, x, _ in train + test)))

    gl.set_value('relation_maxlen',relation_maxlen)
    gl.set_value('ques_maxlen', ques_maxlen)
    # 计算分位数，在get_dialog函数中传参给max_len
    #dia_80 = np.percentile(map(len, (x for x, _, _ in train + test)), 80)
    # 对训练集和测试集，进行word2vec
    ques_train, rela_train = vectorize_dialog(train, wd_idx, relation_maxlen, ques_maxlen)
    ques_test, rela_test = vectorize_dialog(test, wd_idx, relation_maxlen, ques_maxlen)
    return ques_train, rela_train, ques_test, rela_test, wd_idx