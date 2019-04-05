#coding=utf-8
import os
import re
import numpy as np
import globalvar as gl
from keras.preprocessing.sequence import pad_sequences
from itertools import islice


MAX_NB_WORDS = gl.get_MAX_NB_WORDS()


#将每个单词分割来
def tokenize(data):
    import re
    #  \s 匹配任何空白字符，包括空格、制表符、换页符等等。等价于 [ \f\n\r\t\v]。
    return [x.strip() for x in re.split(r'[\s]', data) if x.strip()]

# parse_dialog 将所有的对话进行解析，返回tokenize后的句子
# 如果 only_supporting为真表明只返回含有答案的对话
def parse_dialog(QuesFile,RelaFile,LabelFile):
    data = []
    question = []
    relation = []
    label = []
    with open(QuesFile) as f1:
        for ques in f1:
            ques = ques.strip()
            question.append(tokenize(ques))
    with open(RelaFile) as f2:
        for rela in f2:
            rela = rela.strip()
            relation.append(tokenize(rela))
    with open(LabelFile) as f3:
        for labe in f3:
            labe = labe.strip()
            label.append(labe)

    for i in range(0, len(label)):
        data.append((question[i],relation[i],label[i]))
    return data


def vectorize_dialog(data,wd_idx, rela_maxlen, ques_maxlen):

#向量化,返回对应词表的索引号

    label = []
    ques_vec = []
    rela_vec = []
    for ques ,rela, labe in data:
        rela_idx = [wd_idx[w] for w in rela]
        ques_idx = [wd_idx[w] for w in ques]
        ques_vec.append(ques_idx)
        rela_vec.append(rela_idx)
        label.append(labe)
    #序列长度归一化，分别找出对话，问题和答案的最长长度，然后相对应的对数据进行padding。
    return pad_sequences(ques_vec, maxlen = ques_maxlen),\
        pad_sequences(rela_vec, maxlen = rela_maxlen),label

def preprocess(train_rela_files,train_ques_file,train_label_file,test_rela_files,test_ques_file,test_label_file):
    # 准备数据

    train_data = parse_dialog(train_ques_file,train_rela_files,train_label_file)
    test_data = parse_dialog(test_ques_file, test_rela_files, test_label_file)
    print("train_data")
    print(np.array(train_data).shape)
    print("test_data")
    print(np.array(test_data).shape)
    # 建立词表。词表就是文本中所有出现过的单词组成的词表。
    lexicon = set()
    for ques, rela, labe in train_data + test_data:
        lexicon |= set(ques + rela)
    lexicon = sorted(lexicon)
    lexicon_size = len(lexicon) + 1
    print("lexicon_size")
    print(lexicon_size)
    # word2vec，并求出对话集和问题集的最大长度，padding时用。
    wd_idx = dict((wd, idx + 1) for idx, wd in enumerate(lexicon))
    ques_maxlen = max(map(len, (x for x, _, _ in train_data + test_data)))
    rela_maxlen = max(map(len, (x for _, x, _ in train_data + test_data)))

    print("ques_maxlen")
    print(ques_maxlen)

    print("rela_maxlen")
    print(rela_maxlen)
    # 计算分位数，在get_dialog函数中传参给max_len
    #dia_80 = np.percentile(map(len, (x for x, _, _ in train + test)), 80)
    gl.set_ques_maxlen(ques_maxlen)
    gl.set_relation_maxlen(rela_maxlen)
    # 对训练集和测试集，进行word2vec
    question_train, relation_train, label_train = vectorize_dialog(train_data, wd_idx, rela_maxlen, ques_maxlen)
    question_test, relation_test, label_test = vectorize_dialog(test_data, wd_idx, rela_maxlen, ques_maxlen)

    return question_train, relation_train,label_train, question_test, relation_test, label_test,wd_idx

#从Tecent AI Lab文件中解析出每个词和它所对应的词向量，并用字典的方式存储。
# 然后根据得到的字典生成上文所定义的词向量矩阵
def loadEmbeddingsIndex(path,filename):
    embeddings_index = {}
    f = open(os.path.join(path, filename))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('找到 %s 个词向量。' % len(embeddings_index))
    return embeddings_index
def generateWord2VectorMatrix(embeddings_index, wd_idx):
    EMBEDDING_DIM=gl.get_EMBEDDING_DIM()
    embedding_matrix = np.zeros((len(wd_idx) + 1, EMBEDDING_DIM))
    for word, i in wd_idx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    return embedding_matrix