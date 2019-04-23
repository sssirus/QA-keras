#-*- coding: utf-8 -*-
from imp import reload
import numpy as np
import io
from keras.models import model_from_json

import globalvar as gl
from data_preprocessor import tokenize, loadEmbeddingsIndex, generateWord2VectorMatrix
from keras.preprocessing.sequence import pad_sequences
from loadModel import creatModel, creat_model_for_predicate


import sys
reload(sys)
sys.setdefaultencoding('utf8')

def decode_predictions2(preds, top=1):
    CLASS_INDEX = []

    f2 = io.open('./data/relation.txt', 'r', encoding='UTF-8')
    for rela in f2:
        rela = rela.strip()
        CLASS_INDEX.append(rela)
        print(rela.decode('string_escape'))
    f2.close()

    top_indices = preds.argmax()
    tag=CLASS_INDEX[top_indices+1]
    num=preds[top_indices]
    return tag,num,top_indices





def parse_relation (RelaFile):
    relation = []
    f2 = io.open(RelaFile, 'r', encoding='UTF-8')
    for rela in f2:
        #print(rela)
        relation.append(tokenize(rela.strip()))
    f2.close()
    return relation


def vectorize_dialog(data, wd_idx, maxlen ):
#向量化,返回对应词表的索引号
    vec = []
    for line in data:
        idx = [wd_idx[w] for w in line]
        vec.append(idx)
    #序列长度归一化，分别找出对话，问题和答案的最长长度，然后相对应的对数据进行padding。
    return  pad_sequences(vec, maxlen = maxlen)

def predicated(inpute_question):
    NUM_OF_RELATIONS=708
    gl.set_NUM_OF_RELATIONS(NUM_OF_RELATIONS)
    RelaFile="./data/relation_fenci.txt"


    #构造数据
    str2 = inpute_question.decode('utf-8', 'ignore')
    print(str2)
    question = [tokenize(str2)]
    #print(str(question).decode('string_escape'))
    relations = parse_relation(RelaFile)
    #print(str(relations).decode('string_escape'))
    #f = open('relation.txt', 'w')  # 若是'wb'就表示写二进制文件
    #f.write(str(relations))
    #f.close()
    #f = open('question.txt', 'w')  # 若是'wb'就表示写二进制文件
    #f.write(str(question))
    #f.close()

    #数据预处理
    # 建立词表。词表就是文本中所有出现过的单词组成的词表。
    lexicon = set()
    for words in question :
        lexicon |= set(words)
    for words in relations :
        lexicon |= set(words)
    #print("lexicon:")
    #print(str(lexicon).decode('string_escape'))
    lexicon = sorted(lexicon)
    lexicon_size = len(lexicon) + 1
    print("lexicon_size")
    print(lexicon_size)
    wd_idx = dict((wd, idx + 1) for idx, wd in enumerate(lexicon))
    #获取问题和关系最大长度
    ques_maxlen= 11
    rela_maxlen= 3
    gl.set_ques_maxlen(ques_maxlen)
    gl.set_relation_maxlen(rela_maxlen)
    # 对训练集和测试集，进行word2vec
    question_vec = vectorize_dialog(question, wd_idx,  ques_maxlen)
    relation_vec = vectorize_dialog(relations, wd_idx, rela_maxlen)
    print("question_vec:")
    print(str(question_vec).decode('string_escape'))
    print("relation_vec:")
    print(str(relation_vec).decode('string_escape'))
    questions_vec=np.tile(question_vec,(NUM_OF_RELATIONS,1))
    print("questions_vec")
    print(np.array(questions_vec).shape)
    print("relation_vec")
    print(np.array(relation_vec).shape)

    # 进行embeddings
    EMBEDDING_DIM=200
    gl.set_EMBEDDING_DIM(EMBEDDING_DIM)
    embedding_index = loadEmbeddingsIndex("/data/zjy/", "TencentPreTrain.txt")
    embedding_matrix = generateWord2VectorMatrix(embedding_index, wd_idx)

    # 加载模型
    NUM_FILTERS=128
    LSTM_DIM =150
    gl.set_LSTM_DIM(LSTM_DIM)
    model=creat_model_for_predicate(EMBEDDING_DIM,wd_idx,embedding_matrix,ques_maxlen,rela_maxlen,NUM_FILTERS,LSTM_DIM,0.01)

    model.load_weights(filepath='my_model_weights.h5', by_name=True)


    print("查看模型")
    for layer in model.layers:
        for weight in layer.weights:
            print (weight.name,weight.shape)

    #批量预测
    y = model.predict([relation_vec,questions_vec])
    print(y)
    tag,num,index=decode_predictions2(y, top=1)
    print('Predicted tag=:')
    print(tag.decode('string_escape'))
    print('Predicted num=:')
    print(num)
    print('Predicted index=:')
    print(index)
    #print(result)
    #print('Predicted:')
    #for lines in result:
    #   for line in lines:
    #      print(str(line).decode('string_escape'))
    #结果
    return tag



predicated("创始人")