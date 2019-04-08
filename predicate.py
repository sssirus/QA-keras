#coding=utf-8

import numpy as np

from keras.models import model_from_json

import globalvar as gl
from data_preprocessor import tokenize, loadEmbeddingsIndex, generateWord2VectorMatrix
from keras.preprocessing.sequence import pad_sequences
from loadModel import creatModel, creat_model_for_predicate



def decode_predictions2(preds, top):
    CLASS_INDEX = []
    with open('./data/relation.txt')as f2:
        for rela in f2:
            rela = rela.strip()
            CLASS_INDEX.append(rela)
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_INDEX[i]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results



def parse_relation (RelaFile):
    relation = []
    with open(RelaFile) as f2:
        for rela in f2:
            #print(rela)
            relation.append(tokenize(rela.strip()))
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

    NUM_OF_RELATIONS=708;
    gl.set_NUM_OF_RELATIONS(NUM_OF_RELATIONS)
    RelaFile="./data/test.txt"


    #构造数据

    print(inpute_question)
    question = [tokenize(inpute_question)]
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
    ques_maxlen= 7
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
    #questions_vec=np.tile(question_vec,(NUM_OF_RELATIONS,1))
    print("questions_vec")
    print(np.array(question_vec).shape)
    print("relation_vec")
    print(np.array(relation_vec).shape)

    # 进行embeddings
    EMBEDDING_DIM=200
    gl.set_EMBEDDING_DIM(EMBEDDING_DIM)
    embedding_index = loadEmbeddingsIndex("/data/zjy/", "TencentPreTrain.txt")
    embedding_matrix = generateWord2VectorMatrix(embedding_index, wd_idx)
    #改变矩阵形状
    #embedding_matrix_re = np.resize(embedding_matrix,(1611, 200))
    # 加载模型
    NUM_FILTERS=150
    LSTM_DIM =150
    gl.set_LSTM_DIM(LSTM_DIM)
    model=creat_model_for_predicate(EMBEDDING_DIM,wd_idx,embedding_matrix,ques_maxlen,rela_maxlen,NUM_FILTERS,LSTM_DIM)

    model.load_weights(filepath='my_model_weights.h5', by_name=True)
    #predicate_rela_embedding_layer=model.get_layer(name="predicate_rela_embedding_layer")
    #predicate_rela_embedding_layer.set_weights([embedding_matrix])
    #predicate_ques_embedding_layer = model.get_layer(name="predicate_ques_embedding_layer")
    #predicate_ques_embedding_layer.set_weights([embedding_matrix])
    print("查看模型")
    for layer in model.layers:
        for weight in layer.weights:
            print (weight.name,weight.shape)

    #批量预测
    y = model.predict([question_vec,relation_vec])
    results=decode_predictions2(y, top=1)
    print('Predicted:', y)
    #结果
    return results
#加载模型
# 加载模型结构
#model = model_from_json(open('my_model_architecture.json').read())

# 加载模型参数
#model.load_weights('my_model_weight.h5')

#predicated("ENTITY 创始人 哪位 ")