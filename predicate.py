#coding=utf-8
from keras.models import load_model
import numpy as np
import globalvar as gl
from data_preprocessor import tokenize, loadEmbeddingsIndex, generateWord2VectorMatrix
from keras.preprocessing.sequence import pad_sequences
from embeddings import getEmbeddings

def decode_predictions_custom(preds, CLASS_CUSTOM,top=5):
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(CLASS_CUSTOM[i]) + (pred[i]*100,) for i in top_indices]
        results.append(result)
    return results


def get_class_custom (RelaFile):
    CLASS_CUSTOM = []
    with open(RelaFile) as f2:
        for rela in f2:
            rela = rela.strip()
            CLASS_CUSTOM.append(rela)
    return CLASS_CUSTOM

def parse_relation (RelaFile):
    relation = []
    with open(RelaFile) as f2:
        for rela in f2:
            rela = rela.strip()
            relation.append(tokenize(rela))
    return relation


def vectorize_dialog(data, wd_idx, maxlen ):
#向量化,返回对应词表的索引号
    vec = []
    for words in data:
        idx = [wd_idx[w] for w in words]
        vec.append(idx)
    #序列长度归一化，分别找出对话，问题和答案的最长长度，然后相对应的对数据进行padding。
    return  pad_sequences(vec, maxlen = maxlen)

def predicated(inpute_question):

    NUM_OF_RELATIONS=708;
    gl.set_NUM_OF_RELATIONS(NUM_OF_RELATIONS)
    RelaFile="./data/relation_fenci.txt"

    #加载模型
    model = load_model('my_model.h5')
    #构造数据
    CLASS_CUSTOM = get_class_custom("./data/relation.txt")
    question = [tokenize(inpute_question)]

    relations = parse_relation(RelaFile)
    print("question")
    print(question)
    print("question")
    print(relations)
    #数据预处理
    # 建立词表。词表就是文本中所有出现过的单词组成的词表。
    lexicon = set()
    for words in question + relations:
        lexicon |= set(words)

    lexicon = sorted(lexicon)
    lexicon_size = len(lexicon) + 1
    print("lexicon_size")
    print(lexicon_size)
    wd_idx = dict((wd, idx + 1) for idx, wd in enumerate(lexicon))
    #获取问题和关系最大长度
    ques_maxlen= 7
    rela_maxlen= 3
    # 对训练集和测试集，进行word2vec
    question_vec = vectorize_dialog(question, wd_idx,  ques_maxlen)
    relation_vec = vectorize_dialog(relations, wd_idx, rela_maxlen)

    questions_vec=np.tile(question_vec, (NUM_OF_RELATIONS, 1))
    print("questions_vec")
    print(np.array(questions_vec).shape)
    print("relation_vec")
    print(np.array(relation_vec).shape)
    # 进行embeddings
    EMBEDDING_DIM=200
    embedding_index = loadEmbeddingsIndex("/data/zjy/", "TencentPreTrain.txt")
    embedding_matrix = generateWord2VectorMatrix(embedding_index, wd_idx)
    ques_embedding = getEmbeddings(question_vec, EMBEDDING_DIM, embedding_matrix, len(wd_idx), ques_maxlen)
    rela_embedding = getEmbeddings(relation_vec, EMBEDDING_DIM, embedding_matrix, len(wd_idx), rela_maxlen)
    #批量预测
    y = model.predict([ques_embedding,rela_embedding], batch_size=10, verbose=1)
    results=decode_predictions_custom(y,CLASS_CUSTOM, top=1)
    print('Predicted:', y)
    #结果
    return results
predicated("ENTITY 创始人 哪位 ")