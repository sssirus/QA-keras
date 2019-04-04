from keras.models import load_model
from keras_applications.imagenet_utils import decode_predictions

import globalvar as gl
from data_preprocessor import tokenize
from keras.preprocessing.sequence import pad_sequences

def parse_relation (RelaFile):
    relation = []
    with open(RelaFile) as f2:
        for rela in f2:
            rela = rela.strip()
            relation.append(tokenize(rela))
    return relation


def fenci(inpute_question):
    pass
def vectorize_dialog(data, wd_idx, maxlen ):
#向量化,返回对应词表的索引号
    vec = []
    for words in data:
        idx = [wd_idx[w] for w in words]
        vec.append(idx)
    #序列长度归一化，分别找出对话，问题和答案的最长长度，然后相对应的对数据进行padding。
    return  pad_sequences(vec, maxlen = maxlen)

def predicated(inpute_question):
    NUM_OF_RELATIONS=1000;
    RelaFile=""
    #加载模型
    model = load_model('my_model.h5')
    #构造数据
    inpute_questions=fenci(inpute_question)
    question = [tokenize(inpute_questions)]
    #questions = question * NUM_OF_RELATIONS
    relations = parse_relation(RelaFile)
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
    ques_maxlen= gl.get_ques_maxlen()
    rela_maxlen= gl.get_relation_maxlen()
    # 对训练集和测试集，进行word2vec
    question_vec = vectorize_dialog(question, wd_idx,  ques_maxlen)
    relation_vec = vectorize_dialog(relations, wd_idx, rela_maxlen)
    questions_vec = question_vec * NUM_OF_RELATIONS
    #批量预测
    y = model.predict([questions_vec,relation_vec])
    results=decode_predictions(y, top=1)
    print('Predicted:', results)
    #结果
    return results
