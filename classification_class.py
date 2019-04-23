# coding=utf-8
import numpy as np
from keras.utils import np_utils
from keras_preprocessing.sequence import pad_sequences
from tensorflow.python.estimator import keras
import globalvar as gl
from classification_model import test_cnn
from data_preprocessor import parse_dialog, loadEmbeddingsIndex, generateWord2VectorMatrix
#全局变量


gl.set_train_rela_files("train_case_classify_rela.txt")
gl.set_train_ques_file("train_case_classify_ques.txt")
gl.set_train_label_file("train_case_classify_label.txt")
gl.set_test_rela_files("test_case_rela.txt")
gl.set_test_ques_file("test_case_ques.txt")
gl.set_test_label_file("test_case_label.txt")
gl.set_preprocessWordVector_files("TencentPreTrain.txt")
gl.set_preprocessWordVector_path("/data/zjy/")
gl.set_MAX_NB_WORDS(50)
gl.set_EMBEDDING_DIM(200)
gl.set_LSTM_DIM(150)

train_rela_files=gl.get_train_rela_files()
train_ques_file=gl.get_train_ques_file()
train_label_file=gl.get_train_label_file()
test_rela_files=gl.get_test_rela_files()
test_ques_file=gl.get_test_ques_file()
test_label_file=gl.get_test_label_file()
preprocessWordVector_files=gl.get_preprocessWordVector_files()
preprocessWordVector_path=gl.get_preprocessWordVector_path()
MAX_NB_WORDS=gl.get_MAX_NB_WORDS()
EMBEDDING_DIM=gl.get_EMBEDDING_DIM()
def vectorize_dialog(data,wd_idx, ques_maxlen):
#向量化,返回对应词表的索引号
    label = []
    ques_vec = []
    relation = []
    for ques ,rela, labe in data:
        ques_idx = [wd_idx[w] for w in ques]
        ques_vec.append(ques_idx)
        relation.append(rela)
        label.append(labe)
    #序列长度归一化，分别找出对话，问题和答案的最长长度，然后相对应的对数据进行padding。
    return pad_sequences(ques_vec, maxlen = ques_maxlen),\
        relation,label
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
        lexicon |= set(ques)
    lexicon = sorted(lexicon)
    lexicon_size = len(lexicon) + 1
    print("lexicon_size")
    print(lexicon_size)
    # word2vec，并求出对话集和问题集的最大长度，padding时用。
    wd_idx = dict((wd, idx + 1) for idx, wd in enumerate(lexicon))
    ques_maxlen = max(map(len, (x for x, _, _ in train_data + test_data)))


    print("ques_maxlen")
    print(ques_maxlen)


    # 计算分位数，在get_dialog函数中传参给max_len
    #dia_80 = np.percentile(map(len, (x for x, _, _ in train + test)), 80)
    gl.set_ques_maxlen(ques_maxlen)

    # 对训练集和测试集，进行word2vec
    question_train, relation_train, label_train = vectorize_dialog(train_data, wd_idx,  ques_maxlen)
    question_test, relation_test, label_test = vectorize_dialog(test_data, wd_idx,  ques_maxlen)

    return question_train, relation_train,label_train, question_test, relation_test, label_test,wd_idx
def runcnn():
    # 预处理
    ques_train, rela_train, label_train, ques_test, rela_test, label_test, wd_idx = preprocess(train_rela_files,
                                                                                               train_ques_file,
                                                                                               train_label_file,
                                                                                               test_rela_files,
                                                                                               test_ques_file,
                                                                                               test_label_file)
    embedding_index = loadEmbeddingsIndex(preprocessWordVector_path, preprocessWordVector_files)
    embedding_matrix = generateWord2VectorMatrix(embedding_index, wd_idx)
    print("ques_train")
    print(np.array(ques_train).shape)
    print("rela_train")
    print(np.array(rela_train).shape)

    print("ques_test")
    print(np.array(ques_test).shape)
    print("rela_test")
    print(np.array(rela_test).shape)

    # Embedding+dropout层(输出是三维)

    maxlen = gl.get_ques_maxlen()
    rela_train_label = np_utils.to_categorical(rela_train) #多分类时，此方法将1，2，3，4，....这样的分类转化成one-hot 向量的形式，最终使用softmax做为输出
    rela_test_label = np_utils.to_categorical(rela_test)
    '''
    print(x.shape,y.shape)
    indices = np.arange(len(x))
    lenofdata = len(x)
    np.random.shuffle(indices)
    x_train = x[indices][:int(lenofdata*0.8)]
    y_train = y[indices][:int(lenofdata*0.8)]
    x_test = x[indices][int(lenofdata*0.8):]
    y_test = y[indices][int(lenofdata*0.8):]
    '''

    model = test_cnn( 708 , maxlen, EMBEDDING_DIM,wd_idx,embedding_matrix, filters=250)
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=0,
        verbose=0,
        mode='auto')
    print("training model")
    history = model.fit(ques_train,rela_train_label,validation_split=0.2,batch_size=64,epochs=10,verbose=2,shuffle=True)
    accy=history.history['acc']
    np_accy=np.array(accy)
    np.savetxt('save.txt',np_accy)

    print("pridicting...")
    scores = model.evaluate(ques_test,rela_test_label)
    print('test_loss:%f,accuracy: %f'%(scores[0],scores[1]))

    print("saving %s_textcnnmodel")
    model.save('./predictor/model%s_cnn_large.h5')
    return