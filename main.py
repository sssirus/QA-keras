# coding=utf-8
from keras import Input, Model
import globalvar as gl
from keras.layers import Embedding, Dropout, LSTM, Bidirectional, concatenate, Conv1D, MaxPooling1D, Flatten, Dense
from data_preprocessor import preprocess,generateWord2VectorMatrix
from attention import AttentionLayer

#全局变量
train_files="train.txt"
test_files="test.txt"
preprocessWordVector_files="word2vector.txt"
preprocessWordVector_path=""
MAX_NB_WORDS=10000
EMBEDDING_DIM=100
LSTM_DIM=150

gl._init()

gl.set_value('train_files', train_files)
gl.set_value('test_files', test_files)
gl.set_value('preprocessWordVector_files', preprocessWordVector_files)
gl.set_value('preprocessWordVector_path', preprocessWordVector_path)
gl.set_value('MAX_NB_WORDS', MAX_NB_WORDS)
gl.set_value('EMBEDDING_DIM', EMBEDDING_DIM)
gl.set_value('LSTM_DIM', LSTM_DIM)

#预处理
ques_train, rela_train, ques_test, rela_test,wd_idx=preprocess(train_files,test_files)
embedding_matrix=generateWord2VectorMatrix(preprocessWordVector_path,preprocessWordVector_files,wd_idx)
# Embedding+dropout层(输出是三维)
relation_maxlen= gl.get_value('relation_maxlen')
ques_maxlen= gl.get_value('ques_maxlen')
##question
ques_embedding_layer = Embedding(len(wd_idx) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=ques_maxlen,
                            trainable=False)

question_input = Input(shape=(ques_maxlen,), dtype='int32')
embedded_question = ques_embedding_layer(question_input)
embedded_question = Dropout(0.35)(embedded_question)
##relation
rela_embedding_layer = Embedding(len(wd_idx) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=relation_maxlen,
                            trainable=False)#如果这里报错参考https://www.jianshu.com/p/795a5e2cd10c

rela_input = Input(shape=(relation_maxlen,), dtype='int32')
embedded_relation = rela_embedding_layer(rela_input)
embedded_relation = Dropout(0.35)(embedded_relation)

#Bi-LSTM 层，对问题+关系进行表示（输出是三维）


embedded_relation = Bidirectional(LSTM(LSTM_DIM, activation='tanh', return_sequences=True),merge_mode='concat')(embedded_relation)
embedded_relation = Dropout(0.35)(embedded_relation)

embedded_question = Bidirectional(LSTM(LSTM_DIM, activation='tanh', return_sequences=True),merge_mode='concat')(embedded_question)
embedded_question = Dropout(0.35)(embedded_question)
#Attention层
l_att = AttentionLayer()(embedded_relation,embedded_question)
#Attention层的拼接
merge_layer = concatenate([embedded_question , l_att] , axis=1)

#CNN层
convs = []
NUM_FILTERS =150
gl.set_value('NUM_FILTERS', NUM_FILTERS)
filter_sizes = [1,3,5]
for fsz in filter_sizes:
    conv1 = Conv1D(NUM_FILTERS,kernel_size=fsz,activation='tanh')(merge_layer)
    pool1 = MaxPooling1D(ques_maxlen+relation_maxlen-fsz+1)(conv1)#max-pooling
    pool1 = Flatten()(pool1)
    convs.append(pool1)
merge = concatenate(convs,axis=1)
out = Dropout(0.35)(merge)
#输出层

dense_1 = Dense(NUM_FILTERS, activation='relu')(out)
dense_2 = Dense(NUM_FILTERS, activation='relu')(dense_1)
dense_3 = Dense(NUM_FILTERS, activation='relu')(dense_2)

predictions = Dense(1, activation='sigmoid')(dense_3)
# predictions = Dense(len(labels_index), activation='softmax')(merged_vector)

model = Model(input=[question_input, rela_input], output=predictions)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 下面是训练程序
model.fit([ques_train,rela_train], labels_train, nb_epoch=5)
json_string = model.to_json()  # json_string = model.get_config()
open('my_model_architecture.json','w').write(json_string)
model.save_weights('my_model_weights.h5')
# 下面是训练得到的神经网络进行评估
score = model.evaluate([ques_train,rela_train], labels_train, verbose=0)
print('train score:', score[0]) # 训练集中的loss
print('train accuracy:', score[1]) # 训练集中的准确率
score = model.evaluate([ques_test, rela_test], labels_val, verbose=0)
print('Test score:', score[0])#测试集中的loss
print('Test accuracy:', score[1]) #测试集中的准确率




