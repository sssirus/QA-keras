# coding=utf-8
from keras import Input, Model
import numpy as np
import globalvar as gl
from keras.layers import Embedding, Dropout, LSTM, Bidirectional, concatenate, Conv1D, MaxPooling1D, Flatten, Dense
from data_preprocessor import preprocess,generateWord2VectorMatrix,loadEmbeddingsIndex
from attention import attention_3d_block
from keras import backend as K
from embeddings import getEmbeddings
#全局变量



gl.set_train_rela_files("train_case_rela.txt")
gl.set_train_ques_file("train_case_ques.txt")
gl.set_train_label_file("train_case_label.txt")
gl.set_test_rela_files("test_case_rela.txt")
gl.set_test_ques_file("test_case_ques.txt")
gl.set_test_label_file("test_case_label.txt")
gl.set_preprocessWordVector_files("TencentPreTrain.txt")
gl.set_preprocessWordVector_path("/data/zjy/")
gl.set_MAX_NB_WORDS(10000)
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
LSTM_DIM=gl.get_LSTM_DIM()



#预处理
ques_train, rela_train,label_train, ques_test, rela_test, label_test,wd_idx=preprocess(train_rela_files,train_ques_file,train_label_file,test_rela_files,test_ques_file,test_label_file)
embedding_index=loadEmbeddingsIndex(preprocessWordVector_path,preprocessWordVector_files)
embedding_matrix=generateWord2VectorMatrix(embedding_index,wd_idx)
print("ques_train")
print(np.array(ques_train).shape)
print("rela_train")
print(np.array(rela_train).shape)
print("label_train")
print(np.array(label_train).shape)
print("ques_test")
print(np.array(ques_test).shape)
print("rela_test")
print(np.array(rela_test).shape)
print("label_test")
print(np.array(label_test).shape)
# Embedding+dropout层(输出是三维)仍然算作数据预处理
relation_maxlen= gl.get_relation_maxlen()
ques_maxlen= gl.get_ques_maxlen()

##question
ques_embedding_train = getEmbeddings(ques_train,EMBEDDING_DIM,embedding_matrix,len(wd_idx),ques_maxlen)
ques_embedding_test = getEmbeddings(ques_test,EMBEDDING_DIM,embedding_matrix,len(wd_idx),ques_maxlen)
##relation
rela_embedding_train = getEmbeddings(rela_train,EMBEDDING_DIM,embedding_matrix,len(wd_idx),relation_maxlen)
rela_embedding_test = getEmbeddings(rela_test,EMBEDDING_DIM,embedding_matrix,len(wd_idx),relation_maxlen)


#模型正式开始。。。
question_input = Input(shape=(EMBEDDING_DIM,ques_maxlen,), dtype='float32')
rela_input = Input(shape=(EMBEDDING_DIM,relation_maxlen,), dtype='float32')

#LSTM层

embedded_relation = LSTM(LSTM_DIM, activation='tanh', return_sequences=True)(rela_input)
embedded_relation = Dropout(0.35)(embedded_relation)

embedded_question = LSTM(LSTM_DIM, activation='tanh', return_sequences=True)(question_input)
embedded_question = Dropout(0.35)(embedded_question)
#Attention层
print ("---------------")
l_att = attention_3d_block(embedded_relation,embedded_question)
#Attention层的拼接
merge_layer = concatenate([embedded_question , l_att] , axis=1)

#CNN层
convs = []

gl.set_NUM_FILTERS(150)
gl.set_filter_sizes([1,3,5])

NUM_FILTERS=gl.get_NUM_FILTERS()
filter_sizes=gl.get_filter_sizes()

for fsz in filter_sizes:
    conv1 = Conv1D(NUM_FILTERS,kernel_size=fsz,activation='tanh')(merge_layer)
    pool1 = MaxPooling1D(ques_maxlen+relation_maxlen-fsz+1)(conv1)#max-pooling
    pool1 = Flatten()(pool1)
    convs.append(pool1)
merge = concatenate(convs,axis=1)
out = Dropout(0.35)(merge)
#输出层

dense_1 = Dense(NUM_FILTERS, activation='relu')(out)
predictions = Dense(1, activation='sigmoid')(dense_1)
# predictions = Dense(len(labels_index), activation='softmax')(merged_vector)

model = Model(input=[question_input, rela_input], output=predictions)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 下面是训练程序
model.fit([ques_embedding_train,rela_embedding_train], label_train, nb_epoch=5,batch_size=32,verbose=1)
json_string = model.to_json()  # json_string = model.get_config()
open('my_model_architecture.json','w').write(json_string)
model.save('my_model.h5')
# 下面是训练得到的神经网络进行评估
score = model.evaluate([ques_embedding_train,rela_embedding_train], label_train, verbose=0)
print('train score:', score[0]) # 训练集中的loss
print('train accuracy:', score[1]) # 训练集中的准确率
score = model.evaluate([ques_embedding_test, rela_embedding_test], label_test, verbose=0)
print('Test score:', score[0])#测试集中的loss
print('Test accuracy:', score[1]) #测试集中的准确率




