# coding=utf-8
from keras import Input, Model
import numpy as np
from keras.layers import Embedding, Conv1D, Dropout, MaxPooling1D, Flatten, merge, Dense, LSTM, regularizers, \
    Bidirectional, concatenate

import globalvar as gl
from attention import attention_3d_block
from data_preprocessor import preprocess,generateWord2VectorMatrix,loadEmbeddingsIndex
from loadModel import creatModel
#全局变量



gl.set_train_rela_files("train_case_rela.txt")
gl.set_train_ques_file("train_case_ques.txt")
gl.set_train_label_file("train_case_label.txt")
gl.set_test_rela_files("test_case_rela.txt")
gl.set_test_ques_file("test_case_ques.txt")
gl.set_test_label_file("test_case_label.txt")
gl.set_preprocessWordVector_files("TencentPreTrain.txt")
gl.set_preprocessWordVector_path("/data/zjy/")
gl.set_MAX_NB_WORDS(30)
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

gl.set_NUM_FILTERS(128)
gl.set_filter_sizes([1,3,5])

NUM_FILTERS = gl.get_NUM_FILTERS()
filter_sizes = gl.get_filter_sizes()


#预处理



ques_train, rela_train,label_train, ques_test, rela_test, label_test,wd_idx=preprocess(train_rela_files,train_ques_file,train_label_file,test_rela_files,test_ques_file,test_label_file)
embedding_index=loadEmbeddingsIndex(preprocessWordVector_path,preprocessWordVector_files)
embedding_matrix=generateWord2VectorMatrix(embedding_index,wd_idx)
#model
relation_maxlen= gl.get_relation_maxlen()
ques_maxlen= gl.get_ques_maxlen()
tweet_relation = Input(shape=(relation_maxlen,))
tweet_ques = Input(shape=(ques_maxlen,))

DROPOUT_RATE=0.01



relation_embedding_layer = Embedding(len(wd_idx) + 1, EMBEDDING_DIM, input_length=relation_maxlen, weights=[embedding_matrix], trainable=True)(tweet_relation)


lstm_relation=Bidirectional(LSTM(LSTM_DIM, activation='tanh', return_sequences=True),merge_mode='concat')(relation_embedding_layer)
lstm_relation = Dropout(DROPOUT_RATE)(lstm_relation)

#question
question_embedding_layer = Embedding(len(wd_idx) + 1, EMBEDDING_DIM, input_length=ques_maxlen, weights=[embedding_matrix], trainable=True)(tweet_ques)

lstm_question=Bidirectional(LSTM(LSTM_DIM, activation='tanh', return_sequences=True),merge_mode='concat')(question_embedding_layer)
lstm_question = Dropout(DROPOUT_RATE)(lstm_question)

# Attention层
print("---------------")
l_att = attention_3d_block(lstm_relation, lstm_question)
# Attention层的拼接
merge_layer = concatenate([lstm_question, l_att], axis=1)


# CNN层
convs = []
# for fsz in filter_sizes:
## 1
conv1 = Conv1D(NUM_FILTERS, kernel_size=1, activation='tanh', name="cnn_1_conv")(merge_layer)
pool1 = MaxPooling1D(ques_maxlen + relation_maxlen - 1 + 1, name="cnn_1_maxpool")(conv1)  # max-pooling
pool1 = Flatten()(pool1)
convs.append(pool1)
## 3
conv2 = Conv1D(NUM_FILTERS, kernel_size=3, activation='tanh', name="cnn_2_conv")(merge_layer)
pool2 = MaxPooling1D(ques_maxlen + relation_maxlen - 3 + 1, name="cnn_2_maxpool")(conv2)  # max-pooling
pool2 = Flatten()(pool2)
convs.append(pool2)
## 5
conv3 = Conv1D(NUM_FILTERS, kernel_size=5, activation='tanh', name="cnn_3_conv")(merge_layer)
pool3 = MaxPooling1D(ques_maxlen + relation_maxlen - 5 + 1, name="cnn_3_maxpool")(conv3)  # max-pooling
pool3 = Flatten()(pool3)
convs.append(pool3)

merged_vector = merge(convs, mode='concat') # good
dense_1 = Dense(NUM_FILTERS,activation='relu')(merged_vector)
dense_2 = Dense(NUM_FILTERS,activation='relu')(dense_1)
dense_3 = Dense(NUM_FILTERS,activation='relu')(dense_2)

predictions = Dense(1, activation='sigmoid')(dense_3)
#predictions = Dense(len(labels_index), activation='softmax')(merged_vector)

model = Model(input=[tweet_relation, tweet_ques], output=predictions)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit([rela_train,ques_train], label_train, nb_epoch=10,batch_size=20,verbose=1,shuffle=True)
json_string = model.to_json()  # json_string = model.get_config()
open('my_model_architecture.json','w').write(json_string)
model.save_weights('my_model_weights.h5')

score = model.evaluate([rela_train,ques_train], label_train, verbose=0)
print('train score:', score[0])
print('train accuracy:', score[1])
score = model.evaluate([rela_test, ques_test], label_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
#a = model.predict([rela_test,ques_test])

#print('Predicted:')
#for lines in a:
#    for line in lines:
#        print(line)