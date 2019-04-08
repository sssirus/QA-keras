# coding=utf-8
from keras import Input, Model
import numpy as np
from keras.layers import Embedding, Conv1D, Dropout, MaxPooling1D, Flatten, merge, Dense

import globalvar as gl
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

gl.set_NUM_FILTERS(150)
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



relation_embedding_layer = Embedding(len(wd_idx) + 1, EMBEDDING_DIM, input_length=relation_maxlen, weights=[embedding_matrix], trainable=True)(tweet_relation)

relation_conv1 = Conv1D(128, 3, activation='tanh')(relation_embedding_layer)
relation_drop_1 = Dropout(0.2)(relation_conv1)
relation_max_1 = MaxPooling1D(relation_maxlen-3+1)(relation_drop_1)
relation_conv2 = Conv1D(128, 1, activation='tanh')(relation_max_1)
relation_drop_2 = Dropout(0.2)(relation_conv2)
relation_dmax_2 = MaxPooling1D(1)(relation_drop_2)
#conv2 = Conv1D(128, 3, activation='tanh')(max_1)
#max_2 = MaxPooling1D(3)(conv2)
relation_out_1 = Flatten()(relation_dmax_2)
#out_1 = LSTM(128)(max_1)
#question
question_embedding_layer = Embedding(len(wd_idx) + 1, EMBEDDING_DIM, input_length=ques_maxlen, weights=[embedding_matrix], trainable=True)(tweet_ques)

question_conv1 = Conv1D(128, 3, activation='tanh')(question_embedding_layer)
question_drop_1 = Dropout(0.2)(question_conv1)
question_max_1 = MaxPooling1D(ques_maxlen-3+1)(question_drop_1)
question_conv2 = Conv1D(128, 1, activation='tanh')(question_max_1)
question_drop_2 = Dropout(0.2)(question_conv2)
question_dmax_2 = MaxPooling1D(1)(question_drop_2)
#conv2 = Conv1D(128, 3, activation='tanh')(max_1)
#max_2 = MaxPooling1D(3)(conv2)
question_out_1 = Flatten()(question_dmax_2)
#out_1 = LSTM(128)(max_1)


merged_vector = merge([relation_out_1, question_out_1], mode='concat') # good
dense_1 = Dense(128,activation='relu')(merged_vector)
dense_2 = Dense(128,activation='relu')(dense_1)
dense_3 = Dense(128,activation='relu')(dense_2)

predictions = Dense(1, activation='sigmoid')(dense_3)
#predictions = Dense(len(labels_index), activation='softmax')(merged_vector)

model = Model(input=[tweet_relation, tweet_ques], output=predictions)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit([rela_train,ques_train], label_train, nb_epoch=20,batch_size=5,verbose=1,shuffle=True)
json_string = model.to_json()  # json_string = model.get_config()
open('my_model_architecture.json','w').write(json_string)
model.save_weights('my_model_weights.h5')

score = model.evaluate([rela_train,ques_train], label_train, verbose=0)
print('train score:', score[0])
print('train accuracy:', score[1])
score = model.evaluate([rela_test, ques_test], label_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
a = model.predict([rela_test,ques_test])

print('Predicted:')
for lines in a:
    for line in lines:
        print(line)