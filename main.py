# coding=utf-8
from keras import Input, Model
import numpy as np
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
LSTM_DIM=gl.get_LSTM_DIM()

gl.set_NUM_FILTERS(128)
gl.set_filter_sizes([1,3,5])

NUM_FILTERS = gl.get_NUM_FILTERS()
filter_sizes = gl.get_filter_sizes()


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
# Embedding+dropout层(输出是三维)
relation_maxlen= gl.get_relation_maxlen()
ques_maxlen= gl.get_ques_maxlen()

model=creatModel(EMBEDDING_DIM,wd_idx,embedding_matrix,ques_maxlen,relation_maxlen,NUM_FILTERS,LSTM_DIM,0.01)
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
