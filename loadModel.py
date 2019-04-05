# coding=utf-8
from keras import Input, Model

from keras.layers import Embedding, Dropout, LSTM, Bidirectional, concatenate, Conv1D, MaxPooling1D, Flatten, Dense

from attention import attention_3d_block
from keras import backend as K

def creatModel(EMBEDDING_DIM,wd_idx,embedding_matrix,ques_maxlen,relation_maxlen,NUM_FILTERS,LSTM_DIM):
    ##question
    ques_embedding_layer = Embedding(len(wd_idx) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=ques_maxlen,
                            trainable=False,
                            name="train_ques_embedding_layer")

    question_input = Input(shape=(ques_maxlen,), dtype='int32')
    embedded_question = ques_embedding_layer(question_input)

    print ("question_input: ")
    print(K.int_shape(question_input))
    print ("embedded_question: ")
    print(K.int_shape(embedded_question))
    ##relation

    rela_embedding_layer = Embedding(len(wd_idx) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=relation_maxlen,
                            trainable=False,
                            name="train_rela_embedding_layer")

    rela_input = Input(shape=(relation_maxlen,), dtype='int32')
    embedded_relation = rela_embedding_layer(rela_input)

    print ("rela_input:")
    print (K.int_shape(rela_input))
    print ("embedded_relation")
    print (K.int_shape(embedded_relation))
    #LSTM 层，对问题+关系进行表示（输出是三维）


    embedded_relation = LSTM(LSTM_DIM, activation='tanh', return_sequences=True,name="lstm_relation")(embedded_relation)
    embedded_relation = Dropout(0.35)(embedded_relation)

    embedded_question = LSTM(LSTM_DIM, activation='tanh', return_sequences=True,name="lstm_question")(embedded_question)
    embedded_question = Dropout(0.35)(embedded_question)
    #Attention层
    print ("---------------")
    l_att = attention_3d_block(embedded_relation,embedded_question)
    #Attention层的拼接
    merge_layer = concatenate([embedded_question , l_att] , axis=1)

    #CNN层
    convs = []

    #for fsz in filter_sizes:
    ## 1
    conv1 = Conv1D(NUM_FILTERS,kernel_size=1,activation='tanh',name="cnn_1_conv")(merge_layer)
    pool1 = MaxPooling1D(ques_maxlen+relation_maxlen-1+1,name="cnn_1_maxpool")(conv1)#max-pooling
    pool1 = Flatten()(pool1)
    convs.append(pool1)
    ## 3
    conv2 = Conv1D(NUM_FILTERS,kernel_size=3,activation='tanh',name="cnn_2_conv")(merge_layer)
    pool2 = MaxPooling1D(ques_maxlen+relation_maxlen-3+1,name="cnn_2_maxpool")(conv2)#max-pooling
    pool2 = Flatten()(pool2)
    convs.append(pool2)
    ## 5
    conv3 = Conv1D(NUM_FILTERS,kernel_size=5,activation='tanh',name="cnn_3_conv")(merge_layer)
    pool3 = MaxPooling1D(ques_maxlen+relation_maxlen-5+1,name="cnn_3_maxpool")(conv3)#max-pooling
    pool3 = Flatten()(pool3)
    convs.append(pool3)

    merge = concatenate(convs,axis=1)
    out = Dropout(0.35)(merge)
    #输出层

    dense_1 = Dense(NUM_FILTERS, activation='relu',name="output_dense_1")(out)
    predictions = Dense(1, activation='sigmoid',name="output")(dense_1)
    # predictions = Dense(len(labels_index), activation='softmax')(merged_vector)
    model = Model(input=[question_input, rela_input], output=predictions)
    return model
def creat_model_for_predicate(EMBEDDING_DIM,wd_idx,embedding_matrix,ques_maxlen,relation_maxlen,NUM_FILTERS,LSTM_DIM):
    ##question
    ques_embedding_layer = Embedding(len(wd_idx) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=ques_maxlen,
                            trainable=False,
                            name="predicate_ques_embedding_layer")

    question_input = Input(shape=(ques_maxlen,), dtype='int32')
    embedded_question = ques_embedding_layer(question_input)

    print ("question_input: ")
    print(K.int_shape(question_input))
    print ("embedded_question: ")
    print(K.int_shape(embedded_question))
    ##relation

    rela_embedding_layer = Embedding(len(wd_idx) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=relation_maxlen,
                            trainable=False,
                            name="predicate_rela_embedding_layer")

    rela_input = Input(shape=(relation_maxlen,), dtype='int32')
    embedded_relation = rela_embedding_layer(rela_input)

    print ("rela_input:")
    print (K.int_shape(rela_input))
    print ("embedded_relation")
    print (K.int_shape(embedded_relation))
    #LSTM 层，对问题+关系进行表示（输出是三维）


    embedded_relation = LSTM(LSTM_DIM, activation='tanh', return_sequences=True,name="lstm_relation")(embedded_relation)
    embedded_relation = Dropout(0.35)(embedded_relation)

    embedded_question = LSTM(LSTM_DIM, activation='tanh', return_sequences=True,name="lstm_question")(embedded_question)
    embedded_question = Dropout(0.35)(embedded_question)
    #Attention层
    print ("---------------")
    l_att = attention_3d_block(embedded_relation,embedded_question)
    #Attention层的拼接
    merge_layer = concatenate([embedded_question , l_att] , axis=1)

    #CNN层
    convs = []

    #for fsz in filter_sizes:
    ## 1
    conv1 = Conv1D(NUM_FILTERS,kernel_size=1,activation='tanh',name="cnn_1_conv")(merge_layer)
    pool1 = MaxPooling1D(ques_maxlen+relation_maxlen-1+1,name="cnn_1_maxpool")(conv1)#max-pooling
    pool1 = Flatten()(pool1)
    convs.append(pool1)
    ## 3
    conv2 = Conv1D(NUM_FILTERS,kernel_size=3,activation='tanh',name="cnn_2_conv")(merge_layer)
    pool2 = MaxPooling1D(ques_maxlen+relation_maxlen-3+1,name="cnn_2_maxpool")(conv2)#max-pooling
    pool2 = Flatten()(pool2)
    convs.append(pool2)
    ## 5
    conv3 = Conv1D(NUM_FILTERS,kernel_size=5,activation='tanh',name="cnn_3_conv")(merge_layer)
    pool3 = MaxPooling1D(ques_maxlen+relation_maxlen-5+1,name="cnn_3_maxpool")(conv3)#max-pooling
    pool3 = Flatten()(pool3)
    convs.append(pool3)

    merge = concatenate(convs,axis=1)
    out = Dropout(0.35)(merge)
    #输出层

    dense_1 = Dense(NUM_FILTERS, activation='relu',name="output_dense_1")(out)
    predictions = Dense(1, activation='sigmoid',name="output")(dense_1)
    # predictions = Dense(len(labels_index), activation='softmax')(merged_vector)
    model = Model(input=[question_input, rela_input], output=predictions)
    return model