# coding=utf-8
from keras import Input, Model
from keras.layers import Embedding, Dropout, LSTM, Bidirectional, concatenate, Conv1D, MaxPooling1D, Flatten, Dense
from attention import attention_3d_block


def creatCNNModel(EMBEDDING_DIM,wd_idx,embedding_matrix,ques_maxlen,relation_maxlen,NUM_FILTERS,LSTM_DIM,DROPOUT_RATE = 0.01):
    tweet_relation = Input(shape=(relation_maxlen,))
    tweet_ques = Input(shape=(ques_maxlen,))



    relation_embedding_layer = Embedding(len(wd_idx) + 1, EMBEDDING_DIM, input_length=relation_maxlen,
                                         weights=[embedding_matrix], trainable=False,name="ques_embedding_layer")(tweet_relation)

    lstm_relation = Bidirectional(LSTM(LSTM_DIM, activation='tanh', return_sequences=True,name="lstm_relation"), merge_mode='concat')(
        relation_embedding_layer)
    lstm_relation = Dropout(DROPOUT_RATE)(lstm_relation)

    # question
    question_embedding_layer = Embedding(len(wd_idx) + 1, EMBEDDING_DIM, input_length=ques_maxlen,
                                         weights=[embedding_matrix], trainable=False,name="rela_embedding_layer")(tweet_ques)

    lstm_question = Bidirectional(LSTM(LSTM_DIM, activation='tanh', return_sequences=True,name="lstm_quesiton"), merge_mode='concat')(
        question_embedding_layer)
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
    merged_vector = concatenate(convs)
    #merged_vector = merge(convs, mode='concat')  # good
    dense_1 = Dense(NUM_FILTERS, activation='relu', name="dense_1")(merged_vector)
    dense_2 = Dense(NUM_FILTERS, activation='relu', name="dense_2")(dense_1)
    dense_3 = Dense(NUM_FILTERS, activation='relu', name="dense_3")(dense_2)

    predictions = Dense(1, activation='sigmoid', name="predictions")(dense_3)
    # predictions = Dense(len(labels_index), activation='softmax')(merged_vector)

    model = Model(input=[tweet_relation, tweet_ques], output=predictions)
    return model
def creat_model_for_predicate(EMBEDDING_DIM,wd_idx,embedding_matrix,ques_maxlen,relation_maxlen,NUM_FILTERS,LSTM_DIM,DROPOUT_RATE = 0.01):
    tweet_relation = Input(shape=(relation_maxlen,))
    tweet_ques = Input(shape=(ques_maxlen,))



    relation_embedding_layer = Embedding(len(wd_idx) + 1, EMBEDDING_DIM, input_length=relation_maxlen,
                                         weights=[embedding_matrix], trainable=False,name="predicate_ques_embedding_layer")(tweet_relation)

    lstm_relation = Bidirectional(LSTM(LSTM_DIM, activation='tanh', return_sequences=True,name="lstm_relation"), merge_mode='concat')(
        relation_embedding_layer)
    lstm_relation = Dropout(DROPOUT_RATE)(lstm_relation)

    # question
    question_embedding_layer = Embedding(len(wd_idx) + 1, EMBEDDING_DIM, input_length=ques_maxlen,
                                         weights=[embedding_matrix], trainable=False,name="predicate_rela_embedding_layer")(tweet_ques)

    lstm_question = Bidirectional(LSTM(LSTM_DIM, activation='tanh', return_sequences=True,name="lstm_quesiton"), merge_mode='concat')(
        question_embedding_layer)
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

    merged_vector = merge(convs, mode='concat')  # good
    dense_1 = Dense(NUM_FILTERS, activation='relu', name="dense_1")(merged_vector)
    dense_2 = Dense(NUM_FILTERS, activation='relu', name="dense_2")(dense_1)
    dense_3 = Dense(NUM_FILTERS, activation='relu', name="dense_3")(dense_2)

    predictions = Dense(1, activation='sigmoid', name="predictions")(dense_3)
    # predictions = Dense(len(labels_index), activation='softmax')(merged_vector)

    model = Model(input=[tweet_relation, tweet_ques], output=predictions)
    return model