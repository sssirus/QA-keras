# coding=utf-8
from keras import Input, Model
from keras.layers import Embedding, Dropout, LSTM, Bidirectional, concatenate, Conv1D, MaxPooling1D, Flatten, Dense, merge
import numpy as np
from keras import backend as K

def test_cnn( num_categories , maxlen, embedding_dims,wd_idx,embedding_matrix, filters=128):
    # Inputs
    seq = Input(shape=[maxlen], name='x_seq')

    # Embedding layers
    emb = Embedding(len(wd_idx) + 1, embedding_dims, input_length=maxlen,
                                         weights=[embedding_matrix], trainable=False, name="ques_embedding_layer")(
        seq)
    print("emb_out")
    print(K.int_shape(emb))

    # conv layers
    convs = []
    filter_sizes = [1,2, 3, 4, 5]
    for fsz in filter_sizes:
         conv1 = Conv1D(filters, kernel_size=fsz, activation='tanh')(emb)
         pool1 = MaxPooling1D(maxlen - fsz + 1)(conv1)
         pool1 = Flatten()(pool1)
         convs.append(pool1)
    merge = concatenate(convs, axis=1)
    print("conv_out")
    #print(np.array(merge).shape)
    print(K.int_shape(merge))
    out = Dropout(0.1)(merge)
    dense_1 = Dense(32, activation='relu')(out)
    dense_2 = Dense(128, activation='relu')(dense_1)
    dense_3 = Dense(128, activation='relu')(dense_2)
    print("Dense1_out")
    #print(np.array(output).shape)
    print(K.int_shape(dense_3))
    output = Dense(units=num_categories, activation='sigmoid')(dense_3)
    print("Dense2_out")
    #print(np.array(output).shape)
    print(K.int_shape(output))
    model = Model([seq], output)
    model.summary()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
