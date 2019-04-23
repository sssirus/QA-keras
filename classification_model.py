# coding=utf-8
from keras import Input, Model
from keras.layers import Embedding, Dropout, LSTM, Bidirectional, concatenate, Conv1D, MaxPooling1D, Flatten, Dense, merge



def test_cnn( num_categories , maxlen, embedding_dims,wd_idx,embedding_matrix, filters=250):
    # Inputs
    seq = Input(shape=[maxlen], name='x_seq')

    # Embedding layers
    emb = Embedding(len(wd_idx) + 1, embedding_dims, input_length=maxlen,
                                         weights=[embedding_matrix], trainable=False, name="ques_embedding_layer")(
        seq)

    # conv layers
    convs = []
    filter_sizes = [2, 3, 4, 5]
    for fsz in filter_sizes:
         conv1 = Conv1D(filters, kernel_size=fsz, activation='tanh')(emb)
         pool1 = MaxPooling1D(maxlen - fsz + 1)(conv1)
         pool1 = Flatten()(pool1)
         convs.append(pool1)
    merge = concatenate(convs, axis=1)

    out = Dropout(0.5)(merge)
    output = Dense(32, activation='relu')(out)

    output = Dense(units=num_categories, activation='sigmoid')(output)

    model = Model([seq], output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
