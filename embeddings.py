# coding=utf-8
from keras import Sequential
from keras.layers import Embedding, Permute


def getEmbeddings(input,EMBEDDING_DIM,embedding_matrix,length,maxlen):
    embedding_model = Sequential()
    embedding_model.add(Embedding(length + 1,
                                       EMBEDDING_DIM,
                                       weights=[embedding_matrix],
                                       input_length=maxlen,
                                       trainable=False))
    # 模型将输入一个大小为 (batch, input_length) 的整数矩阵。
    # 输入中最大的整数（即词索引）不应该大于 999 （词汇表大小）
    # 现在 model.output_shape == (None, 10, 64)，其中 None 是 batch 的维度。

    embedding_model.compile('rmsprop', 'mse')
    output_array = embedding_model.predict(input)

    return output_array.transpose(0,2,1)