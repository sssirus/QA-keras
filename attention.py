# Attention GRU network

from keras.layers import Permute, Dense, multiply, Activation, Dot
import globalvar as gl
from keras import backend as K
def attention_3d_block(input_shapeR, input_shapeQ):
    relation_maxlen = gl.get_relation_maxlen()
    LSTM_DIM=gl.get_LSTM_DIM()
    # inputs.shape = (batch_size, time_steps, seq_len)
    print("input_shapeR: ")
    print(K.int_shape(input_shapeR))
    print("input_shapeQ: ")
    print(K.int_shape(input_shapeQ))


    mid = Dense(LSTM_DIM,name="att_dense")(input_shapeR)
    print("mid: ")
    print(K.int_shape(mid))
    print("rq: ")
    rq = Permute((2, 1))(input_shapeQ)
    print(K.int_shape(rq))
    #a = K.batch_dot(mid,rq,axes=[2,1])
    a = Dot(axes=[2,2])([mid,input_shapeQ])
   # a = K.batch_dot(a,mid,axes=2)
    a = Activation('softmax')(a)

    ##rtt =Permute((2, 1))(input_shapeR)
    # x.shape = (batch_size, seq_len, time_steps)
    print("a: ")
    print(K.int_shape(a))


    outputs = Dot(axes=[1, 1])([input_shapeR, a])
    outputs = Permute((2, 1))(outputs)
    print("outputs: ")
    print(K.int_shape(outputs))
    return outputs
