# Attention GRU network
from keras.engine.topology import Layer
from keras import backend as K

class AttentionLayer(Layer):
    def __init__(self, **kwargs):

        super(AttentionLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # inputs.shape = (batch_size, time_steps, seq_len)
        # W.shape = (time_steps, time_steps)
        input_shapeR, input_shapeQ = input_shape
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shapeR[2], input_shapeQ[2]),
                                 initializer='uniform',
                                 trainable=True)

        super(AttentionLayer, self).build(input_shape)

    def call(self, input_shape):
        assert isinstance(input_shape, list)

        input_shapeR, input_shapeQ = input_shape


        # inputs.shape = (batch_size, time_steps, seq_len)
        rt = K.permute_dimensions(input_shapeR, (1, 0, 2))
        qt = K.permute_dimensions(input_shapeQ, (2, 0, 1))
        print (K.int_shape(rt))
        print (K.int_shape(qt))
        # x.shape = (batch_size, seq_len, time_steps)
        mid=K.dot(rt, self.W)
        a = K.softmax(K.dot(mid, qt))
        print(K.int_shape(a))
        rtt = K.permute_dimensions(input_shapeR, (0, 2, 1))
        print(K.int_shape(rtt))
        outputs = K.dot(rtt, a)
        outputs = K.permute_dimensions(outputs, (0, 2, 1))
        return outputs

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        input_shapeR, input_shapeQ = input_shape
        return [input_shapeQ[0], input_shapeQ[1],input_shapeR[2]]
