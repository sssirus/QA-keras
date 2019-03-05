# Attention GRU network
from keras.engine import Layer
from keras import backend as K

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(** kwargs)

    def build(self, input_shapeR,input_shapeQ):
        assert len(input_shapeR) == 3
        assert len(input_shapeQ) == 3
        # inputs.shape = (batch_size, time_steps, seq_len)
        # W.shape = (time_steps, time_steps)
        self.W = self.add_weight(name='att_weight',
                                 shape=(input_shapeR[2], input_shapeQ[2]),
                                 initializer='uniform',
                                 trainable=True)

        super(AttentionLayer, self).build(input_shapeR)
        super(AttentionLayer, self).build(input_shapeQ)
    def call(self, input_shapeR,input_shapeQ):
        # inputs.shape = (batch_size, time_steps, seq_len)
        rt = K.permute_dimensions(input_shapeR, (1, 0, 2))
        qt = K.permute_dimensions(input_shapeQ, (2, 0, 1))
        # x.shape = (batch_size, seq_len, time_steps)
        a = K.softmax(K.dot(K.dot(rt, self.W),qt))
        rtt=K.permute_dimensions(input_shapeR, (0, 2, 1))
        outputs = K.dot(rtt,a)
        outputs = K.permute_dimensions(outputs, (0, 2, 1))

        return outputs

    def compute_output_shape(self, input_shapeR,input_shapeQ):
        return input_shapeQ[0], input_shapeQ[1],input_shapeR[2]
