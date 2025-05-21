import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout,
    BatchNormalization, Layer
)
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Concatenate

@register_keras_serializable()
class WeightedAttentionLayer(Layer):
    def __init__(self, weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.weight = weight

    def build(self, input_shape):
        # Similar to regular AttentionLayer, but we scale based on the weight
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]),
                                 initializer="he_normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],),
                                 initializer="zeros", trainable=True)
        self.V = self.add_weight(name="att_var", shape=(input_shape[-1], 1),
                                 initializer="he_normal", trainable=True)
        self.gamma = self.add_weight(name="att_gamma", shape=(1,),
                                     initializer=tf.initializers.constant(self.weight), trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        score = K.tanh(K.dot(inputs, self.W) + self.b)
        attention_weights = K.softmax(K.dot(score, self.V), axis=1)
        context_vector = attention_weights * inputs
        # Apply the learned gamma scaling factor to adjust attention strength
        return K.sum(context_vector, axis=1) * self.gamma

# Custom Attention Layer
@register_keras_serializable()
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], input_shape[-1]),
                                 initializer="he_normal", trainable=True)
        self.b = self.add_weight(name="att_bias", shape=(input_shape[-1],),
                                 initializer="zeros", trainable=True)
        self.V = self.add_weight(name="att_var", shape=(input_shape[-1], 1),
                                 initializer="he_normal", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        score = K.tanh(K.dot(inputs, self.W) + self.b)
        attention_weights = K.softmax(K.dot(score, self.V), axis=1)
        context_vector = attention_weights * inputs
        return K.sum(context_vector, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def build_dual_input_model(input_shape_1, input_shape_2, num_classes):
    input_main = Input(shape=input_shape_1, name='main_input')   # e.g. (20, 75)
    input_logic = Input(shape=input_shape_2, name='logic_input') # e.g. (20, 5)

    # CNN path
    x1 = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4))(input_main)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)
    x1 = MaxPooling1D(pool_size=2)(x1)

    x1 = Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)
    x1 = MaxPooling1D(pool_size=2)(x1)

    x1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(x1)
    x1 = AttentionLayer()(x1)  # Now shape is (None, 256)

    # Logic path
    x2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(input_logic)
    x2 = AttentionLayer()(x2)  # Shape (None, 256)

    # Merge
    x = Concatenate()([x1, x2])  # Now valid: both (None, 256) → concat → (None, 512)

    # Final processing
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(1e-4))(x)

    return Model(inputs=[input_main, input_logic], outputs=outputs)


