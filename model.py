import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout,
    BatchNormalization, Layer
)
from tensorflow.keras import regularizers

# Custom Attention Layer
@register_keras_serializable()
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight", shape=(input_shape[-1], input_shape[-1]),
            initializer="he_normal", trainable=True
        )
        self.b = self.add_weight(
            name="att_bias", shape=(input_shape[-1],),
            initializer="zeros", trainable=True
        )
        self.V = self.add_weight(
            name="att_var", shape=(input_shape[-1], 1),
            initializer="he_normal", trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        score = K.tanh(K.dot(inputs, self.W) + self.b)
        attention_weights = K.softmax(K.dot(score, self.V), axis=1)
        context_vector = attention_weights * inputs
        return K.sum(context_vector, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# Build CNN-BiLSTM-Attention Model
def build_cnn_bilstm_attention_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # CNN Block 1
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = MaxPooling1D(pool_size=2)(x)

    # CNN Block 2
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same',
               kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = MaxPooling1D(pool_size=2)(x)

    # BiLSTM
    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(x)

    # Attention
    x = AttentionLayer()(x)

    # Output
    outputs = Dense(num_classes, activation='softmax',
                    kernel_regularizer=regularizers.l2(1e-4))(x)

    return Model(inputs=inputs, outputs=outputs)
