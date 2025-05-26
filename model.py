import tensorflow as tf
from tensorflow.keras import backend as K                                                                                                                           # type: ignore
from tensorflow.keras.models import Model                                                                                                                           # type: ignore
from tensorflow.keras.utils import register_keras_serializable                                                                                                      # type: ignore
from tensorflow.keras.layers import (                                                                                                                               # type: ignore
    Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout,
    BatchNormalization, Layer
)
from tensorflow.keras import regularizers                                                                                                                           # type: ignore
from tensorflow.keras.layers import Concatenate                                                                                                                     # type: ignore

"""
model.py constructs the ensemble model for training and ulterior use for prediction
The model consists of a complex pipeline of CNNs, RNNs (BiLSTM) and attention heads/mechanisms
for more weight added to certain feautures

The model applies attention to emphasize important features, with 
a weighting mechanism that allows different emphasis on each input

The architecture promotes flexibility by processing two feature vectors 
separately before merging for classification.
"""

"""
The two defined classes act as building blocks for the attention mechanisms that will work to influence
importance of certain features inside of the two feature vectors
"""

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
    """
    Dual input model with CNNs, BiLSTM and attention

    Input parameters: input_shape_1, input_shape_2 (e.g. 93 and 14, respectively)
                    num_classes: the number of classes used for classification (e.g. 26 - 26 letters in the alphabet)

    Intermediary parameters: Conv1D(64, 3,...) or Conv1D(128, 3,...) - numbers of feature maps which allow each filter to learn a certain
                             pattern from the input data; each filter looks at 3 timesteps at once to observe patterns
                             
                             Dropout(0.3), Dropout(0.4) - 30% of neurons are set to zero in order to ensure robustness of the neural networks
                             and prevent overfitting
                             
                             LSTM(128, ..) specifies the number of memory cells used in the Long Short-Term Memory mechanism - 128 length vector
                             for each timestep
                             
                             Dense(256, ...) implements a fully connected layer of neurons; 256 is the number of neurons, standard choice; the 
                             rectifier (ReLU) ensures positive values are considered while negative ones are discarded; output layer with num_classes
                             neurons, softmax activation ensures probabilities with L2 regularization of factor 1*10^(-4), penalizing large weights
                             
    Returns: tf.keras.Model - used in train.py and saved in defined path
    """
    
    input_main = Input(shape=input_shape_1, name='main_input')  
    input_logic = Input(shape=input_shape_2, name='logic_input')

    # Path of the main (first) feature vector
    x1 = Conv1D(64, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4))(input_main)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)
    x1 = MaxPooling1D(pool_size=2)(x1)

    x1 = Conv1D(128, 3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Dropout(0.3)(x1)
    x1 = MaxPooling1D(pool_size=2)(x1)

    x1 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(x1)
    x1 = AttentionLayer()(x1)

    # Path of the second feature vector
    x2 = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.3))(input_logic)
    x2 = AttentionLayer()(x2) 

    # Merge the two inputs
    x = Concatenate()([x1, x2])  

    # Final layer using ReLU (rectifier) activation function together with softmax
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(1e-4))(x)

    return Model(inputs=[input_main, input_logic], outputs=outputs)


