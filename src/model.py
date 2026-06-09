import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, LSTM, Lambda, Permute, Reshape, Multiply

class StockForecastingModels:
    @staticmethod
    def build_attention_lstm(input_shape):
        """
        Builds an Attention-Augmented Stacked LSTM.
        Fixed architectural calculation bottleneck from raw notebook cells.
        """
        inputs = Input(shape=input_shape)
        
        # Stacked LSTM Layers
        lstm1 = LSTM(64, return_sequences=True)(inputs)
        dropout1 = Dropout(0.2)(lstm1)
        
        # Temporal Attention Mechanism over sequence length steps
        attention_weights = Dense(1, activation='tanh')(dropout1)
        attention_weights = Flatten()(attention_weights)
        attention_weights = Activation('softmax')(attention_weights)
        attention_weights = RepeatVector(64)(attention_weights)
        attention_weights = Permute((2, 1))(attention_weights)
        
        context_vector = Multiply()([dropout1, attention_weights])
        
        lstm2 = LSTM(32, return_sequences=False)(context_vector)
        dropout2 = Dropout(0.2)(lstm2)
        
        dense1 = Dense(32, activation='relu')(dropout2)
        outputs = Dense(1)(dense1)
        
        return Model(inputs=inputs, outputs=outputs, name="Attention_LSTM_Model")

    @staticmethod
    def build_vanilla_lstm(input_shape):
        """Builds a classic baseline vanilla sequential LSTM model."""
        model = Sequential([
            LSTM(32, input_shape=input_shape, return_sequences=False),
            Dense(1)
        ], name="Vanilla_LSTM_Model")
        return model
