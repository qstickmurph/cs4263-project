__all__ = [
    "StackedLSTM",
]

import tensorflow as tf

from cs4263_project.models import BaseModel

class StackedLSTM(BaseModel):

    def __init__(self, 
            num_lstm_layers=5, 
            num_lstm_nodes=32, 
            num_dense_layers=3, 
            num_dense_nodes=256, 
            dropout_rate=0, 
            input_shape=(None, 19), 
            output_shape=(1, 4), 
            learning_rate=1e-3, 
            beta_1=0.9):
        super(StackedLSTM, self)
        self.add(tf.keras.layers.Input(shape=input_shape, name="Input"))

        # Add stacked LSTM layers
        for i in range(num_lstm_layers):
            if i < num_lstm_layers-1:
                self.add(tf.keras.layers.LSTM(units=num_lstm_nodes, input_shape=input_shape, return_sequences=True, name=f'LSTM_{i}'))
            else:
                self.add(tf.keras.layers.LSTM(units=num_lstm_nodes, input_shape=input_shape, return_sequences=False, name=f'LSTM_{i}'))

        # Add stacked dense layers
        for i in range(num_dense_layers):
            # Add dropout layer
            self.add(tf.keras.layers.Dropout(rate=dropout_rate, name=f"Dropout_{i}"))
            # Add dense layer
            self.add(tf.keras.layers.Dense(units=num_dense_nodes, name=f"Dense_{i}"))

        # Add output layer + reshape
        self.add(tf.keras.layers.Dense(units=output_shape[0] * output_shape[1]))
        self.add(tf.keras.layers.Reshape(target_shape=output_shape))

        # Compile self
        self.compile(loss=tf.losses.MeanSquaredError(),
                    optimizer=tf.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1),
                    metrics=[tf.metrics.MeanAbsoluteError(),tf.metrics.RootMeanSquaredError()])
