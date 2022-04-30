__all__ = [
    "create_stacked_lstm",
    "create_stacked_lstm_hp",
    "create_stacked_bilstm",
    "create_stacked_bilstm_hp",
#    "create_ensemble_stacked_bilstm",
    "INPUT_SHAPE",
    "OUTPUT_SHAPE",
]

import tensorflow as tf

INPUT_SHAPE = None
OUTPUT_SHAPE = None

def create_stacked_lstm(
    num_lstm_layers=5, 
    num_lstm_nodes=32, 
    num_dense_layers=3, 
    num_dense_nodes=256, 
    dropout_rate=0, 
    input_shape=None, 
    output_shape=None, 
    learning_rate=1e-3, 
    beta_1=.9):
    # Init model
    model = tf.keras.Sequential()

    # Add input layer
    model.add(tf.keras.layers.Input(shape=input_shape, name="Input"))

    # Add stacked LSTM layers
    for i in range(num_lstm_layers):
        if i < num_lstm_layers-1:
            model.add(tf.keras.layers.LSTM(units=num_lstm_nodes, input_shape=input_shape, return_sequences=True, name=f'LSTM_{i}'))
        else:
            model.add(tf.keras.layers.LSTM(units=num_lstm_nodes, input_shape=input_shape, return_sequences=False, name=f'LSTM_{i}'))

    # Add stacked dense layers
    for i in range(num_dense_layers):
        # Add dropout layer
        model.add(tf.keras.layers.Dropout(rate=dropout_rate, name=f"Dropout_{i}"))
        # Add dense layer
        model.add(tf.keras.layers.Dense(units=num_dense_nodes, name=f"Dense_{i}"))

    # Add output layer + reshape
    model.add(tf.keras.layers.Dense(units=output_shape[0] * output_shape[1]))
    model.add(tf.keras.layers.Reshape(target_shape=output_shape))

    # Compile model
    model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1),
                metrics=[tf.metrics.MeanAbsoluteError(),tf.metrics.RootMeanSquaredError()])

    # Return model (pre-compiled)
    return model


def create_stacked_lstm_hp(hp):
    # Init model
    model = tf.keras.Sequential()

    # Add input layer
    model.add(tf.keras.layers.Input(shape=INPUT_SHAPE, name="Input"))

    num_lstm_layers = hp.Int('num_lstm_layers', min_value=1, max_value=5, step=1)
    num_lstm_nodes = hp.Int('num_lstm_nodes', min_value=32, max_value=256, step=32)
    # Add stacked LSTM layers
    for i in range(num_lstm_layers):
        if i < num_lstm_layers-1:
            model.add(tf.keras.layers.LSTM(units=num_lstm_nodes, input_shape=INPUT_SHAPE, return_sequences=True, name=f'LSTM_{i}'))
        else:
            model.add(tf.keras.layers.LSTM(units=num_lstm_nodes, input_shape=INPUT_SHAPE, return_sequences=False, name=f'LSTM_{i}'))

    num_dense_layers = hp.Int('num_dense_layers', min_value=1, max_value=3, step=1)
    num_dense_nodes = hp.Int('num_dense_nodes', min_value=256, max_value=2048, step=256)
    dropout_rate = hp.Float('dropout_rate', min_value=0, max_value=1, step=0.1)
    # Add stacked dense layers
    for i in range(num_dense_layers):
        # Add dropout layer
        model.add(tf.keras.layers.Dropout(rate=dropout_rate, name=f"Dropout_{i}"))
        # Add dense layer
        model.add(tf.keras.layers.Dense(units=num_dense_nodes, name=f"Dense_{i}"))

    # Add output layer + reshape
    model.add(tf.keras.layers.Dense(units=OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1]))
    model.add(tf.keras.layers.Reshape(target_shape=OUTPUT_SHAPE))

    learning_rate = hp.Float('learning_rate', min_value=1e-6, max_value=1e-1, sampling='log')
    beta_1 = hp.Float('beta_1', min_value=0.5, max_value=1, sampling='linear')
    # Compile model
    model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1),
                metrics=[tf.metrics.MeanAbsoluteError(),tf.metrics.RootMeanSquaredError()])

    # Return model (pre-compiled)
    return model


def create_stacked_bilstm(
    num_lstm_layers=5, 
    num_lstm_nodes=32, 
    num_dense_layers=3, 
    num_dense_nodes=256, 
    dropout_rate=0, 
    input_shape=None, 
    output_shape=None, 
    learning_rate=1e-3, 
    beta_1=.9):
    # Init model
    model = tf.keras.Sequential()

    # Add input layer
    model.add(tf.keras.layers.Input(shape=input_shape, name="Input"))

    # Add stacked LSTM layers
    for i in range(num_lstm_layers):
        if i < num_lstm_layers-1:
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=num_lstm_nodes, input_shape=input_shape, return_sequences=True, name=f'LSTM_{i}')))
        else:
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=num_lstm_nodes, input_shape=input_shape, return_sequences=False, name=f'LSTM_{i}')))

    # Add stacked dense layers
    for i in range(num_dense_layers):
        # Add dropout layer
        model.add(tf.keras.layers.Dropout(rate=dropout_rate, name=f"Dropout_{i}"))
        # Add dense layer
        model.add(tf.keras.layers.Dense(units=num_dense_nodes, name=f"Dense_{i}"))

    # Add output layer + reshape
    model.add(tf.keras.layers.Dense(units=output_shape[0] * output_shape[1]))
    model.add(tf.keras.layers.Reshape(target_shape=output_shape))

    # Compile model
    model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1),
                metrics=[tf.metrics.MeanAbsoluteError(),tf.metrics.RootMeanSquaredError()])

    # Return model (uncompiled)
    return model


def create_stacked_bilstm_hp(hp):
    # Init model
    model = tf.keras.Sequential()

    # Add input layer
    model.add(tf.keras.layers.Input(shape=INPUT_SHAPE, name="Input"))

    num_lstm_layers = hp.Int('num_lstm_layers', min_value=1, max_value=5, step=1)
    num_lstm_nodes = hp.Int('num_lstm_nodes', min_value=32, max_value=128, step=32)
    # Add stacked LSTM layers
    for i in range(num_lstm_layers):
        if i < num_lstm_layers-1:
            model.add(tf.keras.Bidirectional(tf.keras.layers.LSTM(units=num_lstm_nodes, input_shape=INPUT_SHAPE, return_sequences=True, name=f'LSTM_{i}')))
        else:
            model.add(tf.keras.Bidirectional(tf.keras.layers.LSTM(units=num_lstm_nodes, input_shape=INPUT_SHAPE, return_sequences=False, name=f'LSTM_{i}')))

    num_dense_layers = hp.Int('num_dense_layers', min_value=1, max_value=3, step=1)
    num_dense_nodes = hp.Int('num_dense_nodes', min_value=256, max_value=2048, step=256)
    dropout_rate = hp.Float('dropout_rate', min_value=0, max_value=1, step=0.1)
    # Add stacked dense layers
    for i in range(num_dense_layers):
        # Add dropout layer
        model.add(tf.keras.layers.Dropout(rate=dropout_rate, name=f"Dropout_{i}"))
        # Add dense layer
        model.add(tf.keras.layers.Dense(units=num_dense_nodes, name=f"Dense_{i}"))

    # Add output layer + reshape
    model.add(tf.keras.layers.Dense(units=OUTPUT_SHAPE[0] * OUTPUT_SHAPE[1]))
    model.add(tf.keras.layers.Reshape(target_shape=OUTPUT_SHAPE))

    learning_rate = hp.Float('learning_rate', min_value=1e-6, max_value=1e-1, sampling='log')
    beta_1 = hp.Float('beta_1', min_value=0.5, max_value=1, sampling='uniform')
    # Compile model
    model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1),
                metrics=[tf.metrics.MeanAbsoluteError(),tf.metrics.RootMeanSquaredError()])

    # Return model (pre-compiled)
    return model