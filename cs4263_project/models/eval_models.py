__all__ = [
    "eval_hyperparams",
    "model_funct",
    "hyperparam_dims",
    "train_ds",
    "val_ds",
    "MAX_EPOCHS",
    "BATCH_SIZE",
    "best_loss",
    "search_num",
    "MAX_SEARCH"
]

import gc

from numpy import inf, nan
import tensorflow as tf
import skopt

model_funct = None
hyperparam_dims = []
train_ds = None
val_ds = None
INPUT_SHAPE = None
OUTPUT_SHAPE = None
MAX_EPOCHS = 25
BATCH_SIZE = 16
best_loss = inf
search_num = 1
MAX_SEARCH = 25

###
# THE FOLLOWING FUNCTION WAS BASED OFF OF https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/19_Hyper-Parameters.ipynb
###

@skopt.utils.use_named_args(dimensions=hyperparam_dims)
def eval_hyperparams(**kwargs):
    # Get global vars
    global model_funct
    
    global train_ds
    assert train_ds is not None, "train_ds is not set"
    assert isinstance(train_ds, tf.data.Dataset) , \
            "train_ds should be a tf dataset"

    global val_ds
    assert val_ds is not None, "val_ds is not set"
    assert isinstance(val_ds, tf.data.Dataset) , \
            "val_ds should be a tf dataset"

    global INPUT_SHAPE
    assert INPUT_SHAPE is not None, "INPUT SHAPE not set"

    global OUTPUT_SHAPE
    assert OUTPUT_SHAPE is not None, "OUTPUT SHAPE not set"

    global MAX_EPOCHS
    assert MAX_EPOCHS > 0, "MAX_EPOCHS should be > 0"
    assert isinstance(MAX_EPOCHS, int), "MAX_EPOCHS should be an int"

    global BATCH_SIZE
    assert BATCH_SIZE > 0, "BATCH_SIZE should be > 0"
    assert isinstance(BATCH_SIZE, int), "BATCH_SIZE should be an int"

    global best_loss
    assert best_loss > 0, "best_loss should be > 0"

    global search_num
    assert search_num > 0, "search num should be > 0"

    global MAX_SEARCH
    assert MAX_SEARCH > 0, "MAX_SEARCH should be > 0"

    print("Search Num", search_num, "/", MAX_SEARCH)
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # print kwargs
    for arg in kwargs.keys():
        print(arg, kwargs[arg])

    try:
        # Create the model
        model = model_funct(**kwargs)

        # Define early_stopping to not overrun
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min')

        # Train model
        history = model.fit(train_ds,
                            epochs=MAX_EPOCHS,
                            batch_size=BATCH_SIZE,
                            validation_data=val_ds,
                            callbacks=[early_stopping],
                            verbose=0)

        # If loss is better than best loss, set best_loss to loss (and save the model)
        loss = history.history['val_loss'][-1]

        if loss < best_loss:
            if 'file' in kwargs.keys():
                model.save(kwargs['file'])
            best_loss = loss

        del model
    except:
        pass

    if loss == inf or loss == nan:
        loss = 999999999999

    # Print loss
    print(f"\n|| val_loss: {loss} ||\n")

    # Clear the Keras session
    tf.keras.backend.clear_session()
    gc.collect()

    print()
    # Return loss for skopt to minimize
    return loss
    # Compile model
    