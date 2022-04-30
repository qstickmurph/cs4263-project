from cgi import test
from datetime import date, datetime
from tabnanny import verbose
from dateutil.relativedelta import relativedelta
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt

from cs4263_project.models import *
from cs4263_project.data import *

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Index Constants
START_DATE  = datetime(year=2004, month=1, day=5)
END_DATE    = datetime(year=2019, month=6, day=28)
FULL_INDEX  = pd.date_range(START_DATE, END_DATE, freq='d')

LABEL_START_DATE = datetime(year=2013, month=1, day=1)
LABEL_END_DATE   = datetime(year=2019, month=6, day=28)
LABEL_INDEX = pd.date_range(LABEL_START_DATE, LABEL_END_DATE, freq='d')

# Columns Constants
FEATURES = [
    'Spot Price', 'Futures 1 Price', 'Futures 2 Price', 'Futures 3 Price',
    'Futures 4 Price', 'Natural Gas', 'Oil', 'Coal', 'Nuclear Power',
    'Wind Power', 'Hydroelectric', 'Solar Power', 'Gold', 'Silver',
    'Platinum', 'Copper', 'Biofuel', 'Recession', 'CPI'
]
LABELS = [
    'Futures 1 Price','Futures 2 Price','Futures 3 Price','Futures 4 Price'
]

# Dataset Constants
FEATURE_WIDTH = 0
LABEL_WIDTH   = 1
TRAIN_SPLIT   = 0.8
VAL_SPLIT     = 0.1
TEST_SPLIT    = 0.1
BATCH_SIZE    = 16
REPEATS       = 1

### Hyperparam Search Constants
MAX_SEARCH = 50
EPOCHS_PER_SEARCH = 25

SEARCH_STACKED_LSTM_HYPERPARAMS = True
SEARCH_STACKED_BILSTM_HYPERPARAMS = True
TRAIN_STACKED_LSTM = True
TRAIN_STACKED_BILSTM = True
TEST_STACKED_LSTM = True
TEST_STACKED_BILSTM = True

FINAL_TRAINING_EPOCHS = 250

# Fetch Data

nymex_df = read_nymex(
    file="data/US_EIA_NYMEX.csv",
    start_date=START_DATE,
    end_date=END_DATE,
    interpolate=True,
    fill_missing_dates=True)

google_trends_df = read_google_trends(
    file="data/google_trends_dataset.csv",
    keywords=["Natural Gas","Oil","Coal","Nuclear Power","Wind Power",
              "Hydroelectric","Solar Power","Gold","Silver","Platinum","Copper",
              "Biofuel","Recession","CPI"],
    categories=[0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    start_date=START_DATE,
    end_date=END_DATE)

# Standardize Data
print("Standardize datasets")
nymex_df_std = standardize_nymex(nymex_df)
google_trends_df_std = standardize_google_trends(google_trends_df)

# Plot Data
fig = plt.figure(figsize=(24,12))
## Nymex
print("Plotting NYMEX data")
i = 0
for column in nymex_df.columns:
    if i == 0:
        plot(fig, nymex_df[[column]], units="(Dollars per Million Btu)", density=1, 
            file="images/nymex_data_" 
                + column.replace(" ","_") + ".png")
        plot(fig, nymex_df[[column]], units="(Dollars per Million Btu)", density=30, 
            file="images/nymex_data_monthly_" 
                + column.replace(" ","_") + ".png")
    else:
        plot(fig, nymex_df[[column]], units="(Dollars per Million Btu)", density=1, 
            file="images/nymex_data_" 
                + column.replace(" ","_") + ".png", 
            labels=[column], 
            label_dates=LABEL_INDEX)
        plot(fig, nymex_df[[column]], units="(Dollars per Million Btu)", density=1, 
            file="images/nymex_data_monthly_" 
                + column.replace(" ","_") + ".png", 
            labels=[column], 
            label_dates=LABEL_INDEX)
    i+=1

plot(fig, 
    nymex_df,
    units="(Dollars per Million Btu)", 
    seperate=True, 
    density=1, 
    file="images/nymex_data.png", 
    labels=nymex_df.columns[1:], 
    label_dates=LABEL_INDEX)

## Nymex Standardized
print("Plotting standardized NYMEX data")
i = 0
for column in nymex_df_std.columns:
    if i == 0:
        plot(fig, nymex_df_std[[column]], units="(Dollars per Million Btu)", density=1, 
            file="images/nymex_data_" 
                + column.replace(" ","_") + ".png")
        plot(fig, nymex_df_std[[column]], units="(Dollars per Million Btu)", density=30, 
            file="images/nymex_data_monthly_" 
                + column.replace(" ","_") + ".png")
    else:
        plot(fig, nymex_df_std[[column]], units="(Dollars per Million Btu)", density=1, 
            file="images/nymex_data_" 
                + column.replace(" ","_") + ".png", 
            labels=[column], 
            label_dates=LABEL_INDEX)
        plot(fig, nymex_df_std[[column]], units="(Dollars per Million Btu)", density=1, 
            file="images/nymex_data_monthly_" 
                + column.replace(" ","_") + ".png", 
            labels=[column], 
            label_dates=LABEL_INDEX)
    i+=1

plot(fig, 
    nymex_df_std,
    units="(Dollars per Million Btu)", 
    seperate=True, 
    density=1, 
    file="images/nymex_data.png", 
    labels=nymex_df_std.columns[1:], 
    label_dates=LABEL_INDEX)

## Google Trends
print("Plotting Google Trends data")
for column in google_trends_df.columns:
    plot(fig, google_trends_df[[column]], units="Search Volume", density=1, 
            file="images/google_trends_data_" + column.replace(" ","_") 
                    + ".png")
    plot(fig, google_trends_df[[column]], units="Search Volume", density=30, 
            file="images/google_trends_data_monthly_" + column.replace(" ","_") 
                    + ".png")
plot(fig, google_trends_df,units="Search Volume", seperate=True, density=1, 
        file="images/google_trends_data.png")
plot(fig, google_trends_df,units="Search Volume", seperate=True, density=30, 
        file="images/google_trends_data_monthly.png")

print("Plotting standardized Google Trends data")
## Google Trends Standardized
for column in google_trends_df_std.columns:
    plot(fig, google_trends_df_std[[column]], units="Search Volume", density=1, file="images/google_trends_data_" + column.replace(" ","_") + "_standardized.png")
    plot(fig, google_trends_df_std[[column]], units="Search Volume", density=30, file="images/google_trends_data_monthly_" + column.replace(" ","_") + "_standardized.png")
plot(fig, google_trends_df_std,units="Search Volume", seperate=True, density=1, file="images/google_trends_data_standardized.png")
plot(fig, google_trends_df_std,units="Search Volume", seperate=True, density=30, file="images/google_trends_data_monthly_standardized.png")

print("Creating tf Dataset")

full_df = pd.concat([nymex_df_std, google_trends_df_std],axis=1).loc[
    pd.date_range(START_DATE, END_DATE, freq='d')
]

if FEATURE_WIDTH > 0:
    dataset = window_df_to_ds(
        df=full_df.loc[FULL_INDEX], 
        features=FEATURES, 
        labels=LABELS, 
        feature_width=FEATURE_WIDTH, 
        label_width=LABEL_WIDTH, 
        label_dates=LABEL_INDEX)
    create_models.INPUT_SHAPE = (FEATURE_WIDTH, len(FEATURES))
elif BATCH_SIZE > 1:
    dataset = batched_variable_df_to_ds(
        full_df.loc[FULL_INDEX], 
        features=FEATURES, 
        labels=LABELS, 
        label_width=LABEL_WIDTH, 
        label_dates=LABEL_INDEX, 
        batch_size=BATCH_SIZE)
    create_models.INPUT_SHAPE = (None, len(FEATURES))
else:
    dataset = variable_df_to_ds(
        df=full_df.loc[FULL_INDEX], 
        features=FEATURES, 
        labels=LABELS, 
        label_width=LABEL_WIDTH, 
        label_dates=LABEL_INDEX)
    create_models.INPUT_SHAPE = (None, len(FEATURES))

create_models.OUTPUT_SHAPE= (LABEL_WIDTH, len(LABELS))

train_ds, val_ds, test_ds = train_val_test_split(
        ds=dataset, 
        train_split=TRAIN_SPLIT, 
        val_split=VAL_SPLIT, 
        test_split=TEST_SPLIT, 
        batch_size=BATCH_SIZE, 
        repeats=REPEATS, 
        ds_size=len(LABEL_INDEX))

batches_per_epoch = math.ceil(int(TRAIN_SPLIT* len(LABEL_INDEX)) / BATCH_SIZE)

stacked_lstm_tuner = kt.BayesianOptimization(
    create_stacked_lstm_hp,
    objective='loss',
    max_trials=MAX_SEARCH,
    directory='models/hyperparam_search',
    project_name='stacked_lstm',
    overwrite=False
)

if SEARCH_STACKED_LSTM_HYPERPARAMS:
    print()
    print("Searching Stacked LSTM Hyperparameters")
    print()
    stacked_lstm_tuner.search(
        train_ds.repeat(EPOCHS_PER_SEARCH),
        steps_per_epoch=batches_per_epoch,
        epochs=EPOCHS_PER_SEARCH,
        use_multiprocessing=True
    )
    stacked_lstm_tuner.results_summary(num_trials=1)

best_stacked_lstm_hps = stacked_lstm_tuner.get_best_hyperparameters(num_trials=1)[0]

stacked_bilstm_tuner = kt.BayesianOptimization(
    create_stacked_bilstm_hp,
    objective='loss',
    max_trials=MAX_SEARCH,
    directory='models/hyperparam_search',
    project_name='stacked_bilstm',
    overwrite=False
)
if SEARCH_STACKED_BILSTM_HYPERPARAMS:
    print()
    print("Searching Stacked BiLSTM Hyperparameters")
    print()



    stacked_lstm_tuner.search(
        train_ds.repeat(EPOCHS_PER_SEARCH),
        steps_per_epoch=batches_per_epoch,
        epochs=EPOCHS_PER_SEARCH,
        use_multiprocessing=True
    )
    stacked_bilstm_tuner.results_summary(num_trials=1)

best_stacked_bilstm_hps = stacked_lstm_tuner.get_best_hyperparameters(num_trials=1)[0]

if TRAIN_STACKED_LSTM:
    print()
    print("Training best stacked LSTM model")
    print()

    stacked_lstm = stacked_lstm_tuner.hypermodel.build(best_stacked_lstm_hps)

    save_best = tf.keras.callbacks.ModelCheckpoint(
            "models/trained/stacked_lstm",
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            mode='min'
    )

    # Train model
    history = stacked_lstm.fit(train_ds,
                                initial_epoch=0,
                                epochs=FINAL_TRAINING_EPOCHS,
                                batch_size=BATCH_SIZE,
                                validation_data=val_ds,
                                callbacks=[save_best],
                                verbose=1)

    loss_values = history.history['loss']
    val_loss_values = history.history['val_loss']
    epochs = range(1, len(loss_values)+1)
    plt.plot(epochs, loss_values, label='Training Loss')
    plt.plot(epochs, val_loss_values, label='Validation Loss')
    plt.gca().set_yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig("images/stacked_lstm_loss_over_epoch")

if TRAIN_STACKED_BILSTM:
    print()
    print("Training best stacked BiLSTM model")
    print()

    stacked_bilstm = stacked_bilstm_tuner.hypermodel.build(best_stacked_bilstm_hps)

    save_best = tf.keras.callbacks.ModelCheckpoint(
            "models/trained/stacked_bilstm",
            monitor='val_loss',
            verbose=0,
            save_best_only=True,
            mode='min'
    )

    # Train model
    history = stacked_bilstm.fit(train_ds,
                                initial_epoch=0,
                                epochs=FINAL_TRAINING_EPOCHS,
                                batch_size=BATCH_SIZE,
                                validation_data=val_ds,
                                callbacks=[save_best],
                                verbose=1)

    loss_values = history.history['loss']
    val_loss_values = history.history['val_loss']
    epochs = range(1, len(loss_values)+1)
    plt.plot(epochs, loss_values, label='Training Loss')
    plt.plot(epochs, val_loss_values, label='Validation Loss')
    plt.gca().set_yscale('log')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig("images/stacked_bilstm_loss_over_epoch")

test_results = {}

if TEST_STACKED_LSTM:
    print()
    print("Testing Stacked LSTM")

    stacked_lstm = tf.keras.models.load_model("models/trained/stacked_lstm")


    test_results["stacked_lstm"] = dict(zip(stacked_lstm.metrics_names, stacked_lstm.evaluate(test_ds, verbose=0)))

    print("Printing Stacked LSTM Predictions")
    predictions_df = get_predictions_df(model=stacked_lstm, dataset=dataset, label_width=LABEL_WIDTH, labels=LABELS, index=LABEL_DATES)
    for label in LABELS:
        plot(nymex_df[[label]], units="$$$", label_width=LABEL_WIDTH, predictions=predictions_df[label], density=30, seperate=True, file=TOP_FOLDER_NAME+"images/stacked_lstm_predictions_" + label.replace(" ", "_") + "_sparse.png")
        plot(nymex_df[[label]], units="$$$", label_width=LABEL_WIDTH, predictions=predictions_df[label], density=1, seperate=True, file=TOP_FOLDER_NAME+"images/stacked_lstm_predictions_" + label.replace(" ", "_") + "_dense.png")

    plot(nymex_df[LABELS], units="$$$", label_width=LABEL_WIDTH, predictions=predictions_df, density=30, seperate=True, file=TOP_FOLDER_NAME+"images/stacked_lstm_predictions_sparse.png")
    plot(nymex_df[LABELS], units="$$$", label_width=LABEL_WIDTH, predictions=predictions_df, density=1, seperate=True, file=TOP_FOLDER_NAME+"images/stacked_lstm_predictions_dense.png")

tf.keras.backend.clear_session()

if TEST_STACKED_BILSTM:
    print()
    print("Testing Stacked BILSTM")

    stacked_bilstm = tf.keras.models.load_model("models/trained/stacked_bilstm")


    test_results["stacked_bilstm"] = dict(zip(stacked_bilstm.metrics_names, stacked_bilstm.evaluate(test_ds, verbose=0)))

    print("Printing Stacked BILSTM Predictions")
    predictions_df = get_predictions_df(model=stacked_bilstm, dataset=dataset, label_width=LABEL_WIDTH, labels=LABELS, index=LABEL_DATES)
    for label in LABELS:
        plot(nymex_df[[label]], units="$$$", label_width=LABEL_WIDTH, predictions=predictions_df[label], density=30, seperate=True, file=TOP_FOLDER_NAME+"images/stacked_bilstm_predictions_" + label.replace(" ", "_") + "_sparse.png")
        plot(nymex_df[[label]], units="$$$", label_width=LABEL_WIDTH, predictions=predictions_df[label], density=1, seperate=True, file=TOP_FOLDER_NAME+"images/stacked_bilstm_predictions_" + label.replace(" ", "_") + "_dense.png")
        
    plot(nymex_df[LABELS], units="$$$", label_width=LABEL_WIDTH, predictions=predictions_df, density=30, seperate=True, file=TOP_FOLDER_NAME+"images/stacked_bilstm_predictions_sparse.png")
    plot(nymex_df[LABELS], units="$$$", label_width=LABEL_WIDTH, predictions=predictions_df, density=1, seperate=True, file=TOP_FOLDER_NAME+"images/stacked_bilstm_predictions_dense.png")
