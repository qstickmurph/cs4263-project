from datetime import date, datetime
from tkinter import END
from tracemalloc import start
from dateutil.relativedelta import relativedelta
from matplotlib.pyplot import fill

import numpy as np
import pandas as pd
import tensorflow as tf

from cs4263_project.data import *

# Index Constants
START_DATE  = datetime(year=2004, month=1, day=1)
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
LABEL_WIDTH   = 0
TRAIN_SPLIT   = 0.8
VAL_SPLIT     = 0.1
TEST_SPLIT    = 0.1
BATCH_SIZE    = 16
REPEATS       = 1

### Hyperparam Search Constants
MAX_SEARCH = 50
MAX_EPOCHS = 25
INPUT_SHAPE = (None, len(FEATURES))
OUTPUT_SHAPE = (LABEL_WIDTH, len(LABELS))

STACKED_LSTM_HYPERPARAMS = True
STACKED_BILSTM_HYPERPARAMS = True
ENSEMBLE_STACKED_BILSTM_HYPERPARAMS = False

NUM_EPOCHS = 250

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

print(google_trends_df.isna().sum())

nymex_df_std = standardize_nymex(nymex_df)
google_trends_df_std = standardize_google_trends(google_trends_df)

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
elif BATCH_SIZE > 0:
    dataset = variable_df_to_ds(
        df=full_df.loc[FULL_INDEX], 
        features=FEATURES, 
        labels=LABELS, 
        label_width=LABEL_WIDTH, 
        label_dates=LABEL_INDEX)
else:
    dataset = batched_variable_df_to_ds(
        full_df.loc[FULL_INDEX], 
        features=FEATURES, 
        labels=LABELS, 
        label_width=LABEL_WIDTH, 
        label_dates=LABEL_INDEX, 
        batch_size=BATCH_SIZE)

train_ds, val_ds, test_ds = train_val_test_split(
        ds=dataset, 
        train_split=TRAIN_SPLIT, 
        val_split=VAL_SPLIT, 
        test_split=TEST_SPLIT, 
        batch_size=BATCH_SIZE, 
        repeats=REPEATS, 
        ds_size=len(LABEL_INDEX))

