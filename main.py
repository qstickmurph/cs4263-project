from datetime import date, datetime
from tkinter import END
from tracemalloc import start
from dateutil.relativedelta import relativedelta
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from cs4263_project.data import *

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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

