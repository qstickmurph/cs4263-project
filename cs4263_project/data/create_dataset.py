"""
Module containing all functions for creating the tensorflow.data.Dataset from our pandas.DataFrame
"""

__all__ = [
    "window_df_to_ds",
    "variable_df_to_ds",
    "batched_variable_df_to_ds",
]

import pandas as pd
import tensorflow as tf

def window_df_to_ds(df, features=[], labels=[], feature_width=7, label_width=1, label_dates=[]):
    pass

def variable_df_to_ds(df, features=[], labels=[], label_width=1, label_dates=[]):
    pass

def batched_variable_df_to_ds(df, features=[], labels=[], label_width=1, label_dates=[], batch_size=32):
    pass

def train_val_test_split(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, batch_size=8, repeats=1, ds_size=None):
    pass