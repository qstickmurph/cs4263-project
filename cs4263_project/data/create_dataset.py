"""
Module containing all functions for creating the tensorflow.data.Dataset from 
our pandas.DataFrame
"""

__all__ = [
    "window_df_to_ds",
    "variable_df_to_ds",
    "batched_variable_df_to_ds",
    "train_val_test_split",
]

from datetime import datetime
from dateutil.relativedelta import relativedelta
import math

import pandas as pd
import numpy as np
import tensorflow as tf

def window_df_to_ds(
    df, 
    features=[], 
    labels=[], 
    feature_width=7, 
    label_width=1, 
    label_dates=[]):
    """
    df : dataframe with a datetime index
    features : columns of df designated to be features
    labels : columns of df designmated to be labels
    feature_width : time-width of features in window
    label_width : time-width of labels in window
    label_dates : restrict the labels to come from this index
    """
    if label_dates is None:
        label_dates = df.index[feature_width:-(label_width)]

    def feature_gen():
        for label_start_date in label_dates:
            new_feature = df.loc[df.index.intersection(pd.date_range(label_start_date - relativedelta(days=feature_width), label_start_date - relativedelta(days=1), freq='d'))][features].values
            yield tf.convert_to_tensor(new_feature, dtype=np.float64)
    
    def label_gen():
        for label_start_date in label_dates:
            new_label = df.loc[df.index.intersection(pd.date_range(label_start_date, label_start_date + relativedelta(days=label_width - 1), freq='d'))][labels].values
            yield tf.convert_to_tensor(new_label, dtype=np.float64)

    # Turn np arrays into tf datasets
    feature_dataset = tf.data.Dataset.from_generator(feature_gen,
                                                     output_signature=(tf.TensorSpec(shape=(feature_width, len(features)), dtype=np.float64, name='Feature'))
                                                    )
    label_dataset = tf.data.Dataset.from_generator(label_gen,
                                                     output_signature=(tf.TensorSpec(shape=(label_width, len(labels)), dtype=np.float64, name='Label'))
                                                    )

    # Zip datasets together into feature, label pairs
    dataset = tf.data.Dataset.zip((feature_dataset, label_dataset))

    return dataset

def variable_df_to_ds(
    df, 
    features=[], 
    labels=[], 
    label_width=1, 
    label_dates=[]):
    """
    df : dataframe with a datetime index
    features : columns of df designated to be features
    labels : columns of df designmated to be labels
    label_width : time-width of labels in window
    label_dates : restrict the labels to come from this index
    """
    if label_dates is None:
        label_dates = df.index[1:-(label_width)]

    start_date = df.index[0]
    end_date = df.index[-1]

    # Create feature generator
    def feature_gen():
        for label_start_date in label_dates:
            new_feature = df.loc[df.index.intersection(pd.date_range(start_date, label_start_date - relativedelta(days=1), freq='d'))][features].values
            yield tf.convert_to_tensor(new_feature, dtype=np.float64)

    def label_gen():
        for label_start_date in label_dates:
            new_label = df.loc[df.index.intersection(pd.date_range(label_start_date, label_start_date + relativedelta(days=label_width - 1), freq='d'))][labels].values
            yield tf.convert_to_tensor(new_label, dtype=np.float64)

    
    # Turn np arrays into tf datasets
    feature_dataset = tf.data.Dataset.from_generator(feature_gen,
                                                     output_signature=(tf.TensorSpec(shape=(None, len(features)), dtype=np.float64, name='Feature'))
                                                    )
    label_dataset = tf.data.Dataset.from_generator(label_gen,
                                                     output_signature=(tf.TensorSpec(shape=(label_width, len(labels)), dtype=np.float64, name='Label'))
                                                    )

    # Zip datasets together into feature, label pairs
    dataset = tf.data.Dataset.zip((feature_dataset, label_dataset))

    return dataset

def batched_variable_df_to_ds(
    df, 
    features=[], 
    labels=[], 
    label_width=1, 
    label_dates=[], 
    batch_size=32):
    """
    df : dataframe with a datetime index
    features : columns of df designated to be features
    labels : columns of df designmated to be labels
    label_width : time-width of labels in window
    label_dates : restrict the labels to come from this index
    """
    if label_dates is None:
        label_dates = df.index[1:-(label_width)]

    start_date = df.index[0]
    end_date = df.index[-1]

    # Create feature generator
    def feature_gen():
        part_of_batch = 0
        for label_start_date in label_dates:
            new_feature = df.loc[df.index.intersection(pd.date_range(start_date + relativedelta(days=part_of_batch), label_start_date - relativedelta(days=1), freq='d'))][features].values
            yield tf.convert_to_tensor(new_feature, dtype=np.float64)
            part_of_batch = 0 if part_of_batch == 31 else part_of_batch+1

    def label_gen():
        for label_start_date in label_dates:
            new_label = df.loc[df.index.intersection(pd.date_range(label_start_date, label_start_date + relativedelta(days=label_width - 1), freq='d'))][labels].values
            yield tf.convert_to_tensor(new_label, dtype=np.float64)

    
    # Turn np arrays into tf datasets
    feature_dataset = tf.data.Dataset.from_generator(feature_gen,
                                                     output_signature=(tf.TensorSpec(shape=(None, len(features)), dtype=np.float64, name='Feature'))
                                                    )
    label_dataset = tf.data.Dataset.from_generator(label_gen,
                                                     output_signature=(tf.TensorSpec(shape=(label_width, len(labels)), dtype=np.float64, name='Label'))
                                                    )

    # Zip datasets together into feature, label pairs
    dataset = tf.data.Dataset.zip((feature_dataset, label_dataset))

    return dataset

def train_val_test_split(
    ds, 
    train_split=0.8, 
    val_split=0.1, 
    test_split=0.1, 
    shuffle=True, 
    batch_size=8, 
    repeats=1, 
    ds_size=None):
    """
    ds: tensorflow zip dataset
    train_split : 
    val_split :
    test_split :
    shuffle : whether to shuffle the ds or not
    """
    assert (train_split + val_split + test_split) == 1
    shuffle_size=10000

    # First batch
    ds = ds.batch(batch_size)

    # Then shuffle the batches
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=0)
    
    if ds_size is None:
        ds_size = len(ds)
    
    train_size = math.ceil(int(train_split * ds_size) / batch_size)
    val_size = math.ceil(int(val_split * ds_size) / batch_size)
    
    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds