"""
Module containing all preprocessing functions for NYMEX and Google Trends data (as a dataframe)
"""

__all__ = [
    "standardize_nymex",
    "restore_nymex",
    "standardize_google_trends",
    "restore_google_trends",
]

from sqlite3 import DatabaseError
import numpy as np
import pandas as pd

means = {}
stds = {}


def standardize_nymex(nymex_df : pd.DataFrame) -> pd.DataFrame:
    assert (nymex_df <= 0).sum().sum() == 0
    global means
    global stds

    nymex_df_standardized = np.log(nymex_df)
    for column in nymex_df_standardized.columns:
        means[column] = nymex_df_standardized[column].mean()
        stds[column] = nymex_df_standardized[column].std()
        nymex_df_standardized[column] = (nymex_df_standardized[column] 
                                        - means[column]) / stds[column]
    
    return nymex_df_standardized


def restore_nymex(nymex_df_standardized : pd.DataFrame) -> pd.DataFrame:
    global means
    global stds

    for column in nymex_df_standardized:
        nymex_df_standardized[column] = nymex_df_standardized[column] \
                                        * stds[column] \
                                            + means[column]
    nymex_df = np.exp(nymex_df_standardized)
    return nymex_df

def standardize_google_trends(google_trends_df : pd.DataFrame) -> pd.DataFrame:
    global means
    global stds

    google_trends_df_standardized = google_trends_df.copy()
    for column in google_trends_df_standardized.columns:
        google_trends_df_standardized[column] = \
                (google_trends_df_standardized[column] 
                - google_trends_df_standardized[column].mean()) \
                    / google_trends_df_standardized[column].std()

    google_trends_df_standardized

def restore_google_trends(google_trends_df_standardized) -> pd.DataFrame:
    global means
    global stds

    for column in google_trends_df_standardized:
        google_trends_df_standardized[column] = \
                google_trends_df_standardized[column]*stds[column] \
                + means[column]
    return google_trends_df_standardized 