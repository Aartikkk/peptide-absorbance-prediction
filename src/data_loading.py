"""
Data loading and preprocessing utilities for peptide absorbance data.

Handles loading of trial CSV files, computing mean absorbance targets,
and preparing train/test splits.
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple


def load_trial_data(
    trial1_path: str,
    trial2_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load two trial CSV files.

    Args:
        trial1_path: Path to Trial 1 CSV file.
        trial2_path: Path to Trial 2 CSV file.

    Returns:
        Tuple of (df_trial1, df_trial2).
    """
    df1 = pd.read_csv(trial1_path)
    df2 = pd.read_csv(trial2_path)
    return df1, df2


def get_absorbance_columns(df: pd.DataFrame) -> list:
    """Return column names that contain absorbance readings.

    Args:
        df: DataFrame with absorbance columns.

    Returns:
        List of column names containing 'Absorbance'.
    """
    return [col for col in df.columns if 'Absorbance' in col]


def compute_mean_absorbance(df: pd.DataFrame) -> pd.Series:
    """Compute the mean absorbance across all absorbance columns per row.

    Args:
        df: DataFrame with absorbance columns.

    Returns:
        Series of mean absorbance values.
    """
    abs_cols = get_absorbance_columns(df)
    return df[abs_cols].astype(float).mean(axis=1)


def average_trials(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
) -> pd.DataFrame:
    """Average absorbance values across two trials.

    Creates a combined DataFrame with averaged absorbance values
    and summary statistics (mean, std, min, max, CV).

    Args:
        df1: Trial 1 DataFrame.
        df2: Trial 2 DataFrame.

    Returns:
        Combined DataFrame with averaged absorbance and statistics.
    """
    abs_cols = get_absorbance_columns(df1)
    df_avg = df1.copy()

    for col in abs_cols:
        df_avg[col] = (df1[col].astype(float) + df2[col].astype(float)) / 2

    df_avg['mean_abs'] = df_avg[abs_cols].mean(axis=1)
    df_avg['std_abs'] = df_avg[abs_cols].std(axis=1)
    df_avg['max_abs'] = df_avg[abs_cols].max(axis=1)
    df_avg['min_abs'] = df_avg[abs_cols].min(axis=1)
    df_avg['cv'] = df_avg['std_abs'] / df_avg['mean_abs']

    return df_avg


def get_metadata_columns(df: pd.DataFrame) -> list:
    """Return non-absorbance, non-target metadata column names.

    Args:
        df: DataFrame.

    Returns:
        List of metadata column names (e.g., 'aliphatic.index', 'charge').
    """
    abs_cols = set(get_absorbance_columns(df))
    skip = abs_cols | {'Number', 'Sequence', 'Notes', 'Absorbance',
                       'mean_abs', 'std_abs', 'max_abs', 'min_abs', 'cv', 'abs_bin'}
    return [c for c in df.columns if c not in skip]


def get_default_data_dir() -> str:
    """Return the default data directory (relative to repo root).

    Returns:
        Path string to the data/ directory.
    """
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
