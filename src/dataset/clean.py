import numpy as np
import pandas as pd

def replace_outliers_with_mean_iqr(
    df: pd.DataFrame,
    cols: list[str],
    k: float = 1.5
) -> pd.DataFrame:
    """
    For each column in `cols`:
    1) detect outliers with IQR rule
    2) set outliers to NaN
    3) replace NaN by that column's mean (computed after outliers are removed)
    """
    out = df.copy()

    for c in cols:
        s = pd.to_numeric(out[c], errors="coerce")
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - k * iqr, q3 + k * iqr

        outlier_mask = (s < low) | (s > high)
        s_clean = s.mask(outlier_mask)          # outliers -> NaN
        mean_val = s_clean.mean(skipna=True)    # mean without outliers
        out[c] = s_clean.fillna(mean_val)       # fill NaN with mean

    return out

def remove_outliers(
    df: pd.DataFrame,
    cols: list[str],
    k: float = 1.5
) -> pd.DataFrame:
    """
    For each column in `cols`:
    1) detect outliers with IQR rule
    2) set outliers to NaN
    """
    out = df.copy()

    for c in cols:
        s = pd.to_numeric(out[c], errors="coerce")
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - k * iqr, q3 + k * iqr

        outlier_mask = (s < low) | (s > high)
        s_clean = s.mask(outlier_mask)          # outliers -> NaN
        out[c] = s_clean

    return out