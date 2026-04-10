import numpy as np
import pandas as pd
from pathlib import Path
import re
import math
import joblib

def is_valid_csv(path):
    try:
        df = pd.read_csv(path)
        return df.shape[0] > 0 and df.shape[1] > 0
    except Exception:
        return False
    