import numpy as np
import pandas as pd
from pathlib import Path
import re
import math
import joblib


##### utils functions #####

def is_valid_csv(path):
    try:
        df = pd.read_csv(path)
        return df.shape[0] > 0 and df.shape[1] > 0
    except Exception:
        return False

def calculate_dog_age(race_date, birth_date):
    if pd.isnull(birth_date):
        return np.nan
    age_in_days = (race_date - birth_date).days / 365.25
    return age_in_days

def process_SP(sp_value):
    if pd.isnull(sp_value):
        return np.nan

    # Keep only digits, slash, and dot (ignore trailing letters)
    cleaned = re.match(r"[\d./]+", str(sp_value))
    if not cleaned:
        return np.nan

    cleaned = cleaned.group()

    if "/" in cleaned:
        numerator, denominator = cleaned.split("/")
        return float(numerator) / float(denominator)
    else:
        return float(cleaned)

def log_odds_from_fractional(numerator, denominator):
    """
    Fractional odds: numerator/denominator (e.g. 1/10)
    """
    p = denominator / (numerator + denominator)
    return math.log(p / (1 - p))

def calculate_log_odds_SP(sp_value):
    if pd.isnull(sp_value):
        return np.nan

    # Keep only digits, slash, and dot (ignore trailing letters)
    cleaned = re.match(r"[\d./]+", str(sp_value))
    if not cleaned:
        return np.nan

    cleaned = cleaned.group()

    if "/" in cleaned:
        numerator, denominator = cleaned.split("/")
    else:
        numerator = cleaned
        denominator = 1

    return log_odds_from_fractional(float(numerator), float(denominator))
    

def calculate_speed(distance, time):
    if pd.isnull(distance) or pd.isnull(time) or time == 0:
        return np.nan
    elif distance is None or time is None:
        return np.nan
    return float(distance) / float(time)

def parse_btn_distance(value, safe_numeric=True):

    CAP_DISTANCE = 10.0  # safe worst-plausible beaten distance

    MARGIN_MAP = {
        "SH": 0.1,
        "HD": 0.2,
        "NK": 0.3,
        "DH": 0.0
    }

    if value is None:
        return CAP_DISTANCE
    
    if pd.isnull(value):
        return 0.0

    # Skip strip() for numeric types if safe_numeric is True
    if safe_numeric and isinstance(value, (int, float)):
        return float(value)
    
    value = value.strip().upper()

    # Did not finish / disqualified
    if value in {"DNF", "DIS"}:
        return CAP_DISTANCE

    # Special margins
    if value in MARGIN_MAP:
        return MARGIN_MAP[value]

    # Fractional distances like "1 1/2"
    match = re.match(r"(\d+)\s+(\d+)/(\d+)", value)
    if match:
        whole, num, den = map(int, match.groups())
        return whole + num / den

    # Pure numeric
    try:
        return float(value)
    except ValueError:
        # Any unexpected string → treat as very poor performance
        return CAP_DISTANCE
    
def calculate_trap_weight_factor(trap_number_today, trap_number):
    weight = np.exp(-abs(trap_number_today - trap_number))
    return weight

def newcommer_dog_flag(dog_infos):
    if len(dog_infos) == 0:
        return 1
    else:
        return 0

def beginner_dog_flag(dog_infos, min_races=5):
    if len(dog_infos) < min_races:
        return 1
    else:
        return 0

def experienced_dog_flag(dog_infos, min_races=5):
    if len(dog_infos) >= min_races:
        return 1
    else:
        return 0

def compute_win_percentage(dog_infos, n=5):
    last_n = dog_infos.iloc[:n]
    if len(last_n) == 0:
        return np.nan
    win_count = (last_n["resultPosition"] == 1).sum()
    return win_count / len(last_n)

def compute_one_two_percentage(dog_infos, n=5):
    last_n = dog_infos.iloc[:n]
    if len(last_n) == 0:
        return np.nan
    one_two_count = last_n["resultPosition"].isin([1,2]).sum()
    return one_two_count / len(last_n)

def compute_show_percentage(dog_infos, n=5):
    last_n = dog_infos.iloc[:n]
    if len(last_n) == 0:
        return np.nan
    show_count = last_n["resultPosition"].isin([1,2,3]).sum()
    return show_count / len(last_n)

def runner_type(trap_number):
    if trap_number in [1, 2]:
        return "inside"
    elif trap_number in [3, 4]:
        return "middle"
    elif trap_number in [5, 6]:
        return "outside"
    else:
        return "middle"

def compute_trap_percentage(dog_infos, n=5):
    last_n = dog_infos.iloc[:n]
    if len(last_n) == 0:
        return np.nan, np.nan, np.nan
    
    inside = last_n["runnerType_inside"].sum() 
    middle = last_n["runnerType_middle"].sum() 
    outside = last_n["runnerType_outside"].sum() 
    
    total = inside + middle + outside
    if total == 0:
        return np.nan, np.nan, np.nan
    
    return inside / total, middle / total, outside / total

def score_result_comment(comment: str, remark_score: dict) -> float:
    if pd.isna(comment):
        return 0.0

    # normalize
    text = comment.lower()
    tokens = re.findall(r"[a-z]+", text)

    score = 0.0
    for token in tokens:
        if token in remark_score:
            score += remark_score[token]

    return score