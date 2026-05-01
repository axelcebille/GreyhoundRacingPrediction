import numpy as np
import pandas as pd
from pathlib import Path
import re
import math
import joblib

CLASS_CONFIG = {

    # ── A GRADES: directly in LSTM sequence ──────────────────────
    # Numeric grade = inverse of number (A1 = best = highest level)
    'A1':  {'group': 'graded_a', 'relevance': 1.0, 'quality': 13, 'in_lstm': True},
    'A2':  {'group': 'graded_a', 'relevance': 1.0, 'quality': 12, 'in_lstm': True},
    'A3':  {'group': 'graded_a', 'relevance': 1.0, 'quality': 11, 'in_lstm': True},
    'A4':  {'group': 'graded_a', 'relevance': 1.0, 'quality': 10, 'in_lstm': True},
    'A5':  {'group': 'graded_a', 'relevance': 1.0, 'quality':  9, 'in_lstm': True},
    'A6':  {'group': 'graded_a', 'relevance': 1.0, 'quality':  8, 'in_lstm': True},
    'A7':  {'group': 'graded_a', 'relevance': 1.0, 'quality':  7, 'in_lstm': True},
    'A8':  {'group': 'graded_a', 'relevance': 1.0, 'quality':  6, 'in_lstm': True},
    'A9':  {'group': 'graded_a', 'relevance': 1.0, 'quality':  5, 'in_lstm': True},
    'A10': {'group': 'graded_a', 'relevance': 1.0, 'quality':  4, 'in_lstm': True},
    'A11': {'group': 'graded_a', 'relevance': 1.0, 'quality':  3, 'in_lstm': True},
    'A12': {'group': 'graded_a', 'relevance': 1.0, 'quality':  2, 'in_lstm': True},
    'A15': {'group': 'graded_a', 'relevance': 1.0, 'quality':  1, 'in_lstm': True},

    # ── OPEN RACES: static features, quality above A1 ───────────
    'OR':  {'group': 'open', 'relevance': 0.9, 'quality': 16, 'in_lstm': False},
    'OR1': {'group': 'open', 'relevance': 0.9, 'quality': 15, 'in_lstm': False},
    'OR2': {'group': 'open', 'relevance': 0.85,'quality': 14, 'in_lstm': False},
    'OR3': {'group': 'open', 'relevance': 0.80,'quality': 13, 'in_lstm': False},
    'GR':  {'group': 'open', 'relevance': 0.9, 'quality': 15, 'in_lstm': False},
    'E1':  {'group': 'open', 'relevance': 0.85,'quality': 14, 'in_lstm': False},
    'E2':  {'group': 'open', 'relevance': 0.80,'quality': 13, 'in_lstm': False},
    'E3':  {'group': 'open', 'relevance': 0.75,'quality': 12, 'in_lstm': False},

    # ── B GRADES: static only, sex-restricted ───────────────────
    'B1':  {'group': 'graded_b', 'relevance': 0.75,'quality': 10, 'in_lstm': False},
    'B2':  {'group': 'graded_b', 'relevance': 0.75,'quality':  9, 'in_lstm': False},
    'B3':  {'group': 'graded_b', 'relevance': 0.75,'quality':  8, 'in_lstm': False},
    'B4':  {'group': 'graded_b', 'relevance': 0.75,'quality':  7, 'in_lstm': False},
    'B5':  {'group': 'graded_b', 'relevance': 0.75,'quality':  6, 'in_lstm': False},
    'B6':  {'group': 'graded_b', 'relevance': 0.75,'quality':  5, 'in_lstm': False},
    'B7':  {'group': 'graded_b', 'relevance': 0.75,'quality':  4, 'in_lstm': False},
    'B8':  {'group': 'graded_b', 'relevance': 0.75,'quality':  3, 'in_lstm': False},
    'B9':  {'group': 'graded_b', 'relevance': 0.75,'quality':  2, 'in_lstm': False},
    'B15': {'group': 'graded_b', 'relevance': 0.75,'quality':  1, 'in_lstm': False},

    # ── S GRADES: sprint — static only, incompatible times ──────
    'S1':  {'group': 'sprint', 'relevance': 0.55,'quality':  8, 'in_lstm': False},
    'S2':  {'group': 'sprint', 'relevance': 0.55,'quality':  7, 'in_lstm': False},
    'S3':  {'group': 'sprint', 'relevance': 0.55,'quality':  6, 'in_lstm': False},
    'S4':  {'group': 'sprint', 'relevance': 0.55,'quality':  5, 'in_lstm': False},
    'S5':  {'group': 'sprint', 'relevance': 0.55,'quality':  4, 'in_lstm': False},
    'S6':  {'group': 'sprint', 'relevance': 0.55,'quality':  3, 'in_lstm': False},
    'S7':  {'group': 'sprint', 'relevance': 0.55,'quality':  2, 'in_lstm': False},
    'S8':  {'group': 'sprint', 'relevance': 0.55,'quality':  1, 'in_lstm': False},
    'S15': {'group': 'sprint', 'relevance': 0.55,'quality':  1, 'in_lstm': False},

    # ── D GRADES: distance — static only, incompatible times ────
    'D1':  {'group': 'distance', 'relevance': 0.45,'quality':  5, 'in_lstm': False},
    'D2':  {'group': 'distance', 'relevance': 0.45,'quality':  4, 'in_lstm': False},
    'D3':  {'group': 'distance', 'relevance': 0.45,'quality':  3, 'in_lstm': False},
    'D4':  {'group': 'distance', 'relevance': 0.45,'quality':  2, 'in_lstm': False},
    'D5':  {'group': 'distance', 'relevance': 0.45,'quality':  1, 'in_lstm': False},

    # ── M GRADES: middle distance — static only ──────────────────
    'M1':  {'group': 'middle', 'relevance': 0.60,'quality':  5, 'in_lstm': False},
    'M2':  {'group': 'middle', 'relevance': 0.60,'quality':  4, 'in_lstm': False},
    'M3':  {'group': 'middle', 'relevance': 0.60,'quality':  3, 'in_lstm': False},
    'M4':  {'group': 'middle', 'relevance': 0.60,'quality':  2, 'in_lstm': False},
    'M5':  {'group': 'middle', 'relevance': 0.60,'quality':  1, 'in_lstm': False},

    # ── P GRADES: puppy/juvenile — static only ───────────────────
    'P1':  {'group': 'puppy', 'relevance': 0.40,'quality':  9, 'in_lstm': False},
    'P2':  {'group': 'puppy', 'relevance': 0.40,'quality':  8, 'in_lstm': False},
    'P3':  {'group': 'puppy', 'relevance': 0.40,'quality':  7, 'in_lstm': False},
    'P4':  {'group': 'puppy', 'relevance': 0.40,'quality':  6, 'in_lstm': False},
    'P5':  {'group': 'puppy', 'relevance': 0.40,'quality':  5, 'in_lstm': False},
    'P6':  {'group': 'puppy', 'relevance': 0.40,'quality':  4, 'in_lstm': False},
    'P7':  {'group': 'puppy', 'relevance': 0.40,'quality':  3, 'in_lstm': False},
    'P8':  {'group': 'puppy', 'relevance': 0.40,'quality':  2, 'in_lstm': False},
    'P9':  {'group': 'puppy', 'relevance': 0.40,'quality':  1, 'in_lstm': False},
    'P10': {'group': 'puppy', 'relevance': 0.40,'quality':  1, 'in_lstm': False},

    # ── HURDLES: exclude entirely from performance features ──────
    'H1':  {'group': 'hurdles', 'relevance': 0.05,'quality': 0, 'in_lstm': False},
    'H2':  {'group': 'hurdles', 'relevance': 0.05,'quality': 0, 'in_lstm': False},
    'H3':  {'group': 'hurdles', 'relevance': 0.05,'quality': 0, 'in_lstm': False},
    'H4':  {'group': 'hurdles', 'relevance': 0.05,'quality': 0, 'in_lstm': False},
    'HS1': {'group': 'hurdles', 'relevance': 0.05,'quality': 0, 'in_lstm': False},
    'HS2': {'group': 'hurdles', 'relevance': 0.05,'quality': 0, 'in_lstm': False},
    'HD2': {'group': 'hurdles', 'relevance': 0.05,'quality': 0, 'in_lstm': False},
    'HP':  {'group': 'hurdles', 'relevance': 0.05,'quality': 0, 'in_lstm': False},

    # ── TRIALS / SPECIAL: flag only ──────────────────────────────
    'IT':  {'group': 'trial', 'relevance': 0.05,'quality': 0, 'in_lstm': False},
    'IV':  {'group': 'trial', 'relevance': 0.05,'quality': 0, 'in_lstm': False},
    'KS':  {'group': 'trial', 'relevance': 0.05,'quality': 0, 'in_lstm': False},
    'ks':  {'group': 'trial', 'relevance': 0.05,'quality': 0, 'in_lstm': False},
}

FALLBACK = {'group': 'unknown', 'relevance': 0.10, 'quality': 0, 'in_lstm': False}

def get_class_config(race_class):
    if pd.isna(race_class):
        return FALLBACK
    return CLASS_CONFIG.get(str(race_class).strip(), FALLBACK)

def preprocess_pipeline_race_header(race_header):
    race_header["raceDate"] = pd.to_datetime(race_header["raceDate"], format="%d/%m/%Y")
    column_subset = ["raceDate","raceId","meeting_Id","raceType","raceClass","raceDistance"]

    race_header_subset = race_header[column_subset]
    race_header_subset = race_header_subset.copy()
    race_header_subset.loc[:, "raceType"] = pd.Categorical(
        race_header_subset["raceType"],
        categories=["Flat", "Hurdles"]
    )    
    race_header_dummies = pd.get_dummies(
        race_header,
        columns=["raceType"],
        drop_first=False,
        dtype=int
    )

    race_header_dummies["raceYear"] = race_header_dummies["raceDate"].dt.year
    race_header_dummies["raceMonth"] = race_header_dummies["raceDate"].dt.month
    race_header_dummies["raceDayOfWeek"] = race_header_dummies["raceDate"].dt.dayofweek
    race_header_dummies["raceDayOfYear"] = race_header_dummies["raceDate"].dt.dayofyear


    #race_header_dummies["raceGoing"] = pd.to_numeric(race_header_dummies["raceGoing"], errors='coerce')
    #race_header_dummies["raceGoing"] = race_header_dummies["raceGoing"].fillna(0) 
    
    #race_header_dummies["raceClass"] = race_header_dummies["raceClass"].fillna("A6") 
    #race_header_dummies["raceClass"]  = race_header_dummies["raceClass"].apply(lambda x: 1 / int(x[1:]) if isinstance(x, str) and x.startswith('A') else 1/6)

    race_header_dummies['classGroup']   = race_header_dummies['raceClass'].apply(lambda c: get_class_config(c)['group'])
    race_header_dummies['qualityLevel'] = race_header_dummies['raceClass'].apply(lambda c: get_class_config(c)['quality'])

    return race_header_dummies
