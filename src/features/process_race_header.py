import numpy as np
import pandas as pd
from pathlib import Path
import re
import math
import joblib

def preprocess_pipeline_race_header(race_header):
    race_header["raceDate"] = pd.to_datetime(race_header["raceDate"], format="%d/%m/%Y")
    column_subset = ["raceDate","raceId","meeting_Id","raceType","raceClass","raceDistance", "raceGoing"]

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

    race_header_dummies["raceGoing"] = pd.to_numeric(race_header_dummies["raceGoing"], errors='coerce')
    race_header_dummies["raceGoing"] = race_header_dummies["raceGoing"].fillna(0) 
    
    race_header_dummies["raceClass"] = race_header_dummies["raceClass"].fillna("A6") 
    race_header_dummies["raceClass"]  = race_header_dummies["raceClass"].apply(lambda x: 1 / int(x[1:]) if isinstance(x, str) and x.startswith('A') else 1/6)

    return race_header_dummies
