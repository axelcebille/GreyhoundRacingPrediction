import numpy as np
import pandas as pd
from pathlib import Path
import re
import math
import joblib


##### loading functions #####

def load_race_infos(race_info_folder, race_id):
    path = race_info_folder / f"{race_id}.csv"
    dog_info = pd.read_csv(path)
    
    return dog_info

def load_dog_infos(dog_info_source, dog_id, race_date):
    """
    Load a dog's past races before ``race_date``.

    ``dog_info_source`` supports:
    - Path-like folder containing one CSV per dog (legacy behavior)
    - Pandas DataFrame with all dogs races (e.g., all_dog_infos)
    """
    if isinstance(dog_info_source, pd.DataFrame):
        dog_info = dog_info_source[dog_info_source["dogId"] == dog_id].copy()
    else:
        path = dog_info_source / f"{dog_id}.csv"
        dog_info = pd.read_csv(path)

    if dog_info.empty:
        return dog_info.reset_index(drop=True)

    if "raceDate" not in dog_info.columns:
        return dog_info.iloc[0:0].copy()

    if not pd.api.types.is_datetime64_any_dtype(dog_info["raceDate"]):
        dog_info["raceDate"] = pd.to_datetime(
            dog_info["raceDate"],
            format="%d/%m/%Y",
            errors="coerce",
        )
    race_date = pd.to_datetime(race_date, errors="coerce")
    dog_info = dog_info[dog_info["raceDate"] < race_date].reset_index(drop=True)
    return dog_info

def fetch_dog_past_races(dog_id, race_date, all_dogs_infos):
    dog_i_infos = all_dogs_infos[(all_dogs_infos["dogId"]==dog_id) & (all_dogs_infos["raceDate"]<race_date)].copy()
    dog_i_infos.reset_index(drop=True, inplace=True)
    return dog_i_infos

#################################