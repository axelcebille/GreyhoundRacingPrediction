import numpy as np
import pandas as pd
from pathlib import Path
import re
import math
import joblib
from src.dataset.load import load_dog_infos, fetch_dog_past_races
from src.features.process_features import fill_nan_with_column_mean, compute_n_averages_stats, compute_win_percentage, compute_one_two_percentage, compute_show_percentage, compute_trap_percentage
from src.utils import is_valid_csv

def process_dog_infos(dog_id, info_dogs_folder, race_date, trap_number_today, all_dogs_infos):
    #dog_infos = clean_dog_infos(info_dogs_folder, dog_id, race_date, trap_number_today)
    dog_infos = fetch_dog_past_races(dog_id, race_date, all_dogs_infos)


    processed_columns = ["SP","resultPosition","resultBtnDistance","relativeBetweenDistance","resultDogWeight","raceDistance",
                         "dogSpeed","dogSpeedWinner","dogDeltaSpeed","trapWeightFactor"]
    
    columns_to_drop = ["resultComment","winnerOr2ndName","winnerOr2ndId",'resultAdjustedTime', 'raceTime', 'raceDate', 'raceId',
       'raceType', 'raceClass', 'raceGoing', 'raceWinTime', 'meetingId',
       'trackName', 'absolute_raceTime', 'resultRunTime', 'trapNumber', 'resultSectionalTime',"Unnamed: 0",
    "trapNormedSecTimeTrack", "trapNormedResultTimeTrack", "trapNormedDogSpeedTrack", "trapNormedDogSpeedWinnerTrack",
    "trapNormedDogDeltaSpeedTrack","dogId"]
    dog_infos_cleaned = dog_infos.drop(columns=columns_to_drop)
    dog_infos_processed = fill_nan_with_column_mean(dog_infos_cleaned)

    #avg_stats_weighted_last3 = compute_n_averages_stats(dog_infos_processed, n=3, trap_weighted=True)
    avg_stats_last3 = compute_n_averages_stats(dog_infos_processed, n=3, trap_weighted=False)

    #avg_stats_weighted7 = compute_n_averages_stats(dog_infos_processed, n=7, trap_weighted=True)
    avg_stats7 = compute_n_averages_stats(dog_infos_processed, n=7, trap_weighted=False)

    #avg_stats_weighted = compute_n_averages_stats(dog_infos_processed, n=200, trap_weighted=True)
    avg_stats = compute_n_averages_stats(dog_infos_processed, n=200, trap_weighted=False)

    #full_stats = pd.concat([avg_stats_weighted_last3, avg_stats_last3, avg_stats_weighted7, avg_stats7, avg_stats_weighted, avg_stats], axis=1)
    full_stats = pd.concat([avg_stats_last3, avg_stats7, avg_stats], axis=1)

    for n in [3,7,200]:
        full_stats[f"winPercentage{n}"] = compute_win_percentage(dog_infos_processed, n=n)
        full_stats[f"oneTwoPercentage{n}"] = compute_one_two_percentage(dog_infos_processed, n=n)
        full_stats[f"showPercentage{n}"] = compute_show_percentage(dog_infos_processed, n=n)
        full_stats[f"insidePercentage{n}"], full_stats[f"middlePercentage{n}"], full_stats[f"outsidePercentage{n}"] = compute_trap_percentage(dog_infos_processed, n=n)

    last_race_date = dog_infos["raceDate"].iloc[0]
    full_stats["DaysSinceLastRace"] = (race_date - last_race_date).days

    full_stats["num_races"] = len(dog_infos)
    full_stats["beginnerDogFlag"] = beginner_dog_flag(dog_infos, min_races=15)
    #full_stats["newcomerDogFlag"] = newcommer_dog_flag(dog_infos)
    full_stats["experiencedDogFlag"] = experienced_dog_flag(dog_infos, min_races=15)

    return full_stats
    