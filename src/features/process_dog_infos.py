import numpy as np
import pandas as pd
from pathlib import Path
import re
import math
import joblib
from src.dataset.load import load_dog_infos, fetch_dog_past_races
from src.features.process_features import *
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

remark_score = None

def clean_dog_infos(info_dogs_folder, dog_id, race_date, trap_number_today):
    dog_infos = load_dog_infos(info_dogs_folder, dog_id, race_date)

    # replace * in trapNumber with 3
    dog_infos['trapNumber'] = dog_infos['trapNumber'].replace('*', 6)
    dog_infos['trapNumber'] = pd.to_numeric(dog_infos['trapNumber'], errors='coerce')

    # process Btn Distance column
    dog_infos["resultBtnDistance"] = dog_infos["resultBtnDistance"].apply(lambda x: parse_btn_distance(x))

    # relative between distance
    dog_infos["relativeBetweenDistance"] = dog_infos["resultBtnDistance"] / dog_infos["raceDistance"]

    # process SP column
    dog_infos["SP"] = dog_infos["SP"].apply(lambda x: calculate_log_odds_SP(x))

    # process weighted trap factor
    trap_numbers = pd.to_numeric(dog_infos["trapNumber"], errors="coerce")
    dog_infos["trapWeightFactor"] = np.exp(-np.abs(trap_numbers - trap_number_today))

    # process speed columns
    dog_infos["raceDistance"] = pd.to_numeric(dog_infos["raceDistance"], errors="coerce")
    dog_infos["resultDogWeight"] = pd.to_numeric(dog_infos["resultDogWeight"], errors="coerce")
    dog_infos["resultRunTime"] = pd.to_numeric(dog_infos["resultRunTime"], errors="coerce")
    dog_infos["raceWinTime"] = pd.to_numeric(dog_infos["raceWinTime"], errors="coerce")

    valid_run = (
        dog_infos["raceDistance"].notna()
        & dog_infos["resultRunTime"].notna()
        & (dog_infos["resultRunTime"] != 0)
    )
    valid_win = (
        dog_infos["raceDistance"].notna()
        & dog_infos["raceWinTime"].notna()
        & (dog_infos["raceWinTime"] != 0)
    )

    dog_infos["dogSpeed"] = np.where(
        valid_run,
        dog_infos["raceDistance"] / dog_infos["resultRunTime"],
        np.nan,
    )
    dog_infos["dogSpeedWinner"] = np.where(
        valid_win,
        dog_infos["raceDistance"] / dog_infos["raceWinTime"],
        np.nan,
    )
    dog_infos["dogDeltaSpeed"] = dog_infos["dogSpeed"] - dog_infos["dogSpeedWinner"]

    dog_infos["runnerType"] = dog_infos["trapNumber"].apply(runner_type)
    dog_infos["runnerType"] = pd.Categorical(
            dog_infos["runnerType"],
            categories=["inside", "middle", "outside"]
        )    

    dog_infos = pd.get_dummies(
            dog_infos,
            columns=["runnerType"],
            drop_first=False,
            dtype=int)
    
    # Ensure all runner type columns exist even if not in data
    for category in ["inside", "middle", "outside"]:
        col_name = f"runnerType_{category}"
        if col_name not in dog_infos.columns:
            dog_infos[col_name] = 0

    dog_infos["commentScore"] = dog_infos["resultComment"].apply(lambda x: score_result_comment(x, remark_score))

    ### TO ADD BACK LATER:
    #dog_infos = fill_missing_sec_times_mlp(mlp,X_scaler,y_scaler,dog_infos)
    #dog_infos = expected_features_analysis_dog_infos(dog_infos,mean_sec_time,mean_track_trap_result)
    #########################

    #race_info_folder = Path("../GREYHOUND_RACING_ROMAIN/cleaned_files/race_info")
    #dog_infos = early_pos_race_dog(dog_infos, race_info_folder)

    return dog_infos
    