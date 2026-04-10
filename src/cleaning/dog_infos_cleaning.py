import numpy as np
import pandas as pd
from pathlib import Path
import re
import math
import joblib
from src.dataset.load import *
from src.features.process_features import *

def fill_missing_sec_times_mlp(mlp,X_scaler,y_scaler,dog_infos):
    processed_columns = ["SP","resultPosition","resultBtnDistance","relativeBetweenDistance",
                         "resultDogWeight","raceDistance","dogSpeed","dogSpeedWinner","dogDeltaSpeed",
                         "resultAdjustedTime","trapNumber",
                         "raceWinTime","resultSectionalTime"]
    
    dog_data = dog_infos[processed_columns]
    data = dog_data[dog_data['resultSectionalTime'].isna()].copy()
    data = data.drop(columns=["resultSectionalTime"])

    if len(data) != 0:
        data = data.fillna(0) ## might not be the best idea... find better solution
        data_scaled = X_scaler.transform(data)
        y_scaled = mlp.predict(data_scaled)
        y_predicted = y_scaler.inverse_transform(
                y_scaled.reshape(-1, 1)).ravel()
        
        missing_idx = dog_infos[dog_infos['resultSectionalTime'].isna()]["resultSectionalTime"].index
        dog_infos.loc[missing_idx, "resultSectionalTime"] = y_predicted

        return dog_infos
    else:
        return dog_infos

def expected_features_analysis_dog_infos(dog_infos_,mean_sec_time,mean_track_trap_result):
    dog_infos = dog_infos_.copy()
    dog_infos = dog_infos.merge(mean_sec_time, on=["trackName","raceDistance"])
    dog_infos = dog_infos.merge(mean_track_trap_result, on=["trackName","trapNumber"])

    dog_infos["deltaExpectedRaceDistSecTime"] = dog_infos["Mean Sectional Time"] - dog_infos["resultSectionalTime"]
    dog_infos["trapNormedResultPosTrack"] = (dog_infos["resultPosition"] - dog_infos["resultPosition_mean"])/dog_infos["resultPosition_std"]
    dog_infos["trapNormedSecTimeTrack"] = (dog_infos["resultSectionalTime"] - dog_infos["resultSectionalTime_mean"])/dog_infos["resultSectionalTime_std"]
    dog_infos["trapNormedResultTimeTrack"] = (dog_infos["resultAdjustedTime"] - dog_infos["resultAdjustedTime_mean"])/dog_infos["resultAdjustedTime_std"]
    dog_infos["trapNormedDogSpeedTrack"] = (dog_infos["dogSpeed"])/dog_infos["resultAdjustedTime_std"]
    dog_infos["trapNormedDogSpeedWinnerTrack"] = (dog_infos["dogSpeedWinner"])/dog_infos["resultAdjustedTime_std"]
    dog_infos["trapNormedDogDeltaSpeedTrack"] = (dog_infos["dogDeltaSpeed"])/dog_infos["resultAdjustedTime_std"]

    columns_to_drop = ['Mean Sectional Time','resultPosition_mean', 'resultPosition_std',
       'resultSectionalTime_mean', 'resultSectionalTime_std',
       'resultAdjustedTime_mean', 'resultAdjustedTime_std']
    
    dog_infos = dog_infos.drop(columns=columns_to_drop)

    return dog_infos

def early_pos_race_dog(dog_infos, race_info_folder,info_dogs_folder, mlp, X_scaler, y_scaler):
    # Caches to avoid repeated disk reads
    race_cache = {}
    dog_cache = {}
    sec_time_cache = {}
    sec_time_map_cache = {}

    # Pre-convert raceDate once
    dog_infos = dog_infos.copy()
    dog_infos["raceDate"] = pd.to_datetime(dog_infos["raceDate"], format="%d/%m/%Y")

    early_positions = np.full(len(dog_infos), np.nan)

    for race_id, group in dog_infos.groupby("raceId", sort=False):
        try:
            # ---- Load race info (cached) ----
            if race_id not in race_cache:
                race = load_race_infos(race_info_folder, race_id)
                race = race.sort_values("trapNumber").reset_index(drop=True)
                race_cache[race_id] = race
            else:
                race = race_cache[race_id]

            dog_ids = race["dogId"].values
            sec_times = np.empty(len(dog_ids))

            # Assume all rows in group share the same race_date
            race_date = group["raceDate"].iloc[0]

            # ---- Loop dogs in race ----
            for j, dog_id in enumerate(dog_ids):
                cache_key = (dog_id, race_date)
                if cache_key in sec_time_cache:
                    sec_times[j] = sec_time_cache[cache_key]
                    continue

                if dog_id not in dog_cache:
                    dog_df = load_race_infos(info_dogs_folder, dog_id)
                    dog_df["raceDate"] = pd.to_datetime(
                        dog_df["raceDate"], format="%d/%m/%Y"
                    )
                    dog_cache[dog_id] = dog_df

                    sec_time_map_cache[dog_id] = (
                        dog_df.set_index("raceDate")["resultSectionalTime"]
                    )

                if dog_id in sec_time_map_cache:
                    sec_time_series = sec_time_map_cache[dog_id]
                    sec_time = sec_time_series.get(race_date, np.nan)
                else:
                    sec_time = np.nan

                if pd.isna(sec_time):
                    if dog_id in dog_cache:
                        dog_df = dog_cache[dog_id]
                        race_stats = dog_df.loc[dog_df["raceDate"] == race_date]
                        if not race_stats.empty:
                            race_stats = fill_missing_sec_times_mlp(
                                mlp, X_scaler, y_scaler, race_stats
                            )
                            sec_time = race_stats["resultSectionalTime"].iloc[0]

                sec_time_cache[cache_key] = sec_time
                sec_times[j] = sec_time

            # Build positions once per race
            sec_times_clean = np.where(np.isnan(sec_times), np.inf, sec_times)
            order = np.argsort(sec_times_clean)
            positions = np.empty_like(order)
            positions[order] = np.arange(1, len(sec_times_clean) + 1)

            for idx, trap_number in group["trapNumber"].items():
                try:
                    trap_idx = int(trap_number) - 1
                    if trap_idx < 0 or trap_idx >= len(sec_times):
                        early_positions[idx] = np.nan
                    elif np.isnan(sec_times[trap_idx]):
                        early_positions[idx] = np.nan
                    else:
                        early_positions[idx] = positions[trap_idx]
                except Exception:
                    early_positions[idx] = np.nan

        except (KeyError, IndexError, FileNotFoundError):
            for idx in group.index:
                early_positions[idx] = np.nan

    dog_infos["earlyPosition"] = early_positions
    dog_infos["deltaEarlyFinalPosition"] = dog_infos["earlyPosition"] - dog_infos["resultPosition"]
    return dog_infos



def clean_dog_infos(info_dogs_folder, dog_id, race_date, trap_number_today, mlp, X_scaler, y_scaler):
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

    dog_infos = fill_missing_sec_times_mlp(mlp,X_scaler,y_scaler,dog_infos)
    dog_infos = expected_features_analysis_dog_infos(dog_infos,mean_sec_time,mean_track_trap_result)
    #race_info_folder = Path("../GREYHOUND_RACING_ROMAIN/cleaned_files/race_info")
    #dog_infos = early_pos_race_dog(dog_infos, race_info_folder)

    return dog_infos
