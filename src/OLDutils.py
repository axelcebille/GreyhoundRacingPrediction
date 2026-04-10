import numpy as np
import pandas as pd
from pathlib import Path
import re
import math
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]

##### loading utils CSV ####

mean_sec_time = pd.read_csv(PROJECT_ROOT / "mean_results.csv")
mean_sec_time.rename(columns={
    "Track Name": "trackName",
    "Race Distance": "raceDistance"}, inplace=True)

mean_track_trap_result = pd.read_csv(PROJECT_ROOT / "data_analysis_files/feature_stats_track_trap.csv")
mean_track_trap_result["trapNumber"] = pd.to_numeric(mean_track_trap_result['trapNumber'], errors='coerce')

mlp = joblib.load(PROJECT_ROOT / "models/mlp_sectional_time.joblib")
X_scaler = joblib.load(PROJECT_ROOT / "models/scalerX_sec_time.joblib")
y_scaler = joblib.load(PROJECT_ROOT / "models/scalerY_sec_time.joblib")

remarks_df = pd.read_csv(PROJECT_ROOT / "doggos_values_remarks.csv", delimiter=";")
all_dogs_infos = pd.read_csv(PROJECT_ROOT / "ALL_dog_infos_final.csv")
all_dogs_infos["raceDate"] = pd.to_datetime(all_dogs_infos["raceDate"], format="%Y-%m-%d")

# Build dictionary: lowercase remark -> score
remark_score = dict(
    zip(
        remarks_df["Remark"].str.lower(),
        remarks_df["Score"]
    )
)

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

def fill_nan_with_column_mean(df: pd.DataFrame) -> pd.DataFrame:
    # Fill NaNs in all numeric columns with their column mean, or 0 if mean is NaN

    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) == 0:
        return df
    # Normalize infinities before computing means
    df.loc[:, numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    # Use .loc to avoid SettingWithCopyWarning
    means = df[numeric_cols].mean()
    df.loc[:, numeric_cols] = df[numeric_cols].fillna(means.fillna(0))
    return df

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

def all_dogs_n_races_flag(race_id, race_info_folder, race_header_folder, n=7):
    race_infos = pd.read_csv(f"{race_info_folder}/{race_id}")
    race_header = pd.read_csv(f"{race_header_folder}/{race_id}")

    race_date = pd.to_datetime(race_header["raceDate"][0], format="%d/%m/%Y")
    dog_ids = race_infos["dogId"].to_list() 

    for dog_id in dog_ids:
        if len(load_dog_infos(info_dogs_folder, dog_id, race_date)) < n:
            return False
    return True

def compute_n_averages_stats(dog_infos, n=5, trap_weighted=True):
    """
    Compute the averages of the last n races for the given dog_infos DataFrame. 
    """
    last_n = dog_infos.iloc[:n].copy()  # get last n races
    last_n.drop(columns=["runnerType_inside", "runnerType_middle", "runnerType_outside"], inplace=True, errors="ignore")
    if last_n.shape[0] == 0:
        avg_stats = pd.Series(
            [np.nan] * len(dog_infos.columns),
            index=dog_infos.columns
        )
    else:
        # Coerce to numeric and sanitize infinities to avoid invalid reductions
        last_n_numeric = last_n.apply(pd.to_numeric, errors="coerce")
        last_n_numeric = last_n_numeric.replace([np.inf, -np.inf], np.nan)

        if trap_weighted:
            weights = pd.to_numeric(
                last_n["trapWeightFactor"],
                errors="coerce"
            ).fillna(0.0)
            denom = weights.sum()
            if not np.isfinite(denom) or denom == 0:
                weights = pd.Series(
                    np.ones(len(last_n)) / len(last_n),
                    index=last_n.index
                )
            else:
                weights = weights / denom
        else:
            weights = pd.Series(
                np.ones(len(last_n)) / len(last_n),
                index=last_n.index
            )

        avg_stats = (last_n_numeric * weights.values[:, np.newaxis]).sum()

    avg_stats_row = pd.DataFrame(avg_stats.values.reshape(1, -1), columns=avg_stats.index)
    if trap_weighted:
        avg_stats_row.columns = [f"Weighted{col}" + f"{n}" for col in avg_stats_row.columns]
    else:
        avg_stats_row.columns = [col + f"{n}" for col in avg_stats_row.columns]

    return avg_stats_row


#################################

##### cleaning dog info data #####

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

info_dogs_folder = (PROJECT_ROOT / "../GREYHOUND_RACING_ROMAIN/cleaned_files/info_dogs_cleaned/info_dogs").resolve()
def early_pos_race_dog(dog_infos, race_info_folder):
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

    dog_infos = fill_missing_sec_times_mlp(mlp,X_scaler,y_scaler,dog_infos)
    dog_infos = expected_features_analysis_dog_infos(dog_infos,mean_sec_time,mean_track_trap_result)
    #race_info_folder = Path("../GREYHOUND_RACING_ROMAIN/cleaned_files/race_info")
    #dog_infos = early_pos_race_dog(dog_infos, race_info_folder)

    return dog_infos

##########################################

##### preprocessing functions #####

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

#shuffle_type = "trap", "random"
def preprocess_pipeline_race_info(race_header, race_infos, shuffle_type="trap"):
    # get race header infos
    race_header_infos = preprocess_pipeline_race_header(race_header)

    ########## order by trap number or randomly ##########
    if shuffle_type == "trap":
        race_infos = race_infos.sort_values("trapNumber").reset_index(drop=True)
    elif shuffle_type == "random":
        race_infos = race_infos.sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        raise Exception("Shuffle type not accepted...")
    ##################################################
    
    ######## split results and no results ########
    column_results = ["SP","resultPosition","resultMarketPos","resultMarketCnt","resultPriceNumerator","resultPriceDenominator",
                      "resultBtnDistance","resultSectionalTime", "resultComment","resultRunTime","resultDogWeight","resultAdjustedTime"]
    column_no_results = [col for col in race_infos.columns if col not in column_results]

    subset_column = ["trapNumber", "dogBorn", "dogSex", "dogId"]

    race_info_no_results = race_infos[column_no_results].copy()
    race_info_results = race_infos[column_results].copy()
    race_result_positions = race_info_results["resultPosition"].copy()
    race_result_positions = race_result_positions.fillna(6)  # Handle missing result positions
    race_info_subset = race_info_no_results[subset_column].copy()
    ##################################################

    # transform back to datetime objects
    race_info_subset["dogBorn"] = pd.to_datetime(race_info_subset["dogBorn"], format="%b-%Y")

    # missing dog sex: set arbitrary to "b"
    race_info_subset["dogSex"] = race_info_subset["dogSex"].fillna("b")

    ######### one hot encoding categories (force all possible values) #########
    traps = race_info_subset["trapNumber"].copy()
    # force trapNumber 1-6 and dogSex "b","g" (example, adjust if needed)
    race_info_subset["trapNumber"] = pd.Categorical(race_info_subset["trapNumber"], categories=[1,2,3,4,5,6])
    race_info_subset["dogSex"] = pd.Categorical(race_info_subset["dogSex"], categories=["b","d"])
    
    race_info_dummies = pd.get_dummies(
        race_info_subset,
        columns=["trapNumber","dogSex"],
        drop_first=False,
        dtype=int
    )

    race_info_dummies["trapNumber"] = traps
    ###############################################

    ########## calculate dog age ##########
    race_date = race_header_infos["raceDate"][0]
    race_info_dummies["dogAge"] = race_info_dummies["dogBorn"].apply(lambda x: calculate_dog_age(race_date, x)) 
    race_info_dummies.drop(columns=["dogBorn"], inplace=True)

    return race_info_dummies, race_result_positions, race_date

def process_dog_infos(dog_id, info_dogs_folder, race_date, trap_number_today):
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
    
def full_preprocessing_pipeline(race_header, race_info_folder, info_dogs_folder):

    race_id = race_header["raceId"][0]
    race_info = load_race_infos(race_info_folder, race_id)

    race_info_dummies, race_result_positions, race_date = preprocess_pipeline_race_info(race_header, race_info, shuffle_type="trap")

    ####### ignore dog races with not exactly 6 dogs ######
    if (len(race_info_dummies) != 6) or (race_header["raceHandicap"][0]):
        return None
    ####### ignore sprint races ######
    if race_header["raceClass"][0].startswith("D"):
        return None
    #######################################################

    dogs_avg_stats = []
    for i in range(6):
        dog_id = race_info_dummies["dogId"].iloc[i]
        trap_number_today = race_info_dummies["trapNumber"].iloc[i]
        avg_stats_i = process_dog_infos(dog_id, info_dogs_folder, race_date, trap_number_today)
        dogs_avg_stats.append(avg_stats_i)

    df_dog_avg_stats = pd.concat(dogs_avg_stats, ignore_index=True)

    full_df_processed = pd.concat([race_info_dummies, df_dog_avg_stats, race_result_positions], axis=1)

    return full_df_processed

#################################
