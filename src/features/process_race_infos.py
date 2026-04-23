import numpy as np
import pandas as pd
from pathlib import Path
import re
import math
import joblib
from src.utils import is_valid_csv
from src.features.process_features import calculate_dog_age
from src.dataset.load import load_race_infos, load_dog_infos, fetch_dog_past_races
from src.features.process_race_header import preprocess_pipeline_race_header

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
