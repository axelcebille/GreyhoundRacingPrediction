import numpy as np
import pandas as pd
from pathlib import Path
import re
import math
import joblib

from time import perf_counter
from IPython.display import display
from tqdm.auto import tqdm


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

def get_reference_columns(files):
    for path in files:
        if path.stat().st_size == 0:
            continue
        try:
            sample = pd.read_csv(path, nrows=0)
        except pd.errors.EmptyDataError:
            continue
        if len(sample.columns) == 0:
            continue
        return list(sample.columns), path.name
    raise ValueError("No non-empty CSV with a readable header was found.")


def merge_csv_folder(input_dir: Path, output_path: Path, add_source_file: bool = False):
    files = sorted(input_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    reference_columns, schema_source = get_reference_columns(files)
    output_columns = reference_columns + (["source_file"] if add_source_file else [])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    stats = {
        "files_seen": 0,
        "files_written": 0,
        "rows_written": 0,
        "skipped_empty_files": 0,
        "schema_mismatch_files": 0,
    }
    schema_mismatches = []
    wrote_header = False
    start = perf_counter()

    for path in tqdm(files, desc="Merging race_header CSVs"):
        stats["files_seen"] += 1

        if path.stat().st_size == 0:
            stats["skipped_empty_files"] += 1
            continue

        try:
            df = pd.read_csv(path, low_memory=False)
        except pd.errors.EmptyDataError:
            stats["skipped_empty_files"] += 1
            continue

        if df.empty:
            stats["skipped_empty_files"] += 1
            continue

        current_columns = list(df.columns)
        missing_cols = [col for col in reference_columns if col not in current_columns]
        extra_cols = [col for col in current_columns if col not in reference_columns]

        if missing_cols or extra_cols:
            stats["schema_mismatch_files"] += 1
            schema_mismatches.append({
                "source_file": path.name,
                "missing_cols": missing_cols,
                "extra_cols": extra_cols,
            })

        df = df.reindex(columns=reference_columns)
        if add_source_file:
            df["source_file"] = path.name

        df.to_csv(output_path, mode="a", header=not wrote_header, index=False)
        wrote_header = True

        stats["files_written"] += 1
        stats["rows_written"] += len(df)

    stats["elapsed_seconds"] = round(perf_counter() - start, 2)
    stats["output_path"] = str(output_path)
    stats["reference_schema_file"] = schema_source
    stats["output_size_mb"] = round(output_path.stat().st_size / 1_048_576, 2) if output_path.exists() else 0.0

    return pd.Series(stats), pd.DataFrame(schema_mismatches), output_columns
