from __future__ import annotations

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


TARGET_USECOLS = [
    "trapNumber",
    "dogId",
    "dogBorn",
    "resultPosition",
    "resultComment",
    "source_file",
]

HEADER_USECOLS = ["raceDate", "source_file"]

NOTEBOOK_DROP_COLS = {
    "resultComment",
    "raceTime",
    "raceId",
    "raceType",
    "raceClass",
    "meetingId",
    "trackName",
    "dogId",
    "resultSectionalTime_missing",
    "classGroup",
    "inLSTM",
    "raceClassFamily",
}

WINDOWS = (3, 7, 200)
CHUNK_SIZE = 100_000


def load_target_rows(race_infos_path: Path, race_header_path: Path) -> pd.DataFrame:
    race_infos = pd.read_csv(
        race_infos_path,
        usecols=TARGET_USECOLS,
        low_memory=False,
    )
    race_header = pd.read_csv(race_header_path, usecols=HEADER_USECOLS)
    race_header["raceDate"] = pd.to_datetime(
        race_header["raceDate"],
        format="%d/%m/%Y",
        errors="coerce",
    )

    race_infos = race_infos[
        ~race_infos["resultComment"].fillna("").str.contains("NoRace", regex=False)
    ].copy()
    race_infos["didNotRace"] = race_infos["resultPosition"].isna().astype(np.int8)
    race_infos["resultPosition"] = race_infos["resultPosition"].fillna(8)

    race_infos = race_infos.merge(
        race_header,
        on="source_file",
        how="left",
        validate="m:1",
    )
    race_infos["dogBorn"] = pd.to_datetime(
        race_infos["dogBorn"],
        format="%b-%Y",
        errors="coerce",
    )
    race_infos["dogAge"] = (
        race_infos["raceDate"] - race_infos["dogBorn"]
    ).dt.days / 365.25

    race_infos = race_infos.drop(columns=["dogBorn", "resultComment"])
    race_infos = race_infos.sort_values(["source_file", "trapNumber"], kind="mergesort")
    race_infos = race_infos.reset_index(drop=True)
    race_infos["row_id"] = np.arange(len(race_infos), dtype=np.int64)
    return race_infos


def get_history_feature_columns(history_path: Path) -> list[str]:
    history_columns = pd.read_csv(history_path, nrows=0).columns.tolist()
    feature_columns = [
        column
        for column in history_columns
        if column not in NOTEBOOK_DROP_COLS and column != "raceDate"
    ]
    return feature_columns


def load_history_rows(history_path: Path, feature_columns: list[str]) -> pd.DataFrame:
    usecols = ["dogId", "raceDate"] + feature_columns
    history = pd.read_csv(
        history_path,
        usecols=usecols,
        low_memory=False,
    )
    history["raceDate"] = pd.to_datetime(history["raceDate"], errors="coerce")
    history = history.sort_values(["dogId", "raceDate"], kind="mergesort").reset_index(
        drop=True
    )
    return history


def make_feature_names(feature_columns: list[str]) -> list[str]:
    weighted_names = [
        f"Weighted{column}{window}"
        for window in WINDOWS
        for column in feature_columns
    ]
    percentage_names = []
    for window in WINDOWS:
        percentage_names.extend(
            [
                f"winPercentage{window}",
                f"oneTwoPercentage{window}",
                f"showPercentage{window}",
            ]
        )
    return weighted_names + percentage_names + ["DaysSinceLastRace", "num_races"]


def slice_cumsum_2d(cumsum: np.ndarray, positions: np.ndarray, window: int) -> np.ndarray:
    result = np.zeros((len(positions), cumsum.shape[1]), dtype=np.float64)
    valid = positions > 0
    if not valid.any():
        return result

    valid_positions = positions[valid]
    starts = np.maximum(valid_positions - window, 0)
    ends = valid_positions - 1
    window_sums = cumsum[ends].copy()

    start_prev = starts - 1
    has_prefix = start_prev >= 0
    window_sums[has_prefix] -= cumsum[start_prev[has_prefix]]

    result[valid] = window_sums
    return result


def slice_cumsum_1d(cumsum: np.ndarray, positions: np.ndarray, window: int) -> np.ndarray:
    result = np.zeros(len(positions), dtype=np.float64)
    valid = positions > 0
    if not valid.any():
        return result

    valid_positions = positions[valid]
    starts = np.maximum(valid_positions - window, 0)
    ends = valid_positions - 1
    window_sums = cumsum[ends].copy()

    start_prev = starts - 1
    has_prefix = start_prev >= 0
    window_sums[has_prefix] -= cumsum[start_prev[has_prefix]]

    result[valid] = window_sums
    return result


def compute_dog_feature_block(
    dog_history: pd.DataFrame,
    target_dates: np.ndarray,
    feature_columns: list[str],
) -> np.ndarray:
    output_width = len(feature_columns) * len(WINDOWS) + 3 * len(WINDOWS) + 2
    feature_block = np.full((len(target_dates), output_width), np.nan, dtype=np.float32)

    if dog_history.empty:
        feature_block[:, -1] = 0.0
        return feature_block

    history_dates = dog_history["raceDate"].to_numpy(dtype="datetime64[ns]")
    positions = np.searchsorted(history_dates, target_dates, side="left")

    numeric_history = dog_history[feature_columns].apply(pd.to_numeric, errors="coerce")
    numeric_values = numeric_history.replace([np.inf, -np.inf], np.nan).to_numpy(
        dtype=np.float64,
        copy=True,
    )
    numeric_values = np.nan_to_num(numeric_values, nan=0.0, posinf=0.0, neginf=0.0)

    if "relevance" not in dog_history.columns:
        raise KeyError("'relevance' column is required in the history dataset.")

    relevance = pd.to_numeric(dog_history["relevance"], errors="coerce").fillna(0.0)
    relevance_values = relevance.to_numpy(dtype=np.float64, copy=False)

    value_cumsum = np.cumsum(numeric_values, axis=0, dtype=np.float64)
    weighted_value_cumsum = np.cumsum(
        numeric_values * relevance_values[:, None],
        axis=0,
        dtype=np.float64,
    )
    relevance_cumsum = np.cumsum(relevance_values, dtype=np.float64)

    result_positions = pd.to_numeric(
        dog_history["resultPosition"],
        errors="coerce",
    ).to_numpy(dtype=np.float64, copy=False)
    win_cumsum = np.cumsum((result_positions == 1).astype(np.float64))
    one_two_cumsum = np.cumsum(np.isin(result_positions, [1, 2]).astype(np.float64))
    show_cumsum = np.cumsum(np.isin(result_positions, [1, 2, 3]).astype(np.float64))

    write_col = 0
    for window in WINDOWS:
        counts = np.minimum(positions, window).astype(np.float64)

        value_sums = slice_cumsum_2d(value_cumsum, positions, window)
        weighted_sums = slice_cumsum_2d(weighted_value_cumsum, positions, window)
        relevance_sums = slice_cumsum_1d(relevance_cumsum, positions, window)

        averages = np.full_like(value_sums, np.nan, dtype=np.float64)
        valid_rows = counts > 0
        if valid_rows.any():
            nonzero_relevance = valid_rows & (relevance_sums > 0)
            zero_relevance = valid_rows & (relevance_sums <= 0)

            if nonzero_relevance.any():
                averages[nonzero_relevance] = (
                    weighted_sums[nonzero_relevance]
                    / relevance_sums[nonzero_relevance, None]
                )
            if zero_relevance.any():
                averages[zero_relevance] = (
                    value_sums[zero_relevance]
                    / counts[zero_relevance, None]
                )

        next_col = write_col + len(feature_columns)
        feature_block[:, write_col:next_col] = averages.astype(np.float32)
        write_col = next_col

    for window in WINDOWS:
        for indicator_cumsum in (win_cumsum, one_two_cumsum, show_cumsum):
            counts = np.minimum(positions, window).astype(np.float64)
            indicator_sums = slice_cumsum_1d(indicator_cumsum, positions, window)
            ratios = np.divide(
                indicator_sums,
                counts,
                out=np.full(len(target_dates), np.nan, dtype=np.float64),
                where=counts > 0,
            )
            feature_block[:, write_col] = ratios.astype(np.float32)
            write_col += 1

    feature_block[:, write_col] = np.nan
    valid_previous_race = positions > 0
    if valid_previous_race.any():
        last_dates = history_dates[positions[valid_previous_race] - 1]
        day_gaps = (
            (target_dates[valid_previous_race] - last_dates) / np.timedelta64(1, "D")
        ).astype(np.float64)
        feature_block[valid_previous_race, write_col] = day_gaps.astype(np.float32)
    write_col += 1

    feature_block[:, write_col] = positions.astype(np.float32)
    return feature_block


def write_output_csv(
    base_rows: pd.DataFrame,
    feature_memmap: np.memmap,
    feature_names: list[str],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    base_output = base_rows.drop(columns=["row_id", "raceDate"]).copy()

    for start in tqdm(
        range(0, len(base_output), CHUNK_SIZE),
        desc="Writing CSV",
    ):
        end = min(start + CHUNK_SIZE, len(base_output))
        feature_chunk = pd.DataFrame(
            feature_memmap[start:end],
            columns=feature_names,
        )
        output_chunk = pd.concat(
            [base_output.iloc[start:end].reset_index(drop=True), feature_chunk],
            axis=1,
        )
        output_chunk.to_csv(
            output_path,
            mode="a",
            header=start == 0,
            index=False,
        )


def build_race_infos_dataset(
    race_infos_path: Path,
    race_header_path: Path,
    history_path: Path,
    output_path: Path,
) -> Path:
    target_rows = load_target_rows(race_infos_path, race_header_path)
    feature_columns = get_history_feature_columns(history_path)
    history_rows = load_history_rows(history_path, feature_columns)
    feature_names = make_feature_names(feature_columns)

    target_for_processing = target_rows[["row_id", "dogId", "raceDate"]].sort_values(
        ["dogId", "raceDate"],
        kind="mergesort",
    )
    history_for_processing = history_rows.sort_values(
        ["dogId", "raceDate"],
        kind="mergesort",
    )

    empty_history = history_rows.iloc[0:0]
    history_group_indices = history_for_processing.groupby("dogId", sort=False).indices

    temp_file = NamedTemporaryFile(
        prefix="race_infos_features_",
        suffix=".dat",
        delete=False,
    )
    temp_file.close()
    feature_memmap = np.memmap(
        temp_file.name,
        dtype=np.float32,
        mode="w+",
        shape=(len(target_rows), len(feature_names)),
    )
    feature_memmap[:] = np.nan

    target_groups = target_for_processing.groupby("dogId", sort=False)
    for dog_id, target_group in tqdm(target_groups, desc="Computing dog features"):
        target_dates = target_group["raceDate"].to_numpy(dtype="datetime64[ns]")
        history_idx = history_group_indices.get(dog_id)
        dog_history = (
            history_for_processing.iloc[history_idx]
            if history_idx is not None
            else empty_history
        )
        feature_block = compute_dog_feature_block(
            dog_history,
            target_dates,
            feature_columns,
        )
        feature_memmap[target_group["row_id"].to_numpy(dtype=np.int64)] = feature_block

    feature_memmap.flush()
    write_output_csv(target_rows, feature_memmap, feature_names, output_path)

    del feature_memmap
    Path(temp_file.name).unlink(missing_ok=True)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the processed race infos dataset from the notebook pipeline.",
    )
    parser.add_argument(
        "--race-infos",
        type=Path,
        default=Path("data/intermediate/08_all_race_infos.csv"),
        help="Merged race infos CSV.",
    )
    parser.add_argument(
        "--race-header",
        type=Path,
        default=Path("data/intermediate/07_all_race_header.csv"),
        help="Merged race header CSV used to recover race dates.",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("data/processed/05_dog_infos_engineered.csv"),
        help="Historical engineered dog infos CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/10_all_race_infos.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = build_race_infos_dataset(
        race_infos_path=args.race_infos,
        race_header_path=args.race_header,
        history_path=args.history,
        output_path=args.output,
    )
    print(f"Saved processed race infos to {output_path}")


if __name__ == "__main__":
    main()
