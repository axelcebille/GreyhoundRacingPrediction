import numpy as np
import pandas as pd

try:
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    class Dataset:
        pass

class GreyhoundRaceDataset(Dataset):
    def __init__(
        self,
        race_headers,
        race_stats,
        past_dog_infos,
        seq_len=10,
        history_feature_columns=None,
        strict_six_dogs=True,
        drop_incomplete_races=False,
        history_order="most_recent_first",
    ):
        required_header_cols = {"source_file", "raceDate"}
        required_stats_cols = {"source_file", "trapNumber", "dogId"}
        required_history_cols = {"dogId", "raceDate", "inLSTM"}

        missing_header_cols = required_header_cols.difference(race_headers.columns)
        missing_stats_cols = required_stats_cols.difference(race_stats.columns)
        missing_history_cols = required_history_cols.difference(past_dog_infos.columns)

        if missing_header_cols:
            raise KeyError(f"race_headers is missing columns: {sorted(missing_header_cols)}")
        if missing_stats_cols:
            raise KeyError(f"race_stats is missing columns: {sorted(missing_stats_cols)}")
        if missing_history_cols:
            raise KeyError(f"past_dog_infos is missing columns: {sorted(missing_history_cols)}")

        if history_order not in {"most_recent_first", "chronological"}:
            raise ValueError("history_order must be either 'most_recent_first' or 'chronological'.")

        self.seq_len = seq_len
        self.history_order = history_order

        race_headers = race_headers.copy()
        race_headers["raceDate"] = pd.to_datetime(
            race_headers["raceDate"],
            format="%d/%m/%Y",
            errors="coerce",
        )

        stats_counts = race_stats["source_file"].value_counts()
        header_counts = race_headers["source_file"].map(stats_counts)
        missing_stats_mask = header_counts.isna()
        invalid_count_mask = header_counts.notna() & header_counts.ne(6)

        self.validation_summary = {
            "header_races": int(len(race_headers)),
            "header_races_with_stats": int(header_counts.notna().sum()),
            "header_races_missing_stats": int(missing_stats_mask.sum()),
            "header_races_with_exactly_6_dogs": int((header_counts == 6).sum()),
            "header_races_with_non_6_dogs": int(invalid_count_mask.sum()),
        }
        self.invalid_race_stats_counts = (
            race_headers.loc[invalid_count_mask, ["source_file"]]
            .assign(dog_count=header_counts.loc[invalid_count_mask].astype(int).to_numpy())
            .drop_duplicates(subset="source_file")
            .reset_index(drop=True)
        )
        self.missing_race_stats_headers = race_headers.loc[missing_stats_mask, ["source_file"]].drop_duplicates().reset_index(drop=True)

        if strict_six_dogs and not self.invalid_race_stats_counts.empty:
            raise ValueError(
                f"Found {len(self.invalid_race_stats_counts)} header races whose race_stats block does not contain exactly 6 dogs. "
                "Set strict_six_dogs=False and drop_incomplete_races=True to filter them out."
            )

        if strict_six_dogs and not self.missing_race_stats_headers.empty:
            raise ValueError(
                f"Found {len(self.missing_race_stats_headers)} header races without any matching race_stats rows. "
                "Set strict_six_dogs=False and drop_incomplete_races=True to filter them out."
            )

        valid_headers = race_headers.loc[~missing_stats_mask].copy()
        valid_headers["dog_count"] = header_counts.loc[~missing_stats_mask].to_numpy()
        if drop_incomplete_races:
            valid_headers = valid_headers.loc[valid_headers["dog_count"] == 6].copy()

        self.race_headers = valid_headers.drop(columns="dog_count").set_index("source_file", drop=False)
        self.source_files = self.race_headers["source_file"].tolist()

        race_stats_group_indices = race_stats.groupby("source_file", sort=False).indices
        self.race_stats = race_stats
        self.race_stats_group_indices = {
            source_file: race_stats_group_indices[source_file]
            for source_file in self.source_files
            if source_file in race_stats_group_indices
        }

        if history_feature_columns is None:
            history_feature_columns = [
                column
                for column in past_dog_infos.columns
                if column not in {"dogId", "raceDate", "inLSTM"}
                and pd.api.types.is_numeric_dtype(past_dog_infos[column])
            ]

        missing_history_features = set(history_feature_columns).difference(past_dog_infos.columns)
        if missing_history_features:
            raise KeyError(f"Unknown history feature columns: {sorted(missing_history_features)}")

        self.history_feature_columns = list(history_feature_columns)
        if not self.history_feature_columns:
            raise ValueError("history_feature_columns is empty. At least one numeric history feature is required.")

        history_columns = ["dogId", "raceDate", "inLSTM", *self.history_feature_columns]
        history_columns = list(dict.fromkeys(history_columns))

        self.past_dog_infos = past_dog_infos.loc[:, history_columns].copy()
        self.past_dog_infos = self.past_dog_infos.loc[self.past_dog_infos["inLSTM"].fillna(False).astype(bool)].copy()
        self.past_dog_infos = self.past_dog_infos.drop(columns="inLSTM")
        self.past_dog_infos["raceDate"] = pd.to_datetime(self.past_dog_infos["raceDate"], errors="coerce")
        self.past_dog_infos = self.past_dog_infos.sort_values(["dogId", "raceDate"], kind="mergesort").reset_index(drop=True)
        self.history_group_indices = self.past_dog_infos.groupby("dogId", sort=False).indices

    def __len__(self):
        return len(self.source_files)

    def _slice_dog_history(self, dog_id, race_date):
        dog_index = self.history_group_indices.get(dog_id)
        if dog_index is None:
            empty_history = self.past_dog_infos.iloc[0:0].copy()
            empty_matrix = np.full((self.seq_len, len(self.history_feature_columns)), np.nan, dtype=np.float32)
            return empty_history, empty_matrix, 0

        dog_history = self.past_dog_infos.iloc[dog_index]
        dog_dates = dog_history["raceDate"].to_numpy(dtype="datetime64[ns]")
        cutoff = np.searchsorted(dog_dates, np.datetime64(race_date), side="left")
        start = max(0, cutoff - self.seq_len)
        history = dog_history.iloc[start:cutoff].copy()

        if self.history_order == "most_recent_first":
            history = history.iloc[::-1].reset_index(drop=True)
        else:
            history = history.reset_index(drop=True)

        history_matrix = np.full((self.seq_len, len(self.history_feature_columns)), np.nan, dtype=np.float32)
        valid_length = len(history)
        if valid_length:
            history_values = history[self.history_feature_columns].to_numpy(dtype=np.float32, copy=True)
            history_matrix[:valid_length] = history_values

        return history, history_matrix, valid_length

    def __getitem__(self, idx):
        source_file = self.source_files[idx]
        race_header = self.race_headers.loc[source_file].copy()

        race_stats_index = self.race_stats_group_indices[source_file]
        race_stat_block = self.race_stats.iloc[race_stats_index].sort_values("trapNumber", kind="mergesort").reset_index(drop=True)

        dog_histories = []
        dog_history_matrices = []
        dog_history_lengths = []

        for dog_id in race_stat_block["dogId"].tolist():
            dog_history, dog_history_matrix, dog_history_length = self._slice_dog_history(dog_id, race_header["raceDate"])
            dog_histories.append(dog_history)
            dog_history_matrices.append(dog_history_matrix)
            dog_history_lengths.append(dog_history_length)

        return {
            "source_file": source_file,
            "race_header": race_header,
            "race_stats": race_stat_block,
            "dog_histories": dog_histories,
            "dog_history_matrix": np.stack(dog_history_matrices, axis=0),
            "dog_history_lengths": np.asarray(dog_history_lengths, dtype=np.int64),
            "history_feature_columns": self.history_feature_columns,
        }
