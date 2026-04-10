from pathlib import Path

import pandas as pd

from src.dataset.load import load_race_infos


def test_load_race_infos_reads_csv():
    project_root = Path(__file__).resolve().parents[2]
    race_info_path = project_root / "data" / "raw" / "race_info"
    example_id = 172134

    df = load_race_infos(race_info_path, example_id)

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.shape[1] > 0
