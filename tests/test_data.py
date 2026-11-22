import pandas as pd

from src.electricity.data import FEATURES, audit, clean_training


def test_historical_corrections_preserve_unaffected_rows():
    dates = pd.date_range("2026-03-01", periods=9).strftime("%Y-%m-%d")
    train = pd.DataFrame({
        "date": dates, "rooms": [2] + [4] * 7 + [3],
        "members": [2, 3, 4, -1, 5, 6, 3, 2, 1],
        "ac_hours_per_day": [5, 300, 5, 6, 360, 5, 5, 6, 5],
        "feeder": ["Feeder-2" if i in (1, 4) else "Feeder-1" for i in range(9)],
        "connection": ["dom."] + ["domestic"] * 8,
        "avg_temp_c": list(range(20, 29)), "solar_kw": list(range(9)),
        "house_area_sqft": list(range(900, 909)), "units_kwh": list(range(100, 109)),
    })
    raw = pd.concat([train, train.iloc[[3]]], ignore_index=True)
    issues = audit(raw, raw[["date", *FEATURES]])
    assert issues["duplicate_rows"]["count"] == 1
    assert issues["sentinel_value"]["count"] == 2
    assert issues["stuck_sensor"]["start_date"] == dates[1]
    assert issues["stuck_sensor"]["end_date"] == dates[7]
    clean = clean_training(raw, issues)
    assert len(clean) == 9
    assert clean.rooms.iloc[1:8].isna().all()
    assert clean.rooms.iloc[[0, 8]].tolist() == [2, 3]
    assert clean.members.isna().sum() == 1
    assert clean.ac_hours_per_day.tolist() == [5, 5, 5, 6, 6, 5, 5, 6, 5]
    assert clean.connection.unique().tolist() == ["domestic"]
    assert raw.iloc[1].ac_hours_per_day == 300
