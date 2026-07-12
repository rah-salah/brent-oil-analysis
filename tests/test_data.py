import pandas as pd
import pytest
import os


DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "brent_clean.csv")


def test_clean_data_file_exists():
    """The cleaned Brent oil price dataset should exist after running eda.py."""
    assert os.path.exists(DATA_PATH), f"Expected cleaned data file at {DATA_PATH}"


def test_clean_data_has_required_columns():
    """The cleaned dataset should contain a Date and Price column."""
    df = pd.read_csv(DATA_PATH)
    assert "Date" in df.columns
    assert "Price" in df.columns


def test_price_values_are_positive():
    """Oil prices should never be negative or zero in the cleaned dataset."""
    df = pd.read_csv(DATA_PATH)
    assert (df["Price"] > 0).all(), "Found non-positive price values in cleaned data"


def test_missing_file_raises_error():
    """Loading a non-existent file should raise a clear error rather than fail silently."""
    with pytest.raises(FileNotFoundError):
        pd.read_csv("data/processed/this_file_does_not_exist.csv")


def test_events_csv_has_minimum_events():
    """The events dataset should contain at least 10 documented events, per Task 1a."""
    events_path = os.path.join(os.path.dirname(__file__), "..", "data", "events.csv")
    df = pd.read_csv(events_path)
    assert len(df) >= 10, "Events dataset should contain at least 10 events"
    assert set(["date", "event", "category", "description"]).issubset(df.columns)
