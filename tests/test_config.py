"""Tests for project configuration."""

from pathlib import Path

from src import config


def test_data_directories_are_under_project_root() -> None:
    """Verify configured data directories are inside the project folder."""
    assert isinstance(config.BASE_DIR, Path)
    assert config.RAW_DATA_DIR.parent == config.DATA_DIR
    assert config.PROCESSED_DATA_DIR.parent == config.DATA_DIR


def test_raw_file_names_are_expected() -> None:
    """Verify raw data file names match the pipeline contract."""
    assert config.CUSTOMERS_FILE.name == "customers.csv"
    assert config.PRODUCTS_FILE.name == "products.csv"
    assert config.ORDERS_FILE.name == "orders.csv"
