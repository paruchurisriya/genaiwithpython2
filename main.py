"""Run synthetic e-commerce data generation and save raw Parquet files."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import pandas as pd

from src import config
from src.data_generator import SyntheticDataGenerator

logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


def format_file_size(size_bytes: int) -> str:
    """Format a file size in human-readable units.

    Args:
        size_bytes: File size in bytes.

    Returns:
        Human-readable file size string.
    """
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(size_bytes)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size_bytes} B"


def save_parquet(dataframe: pd.DataFrame, output_path: Path) -> int:
    """Save a pandas DataFrame as a Parquet file.

    Args:
        dataframe: DataFrame to save.
        output_path: Destination Parquet path.

    Returns:
        File size in bytes.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_parquet(output_path, index=False, engine="pyarrow")
    file_size = output_path.stat().st_size
    logger.info("Saved %s rows to %s", len(dataframe), output_path)
    return file_size


def main() -> int:
    """Generate synthetic datasets and save them to the raw data folder.

    Returns:
        Process exit code. Returns 0 for success and 1 for failure.
    """
    start_time = time.perf_counter()

    try:
        generator = SyntheticDataGenerator(seed=config.RANDOM_SEED)
        customers, products, orders = generator.generate_all()

        output_files = {
            "customers": config.RAW_DATA_DIR / "customers.parquet",
            "products": config.RAW_DATA_DIR / "products.parquet",
            "orders": config.RAW_DATA_DIR / "orders.parquet",
        }

        file_sizes = {
            "customers": save_parquet(customers, output_files["customers"]),
            "products": save_parquet(products, output_files["products"]),
            "orders": save_parquet(orders, output_files["orders"]),
        }

        elapsed_seconds = time.perf_counter() - start_time
        print(f"Data generation completed in {elapsed_seconds:.2f} seconds")
        print("Generated Parquet files:")
        for dataset_name, output_path in output_files.items():
            print(f"- {dataset_name}: {output_path} ({format_file_size(file_sizes[dataset_name])})")

        return 0
    except Exception:
        logger.exception("Data generation failed")
        print("Data generation failed. Check the logs above for details.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
