"""Run PySpark sales analytics and display results."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from pyspark.sql import DataFrame

from src import config
from src.spark_analytics import SalesAnalytics

logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

T = TypeVar("T")


def time_operation(operation_name: str, operation: Callable[[], T]) -> T:
    """Run an operation and print how long it took.

    Args:
        operation_name: Human-readable operation name.
        operation: Callable operation to execute.

    Returns:
        The operation result.
    """
    start_time = time.perf_counter()
    result = operation()
    elapsed_seconds = time.perf_counter() - start_time
    print(f"{operation_name} completed in {elapsed_seconds:.2f} seconds")
    return result


def show_result(title: str, dataframe: DataFrame, rows: int = 20) -> None:
    """Print a title and display a Spark DataFrame.

    Args:
        title: Title to print before displaying the DataFrame.
        dataframe: Spark DataFrame to display.
        rows: Number of rows to show.
    """
    print(f"\n{title}")
    dataframe.show(rows, truncate=False)


def main() -> int:
    """Load raw Parquet data, run analytics, and display results.

    Returns:
        Process exit code. Returns 0 for success and 1 for failure.
    """
    analytics = SalesAnalytics()

    try:
        time_operation("Spark session creation", analytics.create_spark_session)

        raw_data_dir: Path = config.RAW_DATA_DIR
        customers_df = time_operation(
            "Loading customers.parquet",
            lambda: analytics.load_parquet(raw_data_dir / "customers.parquet"),
        )
        products_df = time_operation(
            "Loading products.parquet",
            lambda: analytics.load_parquet(raw_data_dir / "products.parquet"),
        )
        orders_df = time_operation(
            "Loading orders.parquet",
            lambda: analytics.load_parquet(raw_data_dir / "orders.parquet"),
        )

        print(f"Loaded customers columns: {', '.join(customers_df.columns)}")

        top_customers = time_operation(
            "Top customers by revenue",
            lambda: analytics.top_customers_by_revenue(orders_df, products_df, n=10),
        )
        show_result("Top Customers by Revenue", top_customers)

        category_sales = time_operation(
            "Sales by category",
            lambda: analytics.sales_by_category(orders_df, products_df),
        )
        show_result("Sales by Category", category_sales)

        trends = time_operation(
            "Monthly trends",
            lambda: analytics.monthly_trends(orders_df, products_df),
        )
        show_result("Monthly Revenue Trends", trends)

        return 0
    except Exception:
        logger.exception("Analytics run failed")
        return 1
    finally:
        analytics.stop()


if __name__ == "__main__":
    raise SystemExit(main())
