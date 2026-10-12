"""PySpark sales analytics for synthetic e-commerce data."""

from __future__ import annotations

import logging
from pathlib import Path

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as spark_functions

from src import config

logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class SalesAnalytics:
    """Run sales analytics over generated e-commerce Parquet datasets."""

    def __init__(self, app_name: str = "ECommerceSalesAnalytics") -> None:
        """Initialize the analytics service.

        Args:
            app_name: Name used for the Spark application.
        """
        self.app_name = app_name
        self.spark: SparkSession | None = None

    def create_spark_session(self) -> SparkSession:
        """Create a local Spark session configured for analytics workloads.

        The session uses local mode, 4 GB driver and executor memory, adaptive
        query execution, and Kryo serialization.

        Returns:
            Configured SparkSession instance.
        """
        if self.spark is not None:
            logger.debug("Returning existing Spark session")
            return self.spark

        logger.info("Creating Spark session: %s", self.app_name)
        self.spark = (
            SparkSession.builder.appName(self.app_name)
            .master("local[*]")
            .config("spark.driver.memory", "4g")
            .config("spark.executor.memory", "4g")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.kryoserializer.buffer.max", "512m")
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.sql.adaptive.skewJoin.enabled", "true")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate()
        )
        self.spark.sparkContext.setLogLevel("WARN")
        return self.spark

    def load_parquet(self, path: str | Path) -> DataFrame:
        """Load a Parquet file or directory into a Spark DataFrame.

        Args:
            path: Parquet file or directory path.

        Returns:
            Spark DataFrame loaded from the Parquet path.
        """
        spark = self.create_spark_session()
        parquet_path = Path(path)
        logger.info("Loading Parquet data from %s", parquet_path)
        return spark.read.parquet(str(parquet_path))

    def top_customers_by_revenue(self, orders_df: DataFrame, products_df: DataFrame, n: int = 10) -> DataFrame:
        """Return top customers by total revenue.

        Orders are joined to products on product_id, then revenue is calculated
        as quantity multiplied by product price.

        Args:
            orders_df: Orders DataFrame with customer_id, product_id, and quantity.
            products_df: Products DataFrame with product_id and price.
            n: Number of top customers to return.

        Returns:
            DataFrame with customer_id, total_revenue, total_units, and order_count.
        """
        logger.info("Calculating top %s customers by revenue", n)
        order_revenue = self._orders_with_revenue(orders_df, products_df)

        return (
            order_revenue.groupBy("customer_id")
            .agg(
                spark_functions.round(spark_functions.sum("revenue"), 2).alias("total_revenue"),
                spark_functions.sum("quantity").alias("total_units"),
                spark_functions.countDistinct("order_id").alias("order_count"),
            )
            .orderBy(spark_functions.desc("total_revenue"))
            .limit(n)
        )

    def sales_by_category(self, orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Calculate revenue and units sold by product category.

        Args:
            orders_df: Orders DataFrame with product_id and quantity.
            products_df: Products DataFrame with product_id, category, and price.

        Returns:
            DataFrame with category, total_revenue, units_sold, and order_count.
        """
        logger.info("Calculating sales by category")
        order_revenue = self._orders_with_revenue(orders_df, products_df)

        return (
            order_revenue.groupBy("category")
            .agg(
                spark_functions.round(spark_functions.sum("revenue"), 2).alias("total_revenue"),
                spark_functions.sum("quantity").alias("units_sold"),
                spark_functions.countDistinct("order_id").alias("order_count"),
            )
            .orderBy(spark_functions.desc("total_revenue"))
        )

    def monthly_trends(self, orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Calculate monthly revenue and month-over-month growth percentage.

        Args:
            orders_df: Orders DataFrame with order_date, product_id, and quantity.
            products_df: Products DataFrame with product_id and price.

        Returns:
            DataFrame with order_month, monthly_revenue, previous_month_revenue,
            and revenue_growth_pct.
        """
        logger.info("Calculating monthly revenue trends")
        order_revenue = self._orders_with_revenue(orders_df, products_df)
        month_window = Window.orderBy("order_month")

        monthly_revenue = (
            order_revenue.withColumn("order_month", spark_functions.date_format("order_date", "yyyy-MM"))
            .groupBy("order_month")
            .agg(
                spark_functions.round(spark_functions.sum("revenue"), 2).alias("monthly_revenue"),
                spark_functions.sum("quantity").alias("units_sold"),
                spark_functions.countDistinct("order_id").alias("order_count"),
            )
        )

        return (
            monthly_revenue.withColumn(
                "previous_month_revenue",
                spark_functions.lag("monthly_revenue").over(month_window),
            )
            .withColumn(
                "revenue_growth_pct",
                spark_functions.when(
                    spark_functions.col("previous_month_revenue").isNull()
                    | (spark_functions.col("previous_month_revenue") == 0),
                    None,
                ).otherwise(
                    spark_functions.round(
                        (
                            (spark_functions.col("monthly_revenue") - spark_functions.col("previous_month_revenue"))
                            / spark_functions.col("previous_month_revenue")
                        )
                        * 100,
                        2,
                    )
                ),
            )
            .orderBy("order_month")
        )

    def stop(self) -> None:
        """Stop the active Spark session if one exists."""
        if self.spark is not None:
            logger.info("Stopping Spark session")
            self.spark.stop()
            self.spark = None

    @staticmethod
    def _orders_with_revenue(orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Join orders with products and add a revenue column.

        Args:
            orders_df: Orders DataFrame with product_id and quantity.
            products_df: Products DataFrame with product_id, category, and price.

        Returns:
            Joined DataFrame containing product fields and revenue.
        """
        return (
            orders_df.alias("orders")
            .join(products_df.alias("products"), on="product_id", how="inner")
            .withColumn(
                "revenue",
                spark_functions.col("orders.quantity").cast("double")
                * spark_functions.col("products.price").cast("double"),
            )
        )


def run_analytics() -> None:
    """Load raw Parquet datasets, run analytics, and write processed results."""
    analytics = SalesAnalytics()

    try:
        orders = analytics.load_parquet(config.RAW_DATA_DIR / "orders.parquet")
        products = analytics.load_parquet(config.RAW_DATA_DIR / "products.parquet")
        config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        analytics.top_customers_by_revenue(orders, products).write.mode("overwrite").parquet(
            str(config.PROCESSED_DATA_DIR / "top_customers_by_revenue.parquet")
        )
        analytics.sales_by_category(orders, products).write.mode("overwrite").parquet(
            str(config.PROCESSED_DATA_DIR / "sales_by_category.parquet")
        )
        analytics.monthly_trends(orders, products).write.mode("overwrite").parquet(
            str(config.PROCESSED_DATA_DIR / "monthly_trends.parquet")
        )
        logger.info("Sales analytics complete")
    finally:
        analytics.stop()


if __name__ == "__main__":
    run_analytics()
