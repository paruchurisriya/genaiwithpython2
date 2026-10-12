"""Synthetic e-commerce data generator."""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm

from src.config import CUSTOMERS_FILE, LOG_FORMAT, ORDERS_FILE, PRODUCTS_FILE, RANDOM_SEED, RAW_DATA_DIR

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """Generate synthetic customer, product, and order data for e-commerce analytics."""

    def __init__(self, seed: int = RANDOM_SEED, locale: str = "en_US") -> None:
        """Initialize Faker and NumPy random generators.

        Args:
            seed: Random seed used for reproducible output.
            locale: Faker locale used for names, emails, and locations.
        """
        self.seed = seed
        self.fake = Faker(locale)
        Faker.seed(seed)
        self.rng = np.random.default_rng(seed)
        logger.info("Initialized SyntheticDataGenerator with seed=%s", seed)

    def generate_customers(self, count: int = 100_000) -> pd.DataFrame:
        """Generate synthetic customers.

        Customer age is sampled from a normal distribution centered around 35.

        Args:
            count: Number of customers to generate.

        Returns:
            DataFrame with customer_id, name, email, age, city, country, and
            registration_date columns.
        """
        logger.info("Generating %s customers", count)
        ages = np.clip(np.rint(self.rng.normal(loc=35, scale=12, size=count)), 18, 85).astype(int)
        registration_offsets = self.rng.integers(0, 365 * 5, size=count)
        today = date.today()

        rows: list[dict[str, object]] = []
        for index in tqdm(range(count), desc="Generating customers", unit="customer"):
            rows.append(
                {
                    "customer_id": index + 1,
                    "name": self.fake.name(),
                    "email": self.fake.unique.email(),
                    "age": int(ages[index]),
                    "city": self.fake.city(),
                    "country": self.fake.country(),
                    "registration_date": today - timedelta(days=int(registration_offsets[index])),
                }
            )

        customers = pd.DataFrame(rows)
        logger.info("Generated customers DataFrame with shape=%s", customers.shape)
        return customers

    def generate_products(self, count: int = 10_000) -> pd.DataFrame:
        """Generate synthetic products.

        Args:
            count: Number of products to generate.

        Returns:
            DataFrame with product_id, name, category, price, stock, and rating
            columns.
        """
        logger.info("Generating %s products", count)
        categories = np.array(["Electronics", "Clothing", "Home", "Sports", "Books"])
        selected_categories = self.rng.choice(categories, size=count)
        prices = np.round(self.rng.uniform(10.0, 500.0, size=count), 2)
        stock = self.rng.integers(0, 1_001, size=count)
        ratings = np.round(self.rng.uniform(1.0, 5.0, size=count), 1)

        rows: list[dict[str, object]] = []
        for index in tqdm(range(count), desc="Generating products", unit="product"):
            category = str(selected_categories[index])
            rows.append(
                {
                    "product_id": index + 1,
                    "name": f"{self.fake.word().title()} {category}",
                    "category": category,
                    "price": float(prices[index]),
                    "stock": int(stock[index]),
                    "rating": float(ratings[index]),
                }
            )

        products = pd.DataFrame(rows)
        logger.info("Generated products DataFrame with shape=%s", products.shape)
        return products

    def generate_orders(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        count: int = 1_000_000,
        chunk_size: int = 100_000,
    ) -> pd.DataFrame:
        """Generate synthetic orders with Pareto-distributed customer activity.

        Customer selection uses Pareto weights so a small share of customers
        produces a large share of orders, approximating the 20/80 pattern.

        Args:
            customers: Customer DataFrame containing a customer_id column.
            products: Product DataFrame containing a product_id column.
            count: Number of orders to generate.
            chunk_size: Number of orders generated per progress-bar step.

        Returns:
            DataFrame with order_id, customer_id, product_id, quantity, and
            order_date columns.
        """
        logger.info("Generating %s orders", count)
        self._validate_required_columns(customers, ["customer_id"], "customers")
        self._validate_required_columns(products, ["product_id"], "products")

        customer_ids = customers["customer_id"].to_numpy()
        product_ids = products["product_id"].to_numpy()
        customer_probabilities = self._pareto_customer_probabilities(len(customer_ids))
        today = date.today()

        chunks: list[pd.DataFrame] = []
        next_order_id = 1

        with tqdm(total=count, desc="Generating orders", unit="order") as progress:
            while next_order_id <= count:
                current_chunk_size = min(chunk_size, count - next_order_id + 1)
                order_ids = np.arange(next_order_id, next_order_id + current_chunk_size)
                order_offsets = self.rng.integers(0, 365 * 2, size=current_chunk_size)

                chunks.append(
                    pd.DataFrame(
                        {
                            "order_id": order_ids,
                            "customer_id": self.rng.choice(
                                customer_ids,
                                size=current_chunk_size,
                                replace=True,
                                p=customer_probabilities,
                            ),
                            "product_id": self.rng.choice(product_ids, size=current_chunk_size, replace=True),
                            "quantity": self.rng.integers(1, 11, size=current_chunk_size),
                            "order_date": [today - timedelta(days=int(offset)) for offset in order_offsets],
                        }
                    )
                )

                next_order_id += current_chunk_size
                progress.update(current_chunk_size)

        orders = pd.concat(chunks, ignore_index=True)
        logger.info("Generated orders DataFrame with shape=%s", orders.shape)
        return orders

    def generate_all(
        self,
        customer_count: int = 100_000,
        product_count: int = 10_000,
        order_count: int = 1_000_000,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate customers, products, and orders.

        Args:
            customer_count: Number of customers to generate.
            product_count: Number of products to generate.
            order_count: Number of orders to generate.

        Returns:
            Tuple of customers, products, and orders DataFrames.
        """
        customers = self.generate_customers(customer_count)
        products = self.generate_products(product_count)
        orders = self.generate_orders(customers, products, order_count)
        return customers, products, orders

    def save_all(
        self,
        output_dir: Path = RAW_DATA_DIR,
        customer_count: int = 100_000,
        product_count: int = 10_000,
        order_count: int = 1_000_000,
    ) -> None:
        """Generate all datasets and save them as CSV files.

        Args:
            output_dir: Directory where CSV files will be written.
            customer_count: Number of customers to generate.
            product_count: Number of products to generate.
            order_count: Number of orders to generate.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        customers, products, orders = self.generate_all(customer_count, product_count, order_count)

        self.save_dataframe(customers, output_dir / "customers.csv")
        self.save_dataframe(products, output_dir / "products.csv")
        self.save_dataframe(orders, output_dir / "orders.csv")
        logger.info("Saved all generated datasets to %s", output_dir)

    @staticmethod
    def save_dataframe(dataframe: pd.DataFrame, output_path: Path) -> None:
        """Save a DataFrame to a CSV file without the pandas index.

        Args:
            dataframe: DataFrame to save.
            output_path: Destination CSV file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_csv(output_path, index=False)
        logger.info("Saved %s rows to %s", len(dataframe), output_path)

    def _pareto_customer_probabilities(self, customer_count: int) -> np.ndarray:
        """Create Pareto-based customer order probabilities.

        Args:
            customer_count: Number of customers to weight.

        Returns:
            NumPy array of probabilities that sum to 1.
        """
        weights = self.rng.pareto(a=1.16, size=customer_count) + 1.0
        probabilities = weights / weights.sum()
        logger.debug("Created Pareto customer probabilities for %s customers", customer_count)
        return probabilities

    @staticmethod
    def _validate_required_columns(dataframe: pd.DataFrame, columns: list[str], dataframe_name: str) -> None:
        """Validate that a DataFrame contains required columns.

        Args:
            dataframe: DataFrame to validate.
            columns: Required column names.
            dataframe_name: Human-readable DataFrame name for error messages.

        Raises:
            ValueError: If one or more required columns are missing.
        """
        missing_columns = [column for column in columns if column not in dataframe.columns]
        if missing_columns:
            raise ValueError(f"{dataframe_name} is missing required columns: {missing_columns}")


def generate_customers(customer_count: int = 100_000) -> pd.DataFrame:
    """Generate a customer DataFrame using the default generator."""
    return SyntheticDataGenerator().generate_customers(customer_count)


def generate_products(product_count: int = 10_000) -> pd.DataFrame:
    """Generate a product DataFrame using the default generator."""
    return SyntheticDataGenerator().generate_products(product_count)


def generate_orders(
    customers: pd.DataFrame,
    products: pd.DataFrame,
    order_count: int = 1_000_000,
) -> pd.DataFrame:
    """Generate an orders DataFrame using the default generator."""
    return SyntheticDataGenerator().generate_orders(customers, products, order_count)


def generate_all_data(
    customer_count: int = 100_000,
    product_count: int = 10_000,
    order_count: int = 1_000_000,
) -> None:
    """Generate all datasets and save them to the configured raw data files."""
    generator = SyntheticDataGenerator()
    customers, products, orders = generator.generate_all(customer_count, product_count, order_count)
    generator.save_dataframe(customers, CUSTOMERS_FILE)
    generator.save_dataframe(products, PRODUCTS_FILE)
    generator.save_dataframe(orders, ORDERS_FILE)


if __name__ == "__main__":
    generate_all_data()
