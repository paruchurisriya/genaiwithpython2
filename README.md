# E-Commerce Data Pipeline

This project generates fake e-commerce data and analyzes it with PySpark to
produce business insights. It is designed as a small, practical data engineering
project for testing, learning, and portfolio use.

## Project Purpose

- Generate fake customer, product, and order data for testing.
- Analyze generated data with PySpark.
- Save raw data and processed analytics outputs in separate folders.

## Folder Structure

```text
.
├── data/
│   ├── processed/
│   │   └── .gitkeep
│   └── raw/
│       └── .gitkeep
├── notebooks/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_generator.py
│   └── spark_analytics.py
├── tests/
│   ├── __init__.py
│   └── test_config.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source venv/bin/activate
```

## Generate Raw Data

```bash
python -m src.data_generator
```

This creates:

- `data/raw/customers.csv`
- `data/raw/products.csv`
- `data/raw/orders.csv`

## Run PySpark Analytics

```bash
python -m src.spark_analytics
```

This creates CSV output folders under `data/processed`, including:

- revenue by category
- top customers
- monthly revenue
- order status summary

## Run Tests

```bash
pytest
```

## Notes

PySpark requires Java. If Spark fails to start, install a current Java runtime
and ensure `JAVA_HOME` is configured correctly.
