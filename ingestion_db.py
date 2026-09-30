"""
Data ingestion: loads raw CSV files from data/ into a SQLite database.

SQLite is used so the pipeline runs anywhere (Colab, laptop) without a
database server. The same SQLAlchemy code works for SQL Server/Postgres by
changing only DB_URL.
"""
import os
import time
import pandas as pd
from sqlalchemy import create_engine, text
from logging_setup import setup_logger

logger = setup_logger("ingestion_db")

DATA_DIR = "data"
DB_URL = "sqlite:///churn.db"
DATASET_URL = ("https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
               "master/data/Telco-Customer-Churn.csv")


def create_connection():
    """Create and return a SQLAlchemy engine."""
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info(f"Database connection established: {DB_URL}")
    return engine


def download_dataset_if_missing():
    """Download the IBM Telco churn dataset into data/ if no CSV is present."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if any(f.lower().endswith(".csv") for f in os.listdir(DATA_DIR)):
        return
    logger.info("No CSV in data/, downloading IBM Telco dataset")
    df = pd.read_csv(DATASET_URL)
    df.to_csv(os.path.join(DATA_DIR, "telecom_data.csv"), index=False)
    print("Downloaded dataset to data/telecom_data.csv")


def ingest_db(df, table_name, engine):
    """Write a dataframe into a database table (replaces if it exists)."""
    table_name = table_name.replace(" ", "_").replace("-", "_").replace(".", "_")
    df.to_sql(table_name, con=engine, if_exists="replace", index=False)
    with engine.connect() as conn:
        count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).fetchone()[0]
    logger.info(f"Ingested {count} rows into '{table_name}'")
    print(f"Ingested {count} rows into table '{table_name}'")
    return True


def load_raw_data():
    """Load every CSV in data/ into the database, one table per file."""
    download_dataset_if_missing()
    engine = create_connection()
    start = time.time()
    ok, failed = 0, 0
    try:
        for file in [f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]:
            try:
                df = pd.read_csv(os.path.join(DATA_DIR, file))
                if df.empty:
                    logger.warning(f"{file} is empty, skipping")
                    continue
                ingest_db(df, file[:-4], engine)
                ok += 1
            except Exception as e:
                logger.error(f"Failed to ingest {file}: {e}")
                failed += 1
    finally:
        engine.dispose()
    logger.info(f"Ingestion done: {ok} ok, {failed} failed, {time.time() - start:.1f}s")


if __name__ == "__main__":
    load_raw_data()
