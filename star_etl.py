import glob
import logging
import os
from pathlib import Path
from typing import List

import psycopg2
from pyspark.sql import SparkSession, functions as F, types as T
from pyspark.sql.window import Window

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Переменные окружения
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "bigdata_lab")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/opt/project"))

SPARK_JARS = os.getenv(
    "SPARK_JARS",
    "/opt/bitnami/spark/jars/postgresql-42.7.10.jar,"
    "/opt/bitnami/spark/jars/clickhouse-jdbc-0.7.1.jar",
)

POSTGRES_JDBC_URL = f"jdbc:postgresql://{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
POSTGRES_JDBC_PROPS = {
    "user": POSTGRES_USER,
    "password": POSTGRES_PASSWORD,
    "driver": "org.postgresql.Driver",
    "stringtype": "unspecified",
}

# Список колонок (оставлен без изменений)
RAW_COLUMNS = [
    "sale_id", "sale_date", "sale_customer_id", "customer_first_name",
    "customer_last_name", "customer_age", "customer_email", "customer_country",
    "customer_postal_code", "customer_pet_type", "customer_pet_name", "customer_pet_breed",
    "sale_seller_id", "seller_first_name", "seller_last_name", "seller_email",
    "seller_country", "seller_postal_code", "sale_product_id", "product_name",
    "product_category", "product_price", "product_quantity", "pet_category",
    "product_weight", "product_color", "product_size", "product_brand",
    "product_material", "product_description", "product_rating", "product_reviews",
    "product_release_date", "product_expiry_date", "sale_quantity", "sale_total_price",
    "store_name", "store_location", "store_city", "store_state", "store_country",
    "store_phone", "store_email", "supplier_name", "supplier_contact", "supplier_email",
    "supplier_phone", "supplier_address", "supplier_city", "supplier_country",
]

DIM_CUSTOMERS_COLUMNS = [
    "customer_key", "sale_customer_id", "customer_first_name", "customer_last_name",
    "customer_age", "customer_email", "customer_country", "customer_postal_code",
    "customer_pet_type", "customer_pet_name", "customer_pet_breed",
]

DIM_SELLERS_COLUMNS = [
    "seller_key", "sale_seller_id", "seller_first_name", "seller_last_name",
    "seller_email", "seller_country", "seller_postal_code",
]

DIM_PRODUCTS_COLUMNS = [
    "product_key", "sale_product_id", "product_name", "product_category",
    "product_price", "product_quantity", "pet_category", "product_weight",
    "product_color", "product_size", "product_brand", "product_material",
    "product_description", "product_rating", "product_reviews",
    "product_release_date", "product_expiry_date",
]

DIM_STORES_COLUMNS = [
    "store_key", "store_name", "store_location", "store_city", "store_state",
    "store_country", "store_phone", "store_email",
]

DIM_SUPPLIERS_COLUMNS = [
    "supplier_key", "supplier_name", "supplier_contact", "supplier_email",
    "supplier_phone", "supplier_address", "supplier_city", "supplier_country",
]

DIM_DATES_COLUMNS = [
    "date_key", "full_date", "day_num", "month_num", "month_name",
    "quarter_num", "year_num",
]

FACT_COLUMNS = [
    "sale_key", "source_sale_id", "date_key", "customer_key", "seller_key",
    "product_key", "store_key", "supplier_key", "sale_quantity", "sale_total_price",
]

# Явная схема для CSV (повышает надёжность)
CSV_SCHEMA = T.StructType([
    T.StructField("id", T.LongType(), True),
    T.StructField("sale_date", T.StringType(), True),
    T.StructField("sale_customer_id", T.LongType(), True),
    T.StructField("customer_first_name", T.StringType(), True),
    T.StructField("customer_last_name", T.StringType(), True),
    T.StructField("customer_age", T.IntegerType(), True),
    T.StructField("customer_email", T.StringType(), True),
    T.StructField("customer_country", T.StringType(), True),
    T.StructField("customer_postal_code", T.StringType(), True),
    T.StructField("customer_pet_type", T.StringType(), True),
    T.StructField("customer_pet_name", T.StringType(), True),
    T.StructField("customer_pet_breed", T.StringType(), True),
    T.StructField("sale_seller_id", T.LongType(), True),
    T.StructField("seller_first_name", T.StringType(), True),
    T.StructField("seller_last_name", T.StringType(), True),
    T.StructField("seller_email", T.StringType(), True),
    T.StructField("seller_country", T.StringType(), True),
    T.StructField("seller_postal_code", T.StringType(), True),
    T.StructField("sale_product_id", T.LongType(), True),
    T.StructField("product_name", T.StringType(), True),
    T.StructField("product_category", T.StringType(), True),
    T.StructField("product_price", T.DoubleType(), True),
    T.StructField("product_quantity", T.IntegerType(), True),
    T.StructField("pet_category", T.StringType(), True),
    T.StructField("product_weight", T.DoubleType(), True),
    T.StructField("product_color", T.StringType(), True),
    T.StructField("product_size", T.StringType(), True),
    T.StructField("product_brand", T.StringType(), True),
    T.StructField("product_material", T.StringType(), True),
    T.StructField("product_description", T.StringType(), True),
    T.StructField("product_rating", T.DoubleType(), True),
    T.StructField("product_reviews", T.IntegerType(), True),
    T.StructField("product_release_date", T.StringType(), True),
    T.StructField("product_expiry_date", T.StringType(), True),
    T.StructField("sale_quantity", T.IntegerType(), True),
    T.StructField("sale_total_price", T.DoubleType(), True),
    T.StructField("store_name", T.StringType(), True),
    T.StructField("store_location", T.StringType(), True),
    T.StructField("store_city", T.StringType(), True),
    T.StructField("store_state", T.StringType(), True),
    T.StructField("store_country", T.StringType(), True),
    T.StructField("store_phone", T.StringType(), True),
    T.StructField("store_email", T.StringType(), True),
    T.StructField("supplier_name", T.StringType(), True),
    T.StructField("supplier_contact", T.StringType(), True),
    T.StructField("supplier_email", T.StringType(), True),
    T.StructField("supplier_phone", T.StringType(), True),
    T.StructField("supplier_address", T.StringType(), True),
    T.StructField("supplier_city", T.StringType(), True),
    T.StructField("supplier_country", T.StringType(), True),
])


def get_pg_connection():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )


def execute_sql_file(filepath: Path) -> None:
    if not filepath.exists():
        raise FileNotFoundError(f"SQL file not found: {filepath}")
    sql = filepath.read_text(encoding="utf-8")
    conn = get_pg_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        logger.info(f"Executed SQL from {filepath}")
    finally:
        conn.close()


def truncate_postgres_tables() -> None:
    conn = get_pg_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE TABLE public.mock_data;")
            cur.execute("""
                TRUNCATE TABLE
                    fact_sales, dim_customers, dim_sellers, dim_products,
                    dim_stores, dim_suppliers, dim_dates
                RESTART IDENTITY CASCADE;
            """)
        conn.commit()
        logger.info("Truncated all star schema tables")
    finally:
        conn.close()


def discover_csv_files(project_root: Path) -> List[str]:
    patterns = [
        str(project_root / "**" / "MOCK_DATA*.csv"),
        str(project_root / "**" / "*.csv"),
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern, recursive=True))
    csv_files = sorted({p for p in candidates if "MOCK_DATA" in os.path.basename(p)})
    if not csv_files:
        raise FileNotFoundError("No MOCK_DATA*.csv files found in project.")
    logger.info(f"Found {len(csv_files)} CSV files")
    return csv_files


def build_spark() -> SparkSession:
    builder = (
        SparkSession.builder
        .appName("bigdata-lab-star-etl")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.jars", SPARK_JARS)
        .config("spark.driver.extraClassPath", SPARK_JARS.replace(",", ":"))
        .config("spark.executor.extraClassPath", SPARK_JARS.replace(",", ":"))
    )
    return builder.getOrCreate()


def clean_string_columns(df):
    """Trim string columns and replace empty strings with None."""
    for field in df.schema.fields:
        if isinstance(field.dataType, T.StringType):
            df = df.withColumn(
                field.name,
                F.when(F.trim(F.col(field.name)) == "", None).otherwise(F.trim(F.col(field.name)))
            )
    return df


def read_raw_data(spark: SparkSession, csv_files: List[str]):
    # Читаем с явной схемой
    raw_df = spark.read.schema(CSV_SCHEMA).option("multiLine", True).option("escape", '"').csv(csv_files)
    raw_df = raw_df.withColumnRenamed("id", "sale_id")
    raw_df = clean_string_columns(raw_df)

    # Преобразование дат
    raw_df = raw_df.withColumn("sale_date", F.to_date("sale_date", "M/d/yyyy"))
    raw_df = raw_df.withColumn("product_release_date", F.to_date("product_release_date", "M/d/yyyy"))
    raw_df = raw_df.withColumn("product_expiry_date", F.to_date("product_expiry_date", "M/d/yyyy"))

    # Округление цен
    raw_df = raw_df.withColumn("product_price", F.round(F.col("product_price"), 2))
    raw_df = raw_df.withColumn("sale_total_price", F.round(F.col("sale_total_price"), 2))
    raw_df = raw_df.withColumn("product_weight", F.round(F.col("product_weight"), 2))
    raw_df = raw_df.withColumn("product_rating", F.round(F.col("product_rating"), 2))

    return raw_df.select(*RAW_COLUMNS)


def create_dimension_tables(df):
    # Клиенты
    dim_customers = (
        df.select(
            F.col("sale_customer_id").alias("customer_key"),
            "sale_customer_id", "customer_first_name", "customer_last_name",
            "customer_age", "customer_email", "customer_country",
            "customer_postal_code", "customer_pet_type", "customer_pet_name",
            "customer_pet_breed"
        )
        .dropna(subset=["sale_customer_id"])
        .dropDuplicates(["customer_key"])
        .orderBy("customer_key")
    )

    # Продавцы
    dim_sellers = (
        df.select(
            F.col("sale_seller_id").alias("seller_key"),
            "sale_seller_id", "seller_first_name", "seller_last_name",
            "seller_email", "seller_country", "seller_postal_code"
        )
        .dropna(subset=["sale_seller_id"])
        .dropDuplicates(["seller_key"])
        .orderBy("seller_key")
    )

    # Продукты
    dim_products = (
        df.select(
            F.col("sale_product_id").alias("product_key"),
            "sale_product_id", "product_name", "product_category", "product_price",
            "product_quantity", "pet_category", "product_weight", "product_color",
            "product_size", "product_brand", "product_material", "product_description",
            "product_rating", "product_reviews", "product_release_date", "product_expiry_date"
        )
        .dropna(subset=["sale_product_id"])
        .dropDuplicates(["product_key"])
        .orderBy("product_key")
    )

    # Магазины – суррогатный ключ через monotonically_increasing_id
    dim_stores = (
        df.select(
            "store_name", "store_location", "store_city", "store_state",
            "store_country", "store_phone", "store_email"
        )
        .dropDuplicates()
        .withColumn("store_key", F.monotonically_increasing_id() + 1)
        .select(*DIM_STORES_COLUMNS)
        .orderBy("store_key")
    )

    # Поставщики
    dim_suppliers = (
        df.select(
            "supplier_name", "supplier_contact", "supplier_email", "supplier_phone",
            "supplier_address", "supplier_city", "supplier_country"
        )
        .dropDuplicates()
        .withColumn("supplier_key", F.monotonically_increasing_id() + 1)
        .select(*DIM_SUPPLIERS_COLUMNS)
        .orderBy("supplier_key")
    )

    # Даты
    dim_dates = (
        df.select(F.col("sale_date").alias("full_date"))
        .dropna(subset=["full_date"])
        .dropDuplicates(["full_date"])
        .withColumn("date_key", F.date_format("full_date", "yyyyMMdd").cast(T.IntegerType()))
        .withColumn("day_num", F.dayofmonth("full_date"))
        .withColumn("month_num", F.month("full_date"))
        .withColumn("month_name", F.date_format("full_date", "MMMM"))
        .withColumn("quarter_num", F.quarter("full_date"))
        .withColumn("year_num", F.year("full_date"))
        .select(*DIM_DATES_COLUMNS)
        .orderBy("date_key")
    )

    return dim_customers, dim_sellers, dim_products, dim_stores, dim_suppliers, dim_dates


def assemble_fact_table(df, dim_stores, dim_suppliers):
    # Формируем составной ключ для магазина
    fact_source = df.withColumn(
        "store_nk",
        F.concat_ws(
            "||",
            F.coalesce(F.col("store_name"), F.lit("")),
            F.coalesce(F.col("store_location"), F.lit("")),
            F.coalesce(F.col("store_city"), F.lit("")),
            F.coalesce(F.col("store_country"), F.lit("")),
            F.coalesce(F.col("store_phone"), F.lit("")),
            F.coalesce(F.col("store_email"), F.lit("")),
        )
    )

    dim_stores_for_join = dim_stores.withColumn(
        "store_nk",
        F.concat_ws(
            "||",
            F.coalesce(F.col("store_name"), F.lit("")),
            F.coalesce(F.col("store_location"), F.lit("")),
            F.coalesce(F.col("store_city"), F.lit("")),
            F.coalesce(F.col("store_country"), F.lit("")),
            F.coalesce(F.col("store_phone"), F.lit("")),
            F.coalesce(F.col("store_email"), F.lit("")),
        )
    ).select("store_key", "store_nk")

    supplier_join_cols = [
        "supplier_name", "supplier_contact", "supplier_email", "supplier_phone",
        "supplier_address", "supplier_city", "supplier_country"
    ]

    from pyspark.sql.functions import broadcast
    fact_df = (
        fact_source
        .join(broadcast(dim_stores_for_join), on="store_nk", how="left")
        .join(broadcast(dim_suppliers.select("supplier_key", *supplier_join_cols)),
              on=supplier_join_cols, how="left")
        .withColumn("source_sale_id", F.col("sale_id").cast(T.LongType()))
        .withColumn("date_key", F.date_format("sale_date", "yyyyMMdd").cast(T.IntegerType()))
        .withColumn("customer_key", F.col("sale_customer_id").cast(T.LongType()))
        .withColumn("seller_key", F.col("sale_seller_id").cast(T.LongType()))
        .withColumn("product_key", F.col("sale_product_id").cast(T.LongType()))
        .select(
            "source_sale_id", "date_key", "customer_key", "seller_key",
            "product_key", "store_key", "supplier_key", "sale_quantity", "sale_total_price"
        )
        .withColumn("sale_key", F.row_number().over(Window.orderBy("source_sale_id")))
        .select(*FACT_COLUMNS)
        .orderBy("sale_key")
    )
    return fact_df


def save_to_postgres(df, table_name: str):
    if df.count() == 0:
        logger.warning(f"DataFrame for {table_name} is empty, skipping write")
        return
    df.write.mode("append").jdbc(
        url=POSTGRES_JDBC_URL,
        table=table_name,
        properties=POSTGRES_JDBC_PROPS,
    )
    logger.info(f"Written {df.count()} rows to {table_name}")


def print_postgres_counts(spark: SparkSession):
    tables = [
        "public.mock_data", "public.dim_customers", "public.dim_sellers",
        "public.dim_products", "public.dim_stores", "public.dim_suppliers",
        "public.dim_dates", "public.fact_sales"
    ]
    for table in tables:
        cnt = spark.read.jdbc(
            url=POSTGRES_JDBC_URL,
            table=f"(SELECT COUNT(*) AS cnt FROM {table}) t",
            properties=POSTGRES_JDBC_PROPS,
        ).collect()[0]["cnt"]
        logger.info(f"{table}: {cnt}")


def main():
    logger.info("=== ETL: CSV -> PostgreSQL star schema (via Spark JDBC) ===")
    logger.info(f"PROJECT_ROOT = {PROJECT_ROOT}")

    csv_files = discover_csv_files(PROJECT_ROOT)
    for path in csv_files:
        logger.info(f"  CSV found: {path}")

    execute_sql_file(PROJECT_ROOT / "sql" / "init_postgres.sql")
    execute_sql_file(PROJECT_ROOT / "sql" / "create_star_schema.sql")
    truncate_postgres_tables()

    spark = build_spark()
    try:
        raw_df = read_raw_data(spark, csv_files)
        logger.info(f"Rows read from CSV: {raw_df.count()}")

        dims = create_dimension_tables(raw_df)
        dim_customers, dim_sellers, dim_products, dim_stores, dim_suppliers, dim_dates = dims
        fact_df = assemble_fact_table(raw_df, dim_stores, dim_suppliers)

        save_to_postgres(raw_df.select(*RAW_COLUMNS), "public.mock_data")
        save_to_postgres(dim_customers.select(*DIM_CUSTOMERS_COLUMNS), "public.dim_customers")
        save_to_postgres(dim_sellers.select(*DIM_SELLERS_COLUMNS), "public.dim_sellers")
        save_to_postgres(dim_products.select(*DIM_PRODUCTS_COLUMNS), "public.dim_products")
        save_to_postgres(dim_stores.select(*DIM_STORES_COLUMNS), "public.dim_stores")
        save_to_postgres(dim_suppliers.select(*DIM_SUPPLIERS_COLUMNS), "public.dim_suppliers")
        save_to_postgres(dim_dates.select(*DIM_DATES_COLUMNS), "public.dim_dates")
        save_to_postgres(fact_df.select(*FACT_COLUMNS), "public.fact_sales")

        logger.info("Star schema loading completed.")
        print_postgres_counts(spark)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()