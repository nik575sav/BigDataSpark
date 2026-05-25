import logging
import os

import clickhouse_connect
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.functions import broadcast

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "bigdata_lab")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "postgres")

CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST", "clickhouse")
CLICKHOUSE_HTTP_PORT = int(os.getenv("CLICKHOUSE_HTTP_PORT", "8123"))
CLICKHOUSE_JDBC_PORT = int(os.getenv("CLICKHOUSE_JDBC_PORT", "8123"))
CLICKHOUSE_DB = os.getenv("CLICKHOUSE_DB", "marts")
CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "default")
CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "clickhouse")

SPARK_JARS = os.getenv(
    "SPARK_JARS",
    "/opt/spark/jars/postgresql-42.7.10.jar,"
    "/opt/spark/jars/clickhouse-jdbc-0.7.1.jar",
)

POSTGRES_JDBC_URL = f"jdbc:postgresql://{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
POSTGRES_JDBC_PROPS = {
    "user": POSTGRES_USER,
    "password": POSTGRES_PASSWORD,
    "driver": "org.postgresql.Driver",
}

CLICKHOUSE_JDBC_URL = f"jdbc:clickhouse://{CLICKHOUSE_HOST}:{CLICKHOUSE_JDBC_PORT}/{CLICKHOUSE_DB}"
CLICKHOUSE_JDBC_PROPS = {
    "user": CLICKHOUSE_USER,
    "password": CLICKHOUSE_PASSWORD,
    "driver": "com.clickhouse.jdbc.ClickHouseDriver",
}


def build_spark() -> SparkSession:
    builder = (
        SparkSession.builder
        .appName("bigdata-lab-clickhouse-marts")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.jars.ivy", "/tmp/.ivy2")
        .config("spark.jars", SPARK_JARS)
        .config("spark.driver.extraClassPath", SPARK_JARS.replace(",", ":"))
        .config("spark.executor.extraClassPath", SPARK_JARS.replace(",", ":"))
    )
    return builder.getOrCreate()


def get_clickhouse_client():
    return clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_HTTP_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DB,
    )


def prepare_clickhouse() -> None:
    client = get_clickhouse_client()
    try:
        client.command(f"CREATE DATABASE IF NOT EXISTS {CLICKHOUSE_DB}")

        ddl_statements = [
            f"""
            CREATE TABLE IF NOT EXISTS {CLICKHOUSE_DB}.top_products_by_revenue (
                product_key UInt64,
                product_name Nullable(String),
                product_category Nullable(String),
                total_orders UInt64,
                total_units_sold Int64,
                total_revenue Float64,
                avg_rating Nullable(Float64),
                total_reviews Nullable(Int64)
            ) ENGINE = MergeTree() ORDER BY (product_key)
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {CLICKHOUSE_DB}.best_customers (
                customer_key UInt64,
                customer_name Nullable(String),
                customer_country Nullable(String),
                total_orders UInt64,
                total_revenue Float64,
                avg_check Float64
            ) ENGINE = MergeTree() ORDER BY (customer_key)
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {CLICKHOUSE_DB}.monthly_sales_trends (
                year_num Int32,
                quarter_num Int32,
                month_num Int32,
                month_name Nullable(String),
                total_orders UInt64,
                total_units_sold Int64,
                total_revenue Float64,
                avg_order_value Float64
            ) ENGINE = MergeTree() ORDER BY (year_num, month_num)
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {CLICKHOUSE_DB}.store_performance (
                store_key UInt64,
                store_name Nullable(String),
                store_city Nullable(String),
                store_country Nullable(String),
                total_orders UInt64,
                total_revenue Float64,
                avg_check Float64
            ) ENGINE = MergeTree() ORDER BY (store_key)
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {CLICKHOUSE_DB}.supplier_analysis (
                supplier_key UInt64,
                supplier_name Nullable(String),
                supplier_country Nullable(String),
                total_orders UInt64,
                total_revenue Float64,
                avg_product_price Nullable(Float64),
                total_units_sold Int64
            ) ENGINE = MergeTree() ORDER BY (supplier_key)
            """,
            f"""
            CREATE TABLE IF NOT EXISTS {CLICKHOUSE_DB}.product_rating_insights (
                product_key UInt64,
                product_name Nullable(String),
                product_rating Nullable(Float64),
                product_reviews Nullable(Int64),
                total_units_sold Int64,
                total_orders UInt64,
                total_revenue Float64,
                revenue_per_review Nullable(Float64),
                rating_rank String   -- 'highest' or 'lowest'
            ) ENGINE = MergeTree() ORDER BY (product_key)
            """,
        ]
        for ddl in ddl_statements:
            client.command(ddl)

        # Очистка таблиц
        for table in [
            "top_products_by_revenue", "best_customers", "monthly_sales_trends",
            "store_performance", "supplier_analysis", "product_rating_insights"
        ]:
            client.command(f"TRUNCATE TABLE IF EXISTS {CLICKHOUSE_DB}.{table}")
    finally:
        client.close()


def read_pg_table(spark: SparkSession, table_name: str):
    return spark.read.jdbc(url=POSTGRES_JDBC_URL, table=table_name, properties=POSTGRES_JDBC_PROPS)


def build_enriched_sales_df(spark: SparkSession):
    fact = read_pg_table(spark, "public.fact_sales")
    cust = read_pg_table(spark, "public.dim_customers")
    sell = read_pg_table(spark, "public.dim_sellers")
    prod = read_pg_table(spark, "public.dim_products")
    store = read_pg_table(spark, "public.dim_stores")
    supp = read_pg_table(spark, "public.dim_suppliers")
    date = read_pg_table(spark, "public.dim_dates")

    enriched = (
        fact.alias("f")
        .join(broadcast(cust.alias("c")), on="customer_key", how="left")
        .join(broadcast(sell.alias("s")), on="seller_key", how="left")
        .join(broadcast(prod.alias("p")), on="product_key", how="left")
        .join(broadcast(store.alias("st")), on="store_key", how="left")
        .join(broadcast(supp.alias("sup")), on="supplier_key", how="left")
        .join(broadcast(date.alias("d")), on="date_key", how="left")
        .withColumn("customer_name", F.concat_ws(" ", F.col("customer_first_name"), F.col("customer_last_name")))
        .withColumn("seller_name", F.concat_ws(" ", F.col("seller_first_name"), F.col("seller_last_name")))
        .withColumn("sale_total_price", F.round(F.col("sale_total_price"), 2))
        .withColumn("product_price", F.round(F.col("product_price"), 2))
        .withColumn("product_rating", F.round(F.col("product_rating"), 2))
        .withColumn("sale_quantity", F.col("sale_quantity").cast("long"))
        .withColumn("product_reviews", F.col("product_reviews").cast("long"))
        .withColumn("year_num", F.col("year_num").cast("int"))
        .withColumn("quarter_num", F.col("quarter_num").cast("int"))
        .withColumn("month_num", F.col("month_num").cast("int"))
    )
    return enriched


def top_products_by_revenue(df):
    return (
        df.groupBy("product_key", "product_name", "product_category")
        .agg(
            F.countDistinct("source_sale_id").alias("total_orders"),
            F.sum("sale_quantity").alias("total_units_sold"),
            F.round(F.sum("sale_total_price"), 2).alias("total_revenue"),
            F.round(F.avg("product_rating"), 2).alias("avg_rating"),
            F.max("product_reviews").alias("total_reviews"),
        )
        .orderBy(F.desc("total_revenue"), F.desc("total_units_sold"))
        .limit(10)
    )


def best_customers(df):
    return (
        df.groupBy("customer_key", "customer_name", "customer_country")
        .agg(
            F.countDistinct("source_sale_id").alias("total_orders"),
            F.round(F.sum("sale_total_price"), 2).alias("total_revenue"),
            F.round(F.avg("sale_total_price"), 2).alias("avg_check"),
        )
        .orderBy(F.desc("total_revenue"), F.desc("total_orders"))
        .limit(10)
    )


def monthly_sales_trends(df):
    return (
        df.groupBy("year_num", "quarter_num", "month_num", "month_name")
        .agg(
            F.countDistinct("source_sale_id").alias("total_orders"),
            F.sum("sale_quantity").alias("total_units_sold"),
            F.round(F.sum("sale_total_price"), 2).alias("total_revenue"),
            F.round(F.avg("sale_total_price"), 2).alias("avg_order_value"),
        )
        .orderBy("year_num", "month_num")
    )


def store_performance(df):
    return (
        df.groupBy("store_key", "store_name", "store_city", "store_country")
        .agg(
            F.countDistinct("source_sale_id").alias("total_orders"),
            F.round(F.sum("sale_total_price"), 2).alias("total_revenue"),
            F.round(F.avg("sale_total_price"), 2).alias("avg_check"),
        )
        .orderBy(F.desc("total_revenue"), F.desc("total_orders"))
        .limit(5)
    )


def supplier_analysis(df):
    return (
        df.groupBy("supplier_key", "supplier_name", "supplier_country")
        .agg(
            F.countDistinct("source_sale_id").alias("total_orders"),
            F.round(F.sum("sale_total_price"), 2).alias("total_revenue"),
            F.round(F.avg("product_price"), 2).alias("avg_product_price"),
            F.sum("sale_quantity").alias("total_units_sold"),
        )
        .orderBy(F.desc("total_revenue"), F.desc("total_orders"))
        .limit(5)
    )


def product_rating_insights(df):
    # Агрегация на уровне продукта
    product_agg = (
        df.groupBy("product_key", "product_name")
        .agg(
            F.round(F.avg("product_rating"), 2).alias("product_rating"),
            F.max("product_reviews").alias("product_reviews"),
            F.sum("sale_quantity").alias("total_units_sold"),
            F.countDistinct("source_sale_id").alias("total_orders"),
            F.round(F.sum("sale_total_price"), 2).alias("total_revenue"),
        )
        .withColumn(
            "revenue_per_review",
            F.when(F.col("product_reviews") > 0,
                   F.round(F.col("total_revenue") / F.col("product_reviews"), 2))
             .otherwise(F.lit(None))
        )
    )

    # Продукт с наивысшим рейтингом
    highest_rating = product_agg.orderBy(F.desc("product_rating")).limit(1)
    highest_rating = highest_rating.withColumn("rating_rank", F.lit("highest"))

    # Продукт с наинизшим рейтингом (игнорируем NULL)
    lowest_rating = product_agg.filter(F.col("product_rating").isNotNull()) \
                               .orderBy(F.asc("product_rating")).limit(1)
    lowest_rating = lowest_rating.withColumn("rating_rank", F.lit("lowest"))

    # Объединяем
    result = highest_rating.unionByName(lowest_rating)
    return result


def write_clickhouse_table(df, table_name: str) -> None:
    if df.count() == 0:
        logger.warning(f"DataFrame for {table_name} is empty, skipping write")
        return
    df.write.mode("append").jdbc(
        url=CLICKHOUSE_JDBC_URL,
        table=table_name,
        properties=CLICKHOUSE_JDBC_PROPS,
    )
    logger.info(f"Written {df.count()} rows to {table_name}")


def print_clickhouse_counts():
    client = get_clickhouse_client()
    try:
        for table in [
            "top_products_by_revenue", "best_customers", "monthly_sales_trends",
            "store_performance", "supplier_analysis", "product_rating_insights"
        ]:
            cnt = client.query(f"SELECT count() AS cnt FROM {CLICKHOUSE_DB}.{table}").first_row[0]
            logger.info(f"{table}: {cnt} rows")
    finally:
        client.close()


def main():
    logger.info("=== ETL: PostgreSQL star schema -> ClickHouse marts ===")
    prepare_clickhouse()

    spark = build_spark()
    try:
        enriched = build_enriched_sales_df(spark)

        marts = {
            "top_products_by_revenue": top_products_by_revenue(enriched),
            "best_customers": best_customers(enriched),
            "monthly_sales_trends": monthly_sales_trends(enriched),
            "store_performance": store_performance(enriched),
            "supplier_analysis": supplier_analysis(enriched),
            "product_rating_insights": product_rating_insights(enriched),
        }

        for table_name, df in marts.items():
            write_clickhouse_table(df, table_name)

        print_clickhouse_counts()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
