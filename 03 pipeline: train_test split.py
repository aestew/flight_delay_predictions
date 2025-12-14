# Databricks notebook source
# MAGIC %md
# MAGIC 3M (2015): Train 1-2 month, Test 3rd month
# MAGIC <br>
# MAGIC 6M (2015): Train 1-5 month, Test 6th month
# MAGIC <br>
# MAGIC 1Y (2019): Train 1-3 quarter, Test 4th quarter
# MAGIC <br>
# MAGIC 5Y (2015-2019): Train 1-4 year, Test 5th year
# MAGIC <br>

# COMMAND ----------

from pyspark.sql.functions import unix_timestamp, to_date, col,col, log1p, when, lit, row_number, hour, dayofweek, sign,unix_timestamp, minute
import pyspark.sql.functions as F
from datetime import date
from dateutil.relativedelta import relativedelta
from typing import Tuple

# COMMAND ----------

## RUN TO BUILD FUNCTION FOR TIMEFRAME

from pyspark.sql import functions as F

root = "dbfs:/student-groups/Group_02_01"

def data_set(time):
    if time == 3:
        return "_3m"
    elif time == 6:
        return "_6m"
    elif time == 12:
        return "_1y"
    elif time == 'all':
        return ""
    else:
        raise ValueError("time must be 3, 6, 12, or 'all'")

#Input dataset duration here
time = 'all' ##"time must be 3, 6, 12, or 'all'"

time_length = data_set(time)
source_path = f"{root}/fasw{time_length}/preprocessed"

# COMMAND ----------

df = spark.read.parquet(source_path)

# COMMAND ----------

# Defines the output structure: (Train Start, Test Start, Test End)

def get_3m_split(start_date: date) -> Tuple[date, date, date]:
    """
    Splits a 3-month period into 2 months Train and 1 month Test
    """
    train_start = start_date
    # Train End (Test Start): 5 months after the start date
    test_start = start_date + relativedelta(months=+2)
    # Test End: 6 months after the start date
    test_end = start_date + relativedelta(months=+3)
    
    return train_start, test_start, test_end

def get_6m_split(start_date: date) -> Tuple[date, date, date]:
    """
    Splits a 6-month period into 5 months Train and 1 month Test
    """
    train_start = start_date
    # Train End (Test Start): 5 months after the start date
    test_start = start_date + relativedelta(months=+5)
    # Test End: 6 months after the start date
    test_end = start_date + relativedelta(months=+6)
    
    return train_start, test_start, test_end


def get_1y_split(start_date: date) -> Tuple[date, date, date]:
    """
    Splits a 1-year period into 9 months Train and 3 months Test
    """
    train_start = start_date
    # Train End (Test Start): 9 months after the start date
    test_start = start_date + relativedelta(months=+9)
    # Test End: 12 months after the start date
    test_end = start_date + relativedelta(years=+1)
    
    return train_start, test_start, test_end


def get_5y_split(start_date: date) -> Tuple[date, date, date]:
    """
    Splits a 5-year period into 4 years Train and 1 year Test
    """
    train_start = start_date
    # Train End (Test Start): 4 years after the start date
    test_start = start_date + relativedelta(years=+4)
    # Test End: 5 years after the start date
    test_end = start_date + relativedelta(years=+5)
    
    return train_start, test_start, test_end

# COMMAND ----------

import pandas as pd
from pyspark.sql import functions as F
from datetime import date
# Assume the split functions (get_6month_split_dates, etc.) are defined above

# Boundaries of split
if time == 3:
    start_date_5yr = date(2015, 1, 1)
    train_start, test_start, test_end = get_3m_split(start_date_5yr)
elif time == 6:
    start_date_5yr = date(2015, 1, 1)
    train_start, test_start, test_end = get_6m_split(start_date_5yr)
elif time == 12:
    start_date_5yr = date(2019, 1, 1)
    train_start, test_start, test_end = get_1y_split(start_date_5yr)
elif time == 'all':
    start_date_5yr = date(2015, 1, 1)
    train_start, test_start, test_end = get_5y_split(start_date_5yr)

# 5Y Output:
# train_start: 2015-01-01
# test_start (Train End): 2019-01-01
# test_end: 2020-01-01

# Training Data Filter: CRS_DEP_DATETIME_UTC >= 2015-01-01 AND CRS_DEP_DATETIME_UTC < 2019-01-01
train = df.filter(
    (F.col("CRS_DEP_DATETIME_UTC") >= train_start) & (F.col("CRS_DEP_DATETIME_UTC") < test_start)
)

# Testing Data Filter: CRS_DEP_DATETIME_UTC >= 2019-01-01 AND CRS_DEP_DATETIME_UTC < 2020-01-01
test = df.filter(
    (F.col("CRS_DEP_DATETIME_UTC") >= test_start) & (F.col("CRS_DEP_DATETIME_UTC") < test_end)
)

print(f"Training set covers: {train_start} to {test_start}")
print(f"Testing set covers: {test_start} to {test_end}")

# COMMAND ----------

# DBTITLE 1,STATS
def minmax(df):
    return df.agg(
        F.min("CRS_DEP_DATETIME_UTC").alias("min_time"),
        F.max("CRS_DEP_DATETIME_UTC").alias("max_time")
    ).show(truncate=False)

print("TRAIN:")
minmax(train)

print("TEST:")
minmax(test)

print(f"Train size: {train.count()} rows")
print(f"Test size: {test.count()} rows")

train_count = train.count()
test_count  = test.count()

total = train_count + test_count
print(f"\ntrain %: {train_count/total:.3f}")
print(f"test %:  {test_count/total:.3f}")

# COMMAND ----------

save_path = f"{root}/fasw{time_length}/train_test"

def write_splits(train, test, save_path):
    dbutils.fs.rm(save_path, recurse=True)
    
    (
        train.withColumn("split", F.lit("train"))
        .unionByName(test.withColumn("split", F.lit("test")))
        .write
        .format("parquet")      # parquet output
        .mode("overwrite")
        .partitionBy("split")   # creates split=train/ and split=test/
        .save(save_path)
    )
    
    print(f"Saved train/test at: {save_path}")


# COMMAND ----------

write_splits(train, test, save_path)