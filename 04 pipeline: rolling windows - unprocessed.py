# Databricks notebook source
# MAGIC %md
# MAGIC %md
# MAGIC Rolling windows:
# MAGIC <br>
# MAGIC 2015 6 months (4 windows): train 1-2 months, 3rd month val
# MAGIC <br>
# MAGIC 2019 12 months (8 windows): train 1-4 months, 5th month val
# MAGIC <br>
# MAGIC 2015-2019 5 y (3 windows): train 1-2 years, val 3rd year
# MAGIC
# MAGIC -- Using CRS_DEP_DATETIME_UTC for splits to prevent "predicting past using future"
# MAGIC

# COMMAND ----------

from datetime import date
from dateutil.relativedelta import relativedelta
from typing import List, Tuple

from pyspark.sql import functions as F

# COMMAND ----------

## RUN TO BUILD FUNCTION FOR TIMEFRAME

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
time = 3 ##"time must be 3, 6, 12, or 'all'"

time_length = data_set(time)
source_path = f"{root}/fasw{time_length}/train_test/split=train"

# COMMAND ----------

df = spark.read.parquet(source_path)

# COMMAND ----------

# Start and end dates of train
agg_vals = df.agg(
    F.min(F.to_date("CRS_DEP_DATETIME_UTC")).alias("min_date"),
    F.max(F.to_date("CRS_DEP_DATETIME_UTC")).alias("max_date")
).first()

train_start = agg_vals["min_date"]
train_end = agg_vals["max_date"]

train_start, train_end

# COMMAND ----------

def get_rolling_windows(
train_start: date, 
    train_end: date, 
    cv_train_window: int, # Sub-training window
    cv_val_window: int,   # Validation window
    roll_step: int        # Advance the window each fold
) -> List[Tuple[date, date, date]]:
    """
    Generates rolling (subtrain_start, val_start, val_end) date tuples 
    within the primary training period (train_start to train_end).
    """
    splits = []
    current_val_start = train_start + relativedelta(months=cv_train_window)
    max_val_end = train_end + relativedelta(days=1)
    
    while True:
        val_end = current_val_start + relativedelta(months=cv_val_window)
        
        # Stop condition: If the Validation End Date is after the overall Train End Date, stop.
        if val_end > max_val_end:
            break
            
        subtrain_start = current_val_start - relativedelta(months=cv_train_window)
        val_start = current_val_start
            
        splits.append((subtrain_start, val_start, val_end))
        
        # Advance the window by the roll step
        current_val_start += relativedelta(months=roll_step)

    return splits


# COMMAND ----------

# Define the Rolling CV Parameters
if time == 3:
    cv_train = 1
    cv_val = 1
    rolling_step = 1
elif time == 6:
    cv_train = 2
    cv_val = 1
    rolling_step = 1
elif time == 12:
    cv_train = 3
    cv_val = 1
    rolling_step = 1
else: # 5Y
    cv_train = 24       # 2 years
    cv_val = 12         # 1 year
    rolling_step = 12   # Advance 1 year per fold

cv_rolling_windows = get_rolling_windows(
    train_start=train_start,    # Overall train start date
    train_end=train_end,        # Overall train end date
    cv_train_window=cv_train,
    cv_val_window=cv_val,
    roll_step=rolling_step
)

print(f"Generated {len(cv_rolling_windows)} CV Rolling Windows:")

save_path = f"{root}/fasw{time_length}/rolling_windows"
dbutils.fs.rm(save_path, recurse=True)

for i, (subtrain_start, val_start, val_end) in enumerate(cv_rolling_windows):
    print(f"\n--- Window {i+1} ---")
    print(f"Sub-Train Set: {subtrain_start} to {val_start}")
    print(f"Validation Set: {val_start} to {val_end}")

    subtrain_df = df.filter(
        (F.col("CRS_DEP_DATETIME_UTC") >= subtrain_start) & 
        (F.col("CRS_DEP_DATETIME_UTC") < val_start)
    )
    val_df = df.filter(
        (F.col("CRS_DEP_DATETIME_UTC") >= val_start) & 
        (F.col("CRS_DEP_DATETIME_UTC") < val_end)
    )

    subtrain_df.write.mode("overwrite").parquet(f"{save_path}/window_{i+1}_train")
    val_df.write.mode("overwrite").parquet(f"{save_path}/window_{i+1}_val")
    print(f"Saved train/test at: {save_path}")

# COMMAND ----------

display(val_df)

# COMMAND ----------

