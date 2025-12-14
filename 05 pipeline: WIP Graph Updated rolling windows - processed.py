# Databricks notebook source
# MAGIC %md
# MAGIC

# COMMAND ----------

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
# MAGIC - signed log scaling
# MAGIC - standard scaling
# MAGIC - idx / ohe
# MAGIC - vector assembler
# MAGIC - down sampling
# MAGIC
# MAGIC FINAL VECTORIZED COLUMNS FOR MODELS: (Includes OHE)
# MAGIC - log_scaled_features
# MAGIC   - this one is for linear / NN
# MAGIC   - numerical is log scaled if skew > 1
# MAGIC   - standard scaler is applied on log scaled numerical
# MAGIC - log_unscaled_features
# MAGIC   - numerical is log scaled if skew > 1
# MAGIC   - no other scaling
# MAGIC - full_unscaled_features (not even log)
# MAGIC   - data is not scaled at all
# MAGIC

# COMMAND ----------

from datetime import date
from dateutil.relativedelta import relativedelta
from typing import List, Tuple

from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
from pyspark.ml.feature import VectorAssembler, StandardScaler, OneHotEncoder, StringIndexer
from pyspark.sql.functions import log1p, when, sign, abs, col
from pyspark.sql.types import DoubleType
import pyspark.sql.functions as F
import pandas as pd
import matplotlib.pyplot as plt
import math
from pyspark.sql.functions import skewness

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
source_path = f"{root}/fasw{time_length}/rolling_windows"

# COMMAND ----------

# DBTITLE 1,DOWNSAMPLE FX
def downsample(train_df, target_ratio=1.0, verbose=False):
    '''Downsamples train_df to balance classes'''
    delay_count = train_df.filter(F.col("ARR_DEL15") == 1).count()
    non_delay_count = train_df.filter(F.col("ARR_DEL15") == 0).count()
    
    keep_percent = (delay_count / non_delay_count) / target_ratio

    if keep_percent >= 1.0:
        print("Warning: Target ratio is already balanced or majority is smaller. No sampling applied.")
        return train_df
    
    train_delay = train_df.filter(F.col('ARR_DEL15') == 1)
    train_non_delay = train_df.filter(F.col('ARR_DEL15') == 0).sample(withReplacement=False, fraction=keep_percent, seed=42)
    train_downsampled = train_delay.union(train_non_delay)
    return train_downsampled

# COMMAND ----------

# DBTITLE 1,SKIP: TESTING DOWNSAMPLE
# MAGIC %skip
# MAGIC train_path = f"{source_path}/window_2_train"
# MAGIC train_df = spark.read.parquet(train_path)
# MAGIC val_path = f"{source_path}/window_2_val"
# MAGIC val_df = spark.read.parquet(val_path)
# MAGIC train_df_new, keep_percent = downsample(train_df)
# MAGIC train_df.count(), train_df_new.count(), val_df.count(), keep_percent

# COMMAND ----------

# DBTITLE 1,DEFINE COLS
# Target Variable
target_col = "ARR_DEL15"

# Numerical Cols (21 total - added 7 new)
num_cols = [
    "CRS_ELAPSED_TIME", "DISTANCE", 
    "ORIGIN_LAT", "ORIGIN_LONG", "ORIGIN_ELEVATION_FT", 
    "DEST_LAT", "DEST_LON", "DEST_ELEVATION_FT", 
    "overall_cloud_frac_0_1", "lowest_cloud_ft", "highest_cloud_ft", 
    "HourlyAltimeterSetting", "HourlyWindGustSpeed", "HourlyWindSpeed",
    # New graph features
    "origin_pagerank", "dest_pagerank", "origin_out_degree", "dest_in_degree",
    # New time-based features
    "prev_flight_arr_delay_clean", "crs_time_to_next_flight_diff_mins", 
    "actual_to_crs_time_to_next_flight_diff_mins_clean"
]

# Alphabetize
num_cols_sorted = sorted(num_cols)

# Cols to sign logscale (if skewed) - add new skewed columns
cols_to_logscale = [
    "CRS_ELAPSED_TIME", "DISTANCE", 
    "ORIGIN_ELEVATION_FT", "DEST_ELEVATION_FT", 
    "lowest_cloud_ft", "highest_cloud_ft", 
    "HourlyWindGustSpeed", "HourlyWindSpeed",
    # New columns that may be skewed
    "origin_out_degree", "dest_in_degree",  # Flight counts (likely skewed)
    "crs_time_to_next_flight_diff_mins", 
    "actual_to_crs_time_to_next_flight_diff_mins_clean"  # Turnaround times (likely skewed)
]

# Numerical (Binary) Columns (15 total - no changes)
binary_cols = [
    "IS_US_HOLIDAY",
    "has_few", "has_sct", "has_bkn", "has_ovc", 
    "light", "heavy", "thunderstorm", "rain_or_drizzle",
    "freezing_conditions", "snow", "hail_or_ice", 
    "reduced_visibility", "spatial_effects", "unknown_precip"
]

# Categorical Columns (15 total - no changes)
cat_cols = [
    "OP_UNIQUE_CARRIER", "ORIGIN", "ORIGIN_STATE_ABR", 
    "DEST", "DEST_STATE_ABR", "ORIGIN_SIZE", "DEST_SIZE", 
    "HourlyWindCardinalDirection"
]

cyclical_cols = [
    "QUARTER", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", 
    "CRS_DEP_TIME_BLOCK", "CRS_ARR_TIME_BLOCK"
]

# COMMAND ----------

# DBTITLE 1,# OF WINDOWS
subdirs = [f for f in dbutils.fs.ls(f"{root}/fasw{time_length}/rolling_windows") if f.isDir()]
windows = len(subdirs)
N = int(windows / 2)
N

# COMMAND ----------

# DBTITLE 1,SKIP
# MAGIC %skip
# MAGIC train_path = f"{root}/fasw{time_length}/rolling_windows/window_1_val"
# MAGIC df = spark.read.parquet(train_path)
# MAGIC display(df)

# COMMAND ----------

# DBTITLE 1,SKIP: PRINT COLS FX
# MAGIC %skip
# MAGIC # Check columns
# MAGIC def print_cols(df, label=""):
# MAGIC     print(f"\n==== {label} ({df.columns.__len__()} columns) ====")
# MAGIC     for c in df.columns:
# MAGIC         print(c)
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,SKIP: QC
# MAGIC %skip
# MAGIC input_path = source_path
# MAGIC
# MAGIC for i in range(1):
# MAGIC     train_path = f"{input_path}/window_{i+1}_train"
# MAGIC     val_path = f"{input_path}/window_{i+1}_val"
# MAGIC     train_df = spark.read.parquet(train_path)
# MAGIC     val_df = spark.read.parquet(val_path)
# MAGIC
# MAGIC     print_cols(train_df, f"Window {i+1} — Loaded train_df")
# MAGIC     print_cols(val_df, f"Window {i+1} — Loaded val_df")

# COMMAND ----------

# DBTITLE 1,TRANSFORM WINDOWS
# Directory / paths
input_path = source_path
output_path = f"{root}/fasw{time_length}/processed_rolling_windows"

dbutils.fs.rm(output_path, True)
dbutils.fs.mkdirs(output_path)
display(f"Windows will be saved to: {output_path}")

# Sampling method
SAMPLING_METHOD = 'down'

### Get rolling window train and val files
for i in range(N):
    train_path = f"{input_path}/window_{i+1}_train"
    val_path = f"{input_path}/window_{i+1}_val"
    train_df = spark.read.parquet(train_path)
    val_df = spark.read.parquet(val_path)

    # Check if skewness is calculable and exceeds the threshold
    skew_row = train_df.agg(
        *[F.skewness(c).alias(c) for c in cols_to_logscale]
    ).first().asDict()

    log_scale = []
    log_output_cols = []
    cols_to_log_transform = []

    for col in cols_to_logscale:
        # Get skew value for the current column
        skew_value = skew_row.get(col) 

        # Check if skewness > 1.0 (highly skewed)
        if skew_value is not None and math.fabs(skew_value) > 1.0:
            new_col_name = f"{col}_log"
            
            # Signed log
            log_scale.append(
                (F.sign(F.col(col)) * F.log1p(F.abs(F.col(col)))).alias(new_col_name)
            )
            cols_to_log_transform.append(col)
            log_output_cols.append(new_col_name)

    if log_scale:
        train_df = train_df.select('*', *log_scale)
        val_df = val_df.select('*', *log_scale)
    
    # Remove cols that needed log transform
    cols_no_logscale = [col for col in num_cols if col not in cols_to_log_transform]

    # print(cols_no_logscale)
    # print(cols_to_log_transform)
    # print(num_cols)
    # print(log_output_cols)
    # print(ohe_output_cols)

    ### One-hot encode categorical columns
    ohe_stages = []
    ohe_output_cols = []
    # high cat cols (tail) causing issues
    for col in cat_cols + cyclical_cols:
        # indexing only applies to categorical
        indexer = StringIndexer(inputCol=col, outputCol=f"{col}_idx", handleInvalid="keep")
        encoder = OneHotEncoder(inputCol=f"{col}_idx", outputCol=f"{col}_ohe")
        ohe_stages.extend([indexer, encoder])
        ohe_output_cols.append(f"{col}_ohe")

    # Final features
    # final_features = cols_no_logscale + log_output_cols + ohe_output_cols + binary_cols #+ hash_output_cols -- this is causing too many issues
    num_features = cols_no_logscale + log_output_cols
    num_features = sorted(num_features)
    cat_features = ohe_output_cols + binary_cols

    for col in binary_cols:
        # Replace nulls with 0.0 and cast to DoubleType
        train_df = train_df.withColumn(col, F.when(F.col(col).isNull(), 0.0).otherwise(F.col(col)).cast(DoubleType()))
        val_df = val_df.withColumn(col, F.when(F.col(col).isNull(), 0.0).otherwise(F.col(col)).cast(DoubleType()))
    
    # 1 Numerical vector assembler
    numerical_assembler = VectorAssembler(
        inputCols=num_features, 
        outputCol="num_vector", 
        handleInvalid="keep"
    )
    
    # 2 Standard scaling ONLY the numerical features
    numerical_scaler = StandardScaler(
        inputCol="num_vector",
        outputCol="scaled_num_vector",
        withStd=True, 
        withMean=True
    )

    # 3 Categorical vector assembler
    categorical_assembler = VectorAssembler(
        inputCols=ohe_output_cols,
        outputCol="ohe_vector",
        handleInvalid="keep"
    )

    # 4a Combined vector merge: scaled numerical + categorical
    scaled_final_assembler = VectorAssembler(
        inputCols=["scaled_num_vector", "ohe_vector"] + binary_cols,
        outputCol="log_scaled_features", # final scaled feature vector
        handleInvalid="keep"
    )

    # 4b Combined vector merge: unscaled numerical + categorical
    unscaled_final_assembler = VectorAssembler(
        inputCols=["num_vector", "ohe_vector"] + binary_cols,
        outputCol="log_unscaled_features", # final unscaled feature vector
        handleInvalid="keep"
    )

    # 4c Combined vector merge: no log, unscaled numerical + categorical
    # Order matters - num_cols_sorted = num_features order (sorted)
    unscaled_final_assembler_no_log = VectorAssembler(
        inputCols= num_cols_sorted + ["ohe_vector"] + binary_cols,
        outputCol="full_unscaled_features", # final unscaled feature vector
        handleInvalid="keep"
    )

    stages = ohe_stages + [numerical_assembler, numerical_scaler, categorical_assembler, scaled_final_assembler, unscaled_final_assembler, unscaled_final_assembler_no_log]

    pipeline = Pipeline(stages=stages)
    pipeline_model = pipeline.fit(train_df)
    train_df = pipeline_model.transform(train_df)
    val_df = pipeline_model.transform(val_df)

    # Downsampling
    original_train_rows = train_df.count()
    train_df = downsample(train_df)
    new_train_rows = train_df.count()

    # Saving files
    train_path = f"{output_path}/window_{i+1}_train"
    val_path = f"{output_path}/window_{i+1}_val"
    dbutils.fs.rm(train_path, True)
    dbutils.fs.rm(val_path, True)
    
    train_df.write.format("parquet").mode("overwrite").save(train_path)
    print(f"\nWindows saved as parquet to: {train_path}. Row count: {new_train_rows} with {new_train_rows/original_train_rows * 100:.2f}% reduction")
    val_df.write.format("parquet").mode("overwrite").save(val_path)
    print(f"\nWindows saved as parquet to: {val_path}. Row count: {val_df.count()}")

# COMMAND ----------

for i in range(N):
    train_path = f"{output_path}/window_{i+1}_train"
    train_df = spark.read.parquet(train_path)
    counts = train_df.groupBy("ARR_DEL15").count().orderBy("ARR_DEL15")
    print(f"\nWindow {i+1} train counts:")
    display(counts)

# COMMAND ----------

# DBTITLE 1,SKIP: CHATGPT FX TO CHECK VECTOR COLS
# MAGIC %skip
# MAGIC # --- Function to extract feature names from VectorUDT metadata (Modified to RETURN the list) ---
# MAGIC def get_vector_feature_names(df, vector_col_name="model_features"):
# MAGIC     """
# MAGIC     Extracts the ordered list of feature names from a VectorUDT column's metadata.
# MAGIC     Returns: list of feature names.
# MAGIC     """
# MAGIC     feature_names = []
# MAGIC     try:
# MAGIC         vector_field = df.schema[vector_col_name]
# MAGIC         metadata = vector_field.metadata
# MAGIC         
# MAGIC         # Extract the list of features
# MAGIC         feature_names.extend(metadata.get('ml_attr', {}).get('attrs', {}).get('numeric', []))
# MAGIC         feature_names.extend(metadata.get('ml_attr', {}).get('attrs', {}).get('binary', []))
# MAGIC         
# MAGIC         # Handle OHE outputs, which are often nested under 'attributes'
# MAGIC         if 'attributes' in metadata.get('ml_attr', {}):
# MAGIC              feature_names.extend([
# MAGIC                  attr['name'] for attr in metadata['ml_attr']['attributes']
# MAGIC                  if 'name' in attr
# MAGIC              ])
# MAGIC
# MAGIC         # Fallback extraction logic for different Spark versions/structures
# MAGIC         if not feature_names and metadata.get('ml_attr', {}).get('attrs', {}):
# MAGIC             for attr_list in metadata['ml_attr']['attrs'].values():
# MAGIC                 feature_names.extend([attr['name'] for attr in attr_list if 'name' in attr])
# MAGIC
# MAGIC         return feature_names
# MAGIC
# MAGIC     except Exception as e:
# MAGIC         print(f"Error extracting metadata for {vector_col_name}: {e}")
# MAGIC         return []
# MAGIC # ---------------------------------------------------------------------------------------------
# MAGIC
# MAGIC # --- NEW HELPER FUNCTION: Maps model indices to feature names ---
# MAGIC def map_indices_to_features(feature_names_list, indices_to_map):
# MAGIC     """
# MAGIC     Maps a list of integer indices (e.g., from feature importance) 
# MAGIC     to the corresponding feature names.
# MAGIC     
# MAGIC     Args:
# MAGIC         feature_names_list (list): The complete ordered list of feature names 
# MAGIC                                    (e.g., from get_vector_feature_names).
# MAGIC         indices_to_map (list): The indices to look up (e.g., [0, 17, 336]).
# MAGIC
# MAGIC     Returns:
# MAGIC         dict: A dictionary mapping index -> feature name.
# MAGIC     """
# MAGIC     mapped_features = {}
# MAGIC     max_index = len(feature_names_list) - 1
# MAGIC     
# MAGIC     print("\n--- Feature Index Mapping ---")
# MAGIC     for index in indices_to_map:
# MAGIC         if 0 <= index <= max_index:
# MAGIC             name = feature_names_list[index]
# MAGIC             mapped_features[index] = name
# MAGIC             print(f"Index {index}: {name}")
# MAGIC         else:
# MAGIC             print(f"Index {index}: ERROR - Index out of bounds (Max Index: {max_index})")
# MAGIC     print("------------------------------\n")
# MAGIC     return mapped_features

# COMMAND ----------

# DBTITLE 1,SKIP: CHATGPT FX TO CHECK VECTOR COLS
# MAGIC %skip
# MAGIC scaled_names = get_vector_feature_names(train_df, "log_scaled_features")
# MAGIC print(f"\n--- Features Assembled in 'model_features' (Total Length: {len(scaled_names)}) ---")
# MAGIC for name in scaled_names:
# MAGIC     print(f"- {name}")
# MAGIC print("--------------------------------------------------------------------------------\n")
# MAGIC
# MAGIC # --- DIAGNOSTIC STEP: Example mapping using the user's specific indices ---
# MAGIC sample_indices_to_map = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 146, 336, 386, 692, 737, 739, 747, 752, 753, 754, 780, 787, 796, 820, 839, 840, 842] # Using a sample of your provided indices
# MAGIC map_indices_to_features(scaled_names, sample_indices_to_map)

# COMMAND ----------

# DBTITLE 1,SKIP: FINAL QC
# MAGIC %skip
# MAGIC train_path = f"{root}/fasw{time_length}/processed_rolling_windows/window_1_train"
# MAGIC df = spark.read.parquet(train_path)
# MAGIC display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Details/Stats
# MAGIC
# MAGIC ###3m:
# MAGIC Windows saved as parquet to: dbfs:/student-groups/Group_02_01/fasw_3m/processed_rolling_windows/window_1_train. Row count: 188556 with 42.04% reduction
# MAGIC <br>
# MAGIC Windows saved as parquet to: dbfs:/student-groups/Group_02_01/fasw_3m/processed_rolling_windows/window_1_val. Row count: 402123
# MAGIC
# MAGIC ##6m:
# MAGIC Windows saved as parquet to: dbfs:/student-groups/Group_02_01/fasw_6m/processed_rolling_windows/window_1_train. Row count: 375192 with 44.11% reduction
# MAGIC <br>
# MAGIC Windows saved as parquet to: dbfs:/student-groups/Group_02_01/fasw_6m/processed_rolling_windows/window_1_val. Row count: 485186
# MAGIC
# MAGIC Windows saved as parquet to: dbfs:/student-groups/Group_02_01/fasw_6m/processed_rolling_windows/window_2_train. Row count: 375362 with 42.30% reduction
# MAGIC <br>
# MAGIC Windows saved as parquet to: dbfs:/student-groups/Group_02_01/fasw_6m/processed_rolling_windows/window_2_val. Row count: 472869
# MAGIC
# MAGIC Windows saved as parquet to: dbfs:/student-groups/Group_02_01/fasw_6m/processed_rolling_windows/window_3_train. Row count: 350805 with 36.62% reduction
# MAGIC <br>
# MAGIC Windows saved as parquet to: dbfs:/student-groups/Group_02_01/fasw_6m/processed_rolling_windows/window_3_val. Row count: 483677
# MAGIC
# MAGIC
# MAGIC ##1y:
# MAGIC Windows saved as parquet to: dbfs:/student-groups/Group_02_01/fasw_1y/processed_rolling_windows/window_1_train. Row count: 653573 with 38.91% reduction
# MAGIC <br>
# MAGIC Windows saved as parquet to: dbfs:/student-groups/Group_02_01/fasw_1y/processed_rolling_windows/window_1_val. Row count: 590763
# MAGIC
# MAGIC Windows saved as parquet to: dbfs:/student-groups/Group_02_01/fasw_1y/processed_rolling_windows/window_2_train. Row count: 662457 with 38.64% reduction
# MAGIC <br>
# MAGIC Windows saved as parquet to: dbfs:/student-groups/Group_02_01/fasw_1y/processed_rolling_windows/window_2_val. Row count: 614930
# MAGIC
# MAGIC Windows saved as parquet to: dbfs:/student-groups/Group_02_01/fasw_1y/processed_rolling_windows/window_3_train. Row count: 676281 with 37.18% reduction
# MAGIC <br>
# MAGIC Windows saved as parquet to: dbfs:/student-groups/Group_02_01/fasw_1y/processed_rolling_windows/window_3_val. Row count: 614521
# MAGIC
# MAGIC Windows saved as parquet to: dbfs:/student-groups/Group_02_01/fasw_1y/processed_rolling_windows/window_4_train. Row count: 767377 with 42.16% reduction
# MAGIC <br>
# MAGIC Windows saved as parquet to: dbfs:/student-groups/Group_02_01/fasw_1y/processed_rolling_windows/window_4_val. Row count: 636861
# MAGIC
# MAGIC Windows saved as parquet to: dbfs:/student-groups/Group_02_01/fasw_1y/processed_rolling_windows/window_5_train. Row count: 820892 with 43.98% reduction
# MAGIC <br>
# MAGIC Windows saved as parquet to: dbfs:/student-groups/Group_02_01/fasw_1y/processed_rolling_windows/window_5_val. Row count: 640936
# MAGIC
# MAGIC Windows saved as parquet to: dbfs:/student-groups/Group_02_01/fasw_1y/processed_rolling_windows/window_6_train. Row count: 835716 with 44.16% reduction
# MAGIC <br>
# MAGIC Windows saved as parquet to: dbfs:/student-groups/Group_02_01/fasw_1y/processed_rolling_windows/window_6_val. Row count: 588730
# MAGIC
# MAGIC
# MAGIC ##5y:
# MAGIC Windows saved as parquet to: dbfs:/student-groups/Group_02_01/fasw/processed_rolling_windows/window_1_train. Row count: 4054196 with 36.15% reduction
# MAGIC <br>
# MAGIC Windows saved as parquet to: dbfs:/student-groups/Group_02_01/fasw/processed_rolling_windows/window_1_val. Row count: 5572019
# MAGIC
# MAGIC Windows saved as parquet to: dbfs:/student-groups/Group_02_01/fasw/processed_rolling_windows/window_2_train. Row count: 3987035 with 35.95% reduction
# MAGIC <br>
# MAGIC Windows saved as parquet to: dbfs:/student-groups/Group_02_01/fasw/processed_rolling_windows/window_2_val. Row count: 7086165

# COMMAND ----------

