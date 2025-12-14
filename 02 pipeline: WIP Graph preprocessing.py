# Databricks notebook source
# MAGIC %md
# MAGIC Notes:
# MAGIC - To handle datatype conversions + custom null handling
# MAGIC - Takes care of all nulls; no imputation needed at this time
# MAGIC - Further restrict the dataframe columns
# MAGIC
# MAGIC **Add Time-based feature(s) and Graph feature(s) here as an extra column!**

# COMMAND ----------

# DBTITLE 1,IMPORT PKG
from pyspark.sql.types import IntegerType, DoubleType, StringType
from pyspark.sql.functions import unix_timestamp, to_date, col,col, log1p, when, lit, row_number, hour, dayofweek, sign,unix_timestamp, minute
import pyspark.sql.functions as F
from pyspark.ml.feature import Imputer
from pyspark.sql.window import Window
import pandas as pd

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
source_path = f"{root}/fasw{time_length}/final_join"

# COMMAND ----------

# DBTITLE 1,READ DATA
df = spark.read.format("parquet").load(source_path)
print(f"Data loaded successfully from {source_path}. Initial row count: {df.count()}")

# COMMAND ----------

display(df)

# COMMAND ----------

# DBTITLE 1,AIRPORT-AIRPORT GRAPH FEATURES
def create_airport_graph_features(df):
    import networkx as nx
    from pyspark.sql import functions as F
    
    # Build directed graph using ORIGIN and DEST
    G = nx.DiGraph()
    
    # PySpark version: use count() instead of size()
    route_counts = df.groupBy('ORIGIN', 'DEST').count().withColumnRenamed('count', 'weight')
    
    # Collect to pandas for NetworkX processing (NetworkX needs local data)
    route_counts_pd = route_counts.toPandas()
    
    for _, row in route_counts_pd.iterrows():
        G.add_edge(row['ORIGIN'], row['DEST'], weight=row['weight'])
    
    # Calculate metrics
    pagerank = nx.pagerank(G, weight='weight')
    in_degree = dict(G.in_degree(weight='weight'))
    out_degree = dict(G.out_degree(weight='weight'))
    
    # Convert dictionaries to DataFrames for joining
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType
    
    # Create mapping DataFrames
    pagerank_data = [(k, v) for k, v in pagerank.items()]
    in_degree_data = [(k, float(v)) for k, v in in_degree.items()]
    out_degree_data = [(k, float(v)) for k, v in out_degree.items()]
    
    schema = StructType([
        StructField("airport", StringType(), True),
        StructField("value", DoubleType(), True)
    ])
    
    pagerank_df = spark.createDataFrame(pagerank_data, schema)
    in_degree_df = spark.createDataFrame(in_degree_data, schema)
    out_degree_df = spark.createDataFrame(out_degree_data, schema)
    
    # Join back to original dataframe
    df = df.join(pagerank_df.withColumnRenamed('airport', 'ORIGIN')
                            .withColumnRenamed('value', 'origin_pagerank'),
                 on='ORIGIN', how='left')
    
    df = df.join(pagerank_df.withColumnRenamed('airport', 'DEST')
                            .withColumnRenamed('value', 'dest_pagerank'),
                 on='DEST', how='left')
    
    df = df.join(out_degree_df.withColumnRenamed('airport', 'ORIGIN')
                              .withColumnRenamed('value', 'origin_out_degree'),
                 on='ORIGIN', how='left')
    
    df = df.join(in_degree_df.withColumnRenamed('airport', 'DEST')
                             .withColumnRenamed('value', 'dest_in_degree'),
                 on='DEST', how='left')
    
    return df

# COMMAND ----------

# DBTITLE 1,TIME BASED FEATURES - NO LEAKAGE
def create_time_based_features_no_leakage(df):
    from pyspark.sql import Window
    from pyspark.sql import functions as F
    
    prediction_offset_seconds = 3661  # 1h 1m
    
    # Calculate prediction time for each flight (unix timestamp)
    df = df.withColumn('crs_dep_offset_utc_unix', 
                       F.unix_timestamp('CRS_DEP_DATETIME_UTC') - prediction_offset_seconds)
    
    # === LAG FEATURES (FIXED FOR LEAKAGE) ===
    window_tail = Window.partitionBy('TAIL_NUM').orderBy('CRS_DEP_DATETIME_UTC')
    
    # Get previous flight's data
    # Actual arrival time - need to protect from leakage
    df = df.withColumn('prev_arr_delay_min', 
                       F.lag('ARR_DELAY', 1).over(window_tail))
    df = df.withColumn('prev_arr_time_utc_unix', 
                       F.lag(F.unix_timestamp('ARR_DATETIME_UTC'), 1).over(window_tail))
    df = df.withColumn('prev_arr_time_utc',
                       F.lag(F.col('ARR_DATETIME_UTC'), 1).over(window_tail))  
    
    # CRS_planned_arrival_time
    df = df.withColumn('prev_crs_arr_time_utc', 
                       F.lag(F.col('CRS_ARR_DATETIME_UTC'), 1).over(window_tail))
    df = df.withColumn('prev_crs_arr_time_utc_unix', 
                       F.lag(F.unix_timestamp('CRS_ARR_DATETIME_UTC'), 1).over(window_tail))
   
    # CRITICAL: Only use previous flight if it arrived BEFORE prediction time
    df = df.withColumn('prev_flight_arr_delay_clean',
                       F.when((F.col('prev_arr_time_utc_unix') < F.col('crs_dep_offset_utc_unix')) & 
                              (F.col('prev_arr_delay_min') >= 0),
                              F.col('prev_arr_delay_min')).otherwise(None))

    ## Calculate New Columns
    
    # CRS Time difference in MINUTES between current departure and previous arrival
    df = df.withColumn('crs_time_to_next_flight_diff_mins',
                       (F.unix_timestamp('CRS_DEP_DATETIME_UTC') - F.col('prev_crs_arr_time_utc_unix')) / 60)
    
    # ACTUAL Time difference in MINUTES between current departure and previous arrival (prevent data leakage)
    df = df.withColumn('actual_to_crs_time_to_next_flight_diff_mins_clean',
                       F.when(F.col('prev_arr_time_utc_unix') < F.col('crs_dep_offset_utc_unix'),
                              (F.unix_timestamp('CRS_DEP_DATETIME_UTC') - F.col('prev_arr_time_utc_unix')) / 60).otherwise(None))
    
    # Leakage in hours
    df = df.withColumn('leakage',
                       (F.unix_timestamp('CRS_DEP_DATETIME_UTC') - F.col('prev_arr_time_utc_unix')) / 3600)
    
    # Drop intermediate columns
    df = df.drop('prev_arr_delay_min', 'prev_arr_time_utc','prev_arr_time_utc_unix', 'crs_dep_offset_utc_unix')
    
    return dfoh 

# COMMAND ----------

df_new_cols = df

# COMMAND ----------

# DBTITLE 1,ADD COLS FOR GRAPH TIME BASED + LEAKAGE CHECK
# === Create Final DF with All Features ===
df_graph = create_airport_graph_features(df_new_cols)
df_final = create_time_based_features_no_leakage(df_graph)

# === LEAKAGE CHECK ===
# QC Check: Verify that when leakage < 1h1m, the features are null
leakage_threshold_hours = 3661 / 3600  # 1h1m in hours

leakage_violations = df_final.select(
    F.sum(F.when(
        (F.col('leakage') < leakage_threshold_hours) & 
        (F.col('prev_flight_arr_delay_clean').isNotNull()),
        1
    ).otherwise(0)).alias('prev_delay_violations'),
    
    F.sum(F.when(
        (F.col('leakage') < leakage_threshold_hours) & 
        (F.col('actual_to_crs_time_to_next_flight_diff_mins_clean').isNotNull()),
        1
    ).otherwise(0)).alias('actual_time_violations'),
    
    F.sum(F.when(F.col('leakage') < leakage_threshold_hours, 1).otherwise(0)).alias('total_leakage_risk_rows'),
    F.count('*').alias('total_rows')
).collect()[0]

# print("\n=== Leakage Violation Check ===")
# print(f"Total rows: {leakage_violations['total_rows']:,}")
# print(f"Rows potential with leakage risk < 1h1m: {leakage_violations['total_leakage_risk_rows']:,}")
# print(f"\nViolations (leakage < 1h1m BUT feature not null):")
# print(f"  prev_flight_arr_delay_clean not null: {leakage_violations['prev_delay_violations']:,}")
# print(f"  actual_to_crs_time_to_next_flight_diff_mins not null: {leakage_violations['actual_time_violations']:,}")

if leakage_violations['prev_delay_violations'] > 0 or leakage_violations['actual_time_violations'] > 0:
    print(f"\n⚠️  WARNING: Found {leakage_violations['prev_delay_violations'] + leakage_violations['actual_time_violations']:,} total violations!")
else:
    print(f"\n✅ No leakage violations! All features properly nullified when leakage < 1h1m")

# Drop leakage column
df_final = df_final.drop('leakage','prev_crs_arr_time_utc_unix')

# === Null Count for New Features ===
# new_features = [
#     'origin_pagerank', 
#     'dest_pagerank', 
#     'origin_out_degree', 
#     'dest_in_degree',
#     'prev_flight_arr_delay_clean',
#     'crs_time_to_next_flight_diff_mins',
#     'prev_crs_arr_time_utc',
#     'actual_to_crs_time_to_next_flight_diff_mins_clean'
# ]

# print("\n=== Null Counts for New Features ===")
# null_counts = df_final.select([
#     F.sum(F.col(c).isNull().cast('int')).alias(c) 
#     for c in new_features
# ]).collect()[0]

# total_rows = df_final.count()
# print(f"Total rows: {total_rows:,}\n")

# for col_name in new_features:
#     null_count = null_counts[col_name]
#     null_pct = (null_count / total_rows) * 100
#     print(f"{col_name:50s}: {null_count:10,} ({null_count/total_rows*100:5.1f}%)")

df=df_final

print("\n✅ df_final ready for downstream processing")

# COMMAND ----------

# DBTITLE 1,NULL COLS
# MAGIC %skip
# MAGIC null_counts_df = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns])
# MAGIC vertical_null_counts = null_counts_df.toPandas().T.reset_index()
# MAGIC vertical_null_counts.columns = ['column', 'null_count']
# MAGIC display(vertical_null_counts)

# COMMAND ----------

# DBTITLE 1,MANUAL NULL HANDLING FUNCTION
def clean_nulls(df):

    # Null columns
    null_counts_pdf = (
        df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns])
          .toPandas()
          .T.rename(columns={0:"null_count"})
          .sort_values("null_count", ascending=False)
    )
    cols_with_nulls = [c for c, n in null_counts_pdf["null_count"].items() if n > 0]

    # Null stats
    null_cols = [
        # Delay decomposition
        "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
        # Cloud features
        "overall_cloud_frac_0_1", "lowest_cloud_ft", "highest_cloud_ft", "has_few", "has_sct", "has_bkn", "has_ovc",
        # Wind and atmospheric features
        "HourlyWindDirection", "HourlyWindSpeed", "HourlyWindGustSpeed", "HourlyAltimeterSetting",
        # Weather station metadata
        "WX_STATION", "MATCHED_OBS_TIME_UTC", "LATITUDE", "LONGITUDE", "ELEVATION", "NAME", "REPORT_TYPE", "SOURCE",
        # Weather conditions
        "light", "heavy", "thunderstorm", "rain_or_drizzle", "freezing_conditions", "snow", "hail_or_ice",
        "reduced_visibility", "spatial_effects", "unknown_precip"
    ]

    missing_expected = sorted(set(null_cols) - set(df.columns))
    unexpected_nulls = sorted(set(cols_with_nulls) - set(null_cols))
    missing_nulls_from_expected = sorted(set(null_cols) - set(cols_with_nulls))

    # Fill with 0s
    zero_fill_cols = [
        # Delay decomposition
        "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY",
        # Cloud features
        "overall_cloud_frac_0_1", "lowest_cloud_ft", "highest_cloud_ft", "has_few", "has_sct", "has_bkn", "has_ovc",       
        # Wind features
        "HourlyWindDirection", "HourlyWindSpeed", "HourlyWindGustSpeed",    
        # Weather conditions
        "light", "heavy", "thunderstorm", "rain_or_drizzle", "freezing_conditions", "snow", "hail_or_ice", 
        "reduced_visibility", "spatial_effects", "unknown_precip",       
        # Graph features
        "origin_pagerank", "dest_pagerank", "origin_out_degree", "dest_in_degree",
        # Time-based lag features
        "prev_flight_arr_delay_clean", "actual_to_crs_time_to_next_flight_diff_mins_clean"
    ]

    zero_fill_cols = [c for c in zero_fill_cols if c in df.columns]
    if zero_fill_cols:
        df = df.fillna({c: 0 for c in zero_fill_cols})
    
    # Fill with Median
    median_col = "crs_time_to_next_flight_diff_mins"
    if median_col in df.columns:
        median_value = df.approxQuantile(median_col, [0.5], 0.01)[0]
        df = df.fillna({median_col: median_value})

    # HourlyAltimeterSetting Standard = 29.92
    if "HourlyAltimeterSetting" in df.columns:
        df = df.fillna({"HourlyAltimeterSetting": 29.92})

    # Remaining null columns
    post_null_counts_pdf = (
        df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns])
          .toPandas()
          .T.rename(columns={0:"null_count"})
          .sort_values("null_count", ascending=False)
    )
    remaining_null_cols = [c for c, n in post_null_counts_pdf["null_count"].items() if n > 0]

    print("\n\nRemaining columns with nulls:", remaining_null_cols)

    return df, remaining_null_cols, post_null_counts_pdf

# COMMAND ----------

# MAGIC %skip
# MAGIC display(df)

# COMMAND ----------

# DBTITLE 1,Columns and datatypes
# MAGIC %skip
# MAGIC display(pd.DataFrame({'column': df.columns, 'datatype': [dtype for _, dtype in df.dtypes]}))

# COMMAND ----------

# DBTITLE 1,DATATYPES AND SELECTED COLS
# Define key columns
timestamp_col = 'CRS_DEP_DATETIME_UTC'
target_column = 'ARR_DEL15'

# Columns to convert to correct datatype
convert_to_double = ["ORIGIN_LAT","ORIGIN_LONG","HourlyAltimeterSetting","DEST_LAT","DEST_LON"]
convert_to_int = ["ARR_DEL15","CARRIER_DELAY","NAS_DELAY","SECURITY_DELAY","LATE_AIRCRAFT_DELAY","HourlyWindGustSpeed","HourlyWindSpeed","HourlyWindDirection"]

# Convert Wind Direction to Cardinal Directions

# Reference the wind direction column

wind_direction_category = (
    F.when(F.col("HourlyWindDirection").isNull() | (F.col("HourlyWindDirection") == 0) | (F.col("HourlyWindDirection") == 360), "Calm")
    .when((F.col("HourlyWindDirection") >= 337.5) | (F.col("HourlyWindDirection") < 22.5), "N")
    .when((F.col("HourlyWindDirection") >= 22.5) & (F.col("HourlyWindDirection") < 67.5), "NE")
    .when((F.col("HourlyWindDirection") >= 67.5) & (F.col("HourlyWindDirection") < 112.5), "E")
    .when((F.col("HourlyWindDirection") >= 112.5) & (F.col("HourlyWindDirection") < 157.5), "SE")
    .when((F.col("HourlyWindDirection") >= 157.5) & (F.col("HourlyWindDirection") < 202.5), "S")
    .when((F.col("HourlyWindDirection") >= 202.5) & (F.col("HourlyWindDirection") < 247.5), "SW")
    .when((F.col("HourlyWindDirection") >= 247.5) & (F.col("HourlyWindDirection") < 292.5), "W")
    .when((F.col("HourlyWindDirection") >= 292.5) & (F.col("HourlyWindDirection") < 337.5), "NW")
    .otherwise("OTHER")
)

df = df.withColumn("HourlyWindCardinalDirection", wind_direction_category)

# Create CRS_DEP_BLOCK in 24 hour cycle (e.g., 134 -> '0100-0159')
df = df \
    .withColumn("CRS_DEP_TIME_BLOCK",
    F.format_string(
        "%02d00-%02d59",
        (F.col("CRS_DEP_TIME") / 100).cast("int"),
        (F.col("CRS_DEP_TIME") / 100).cast("int")
        )
    ) \
    .withColumn("CRS_ARR_TIME_BLOCK",
    F.format_string(
        "%02d00-%02d59",
        (F.col("CRS_ARR_TIME") / 100).cast("int"),
        (F.col("CRS_ARR_TIME") / 100).cast("int")
        )
    )

# Final set for models
### Need CRS_DEP_DATETIME_UTC for train-test splitting
final_cols = [
    'YEAR', 'QUARTER', 'MONTH', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'IS_US_HOLIDAY','OP_UNIQUE_CARRIER', 'TAIL_NUM', 'ORIGIN', 'ORIGIN_STATE_ABR', 'DEST', 'DEST_STATE_ABR', 'CRS_DEP_TIME_BLOCK', 'CRS_DEP_DATETIME_UTC', 'CRS_ARR_TIME_BLOCK', 'ARR_DEL15', 'CRS_ELAPSED_TIME', 'CARRIER_DELAY', 'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'DISTANCE', 'ORIGIN_LAT', 'ORIGIN_LONG', 'ORIGIN_ELEVATION_FT', 'ORIGIN_SIZE', 'DEST_LAT', 'DEST_LON', 'DEST_ELEVATION_FT', 'DEST_SIZE', 'overall_cloud_frac_0_1', 'lowest_cloud_ft', 'highest_cloud_ft', 'has_few', 'has_sct', 'has_bkn', 'has_ovc', 'HourlyAltimeterSetting', 'HourlyWindCardinalDirection', 'HourlyWindGustSpeed', 'HourlyWindSpeed', 'light', 'heavy', 'thunderstorm', 'rain_or_drizzle', 'freezing_conditions', 'snow', 'hail_or_ice', 'reduced_visibility', 'spatial_effects', 'unknown_precip', "origin_pagerank", "dest_pagerank", "origin_out_degree", "dest_in_degree", "prev_flight_arr_delay_clean", "actual_to_crs_time_to_next_flight_diff_mins_clean","crs_time_to_next_flight_diff_mins"
]

# Datatype Conversion
for col_name in (convert_to_int):
        if dict(df.dtypes)[col_name] != 'int':
            df = df.withColumn(col_name, col(col_name).cast(IntegerType()))

# Cast double columns to DoubleType
for col_name in convert_to_double:
    if dict(df.dtypes)[col_name] != 'double':
        df = df.withColumn(col_name, col(col_name).cast(DoubleType()))


### No longer needed as we will manually define window times

# df = df.orderBy(timestamp_col)
# window_spec = Window.orderBy(timestamp_col)
# df = df.withColumn("record_id", row_number().over(window_spec) - 1)
# print("Data sorted by primary timestamp and 'record_id' added.")

# Null population
df_clean, remaining_null_cols, post_null_report = clean_nulls(df)

# COMMAND ----------

# DBTITLE 1,ARR_DEL15 FLAGS ARE NULL
df_clean = df_clean.withColumn(
    "ARR_DEL15",
    F.when(
        F.col("ARR_DEL15").isNull() & ((F.col("ACTUAL_ELAPSED_TIME") - F.col("CRS_ELAPSED_TIME")) >= 15),
        1
    ).when(
        F.col("ARR_DEL15").isNull(),
        0
    ).otherwise(F.col("ARR_DEL15"))
)

# COMMAND ----------

# DBTITLE 1,SKIP: REMOVE RECORDS WITH NULL AIR TIME
# MAGIC %skip
# MAGIC ### only needed if we are including air_time column
# MAGIC # only affects 2 records in 5 years
# MAGIC df_clean = df_clean.filter(F.col("AIR_TIME").isNotNull())

# COMMAND ----------

# DBTITLE 1,VALIDATE NULLS IN SELECTED COLUMNS
#%skip
df_clean = df_clean[final_cols]
null_counts_df = df_clean.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df_clean.columns])
vertical_null_counts = null_counts_df.toPandas().T.reset_index()
vertical_null_counts.columns = ['column', 'null_count']
display(vertical_null_counts)

# COMMAND ----------

# MAGIC %skip
# MAGIC display(df_clean)

# COMMAND ----------

# DBTITLE 1,DATATYPE CHECK
# MAGIC %skip
# MAGIC display(pd.DataFrame({'column': df_clean.columns, 'datatype': [dtype for _, dtype in df_clean.dtypes]}))

# COMMAND ----------

save_path = f"{root}/fasw{time_length}/preprocessed"
dbutils.fs.rm(save_path, True)

df_clean[final_cols].write.format("parquet").mode("overwrite").save(save_path)
print(f"Base processed DataFrame saved as parquet to: {save_path}.")

# COMMAND ----------

