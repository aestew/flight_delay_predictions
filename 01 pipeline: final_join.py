# Databricks notebook source
# MAGIC %md
# MAGIC #New Joined Data Pipeline
# MAGIC ##Running Overwrites Data

# COMMAND ----------

# MAGIC %md
# MAGIC ###End to End Data Cleaning / Joining to saved checkpoint and split data

# COMMAND ----------

# DBTITLE 1,IMPORT PACKAGES
# Import Packages
try:
    import timezonefinder
except ImportError:
    %pip install timezonefinder

import pandas as pd

from pyspark.sql.functions import convert_timezone, pandas_udf, lit, to_timestamp, lpad, concat_ws, to_timestamp, date_format, col, when, expr, split, length, to_date, coalesce
from pyspark.sql.types import StringType
from timezonefinder import TimezoneFinder
from pyspark.sql import functions as F, Window as W, types as T
import time as timer

# COMMAND ----------

# DBTITLE 1,DATA_SET FUNCTION
# Specify timeframe
def data_set(time):
    if time == 3:
        t = "_3m"
    elif time == 6:
        t = "_6m"
    elif time == 12:
        t = "_1y"
    elif time == 'all': # run for 5 years
        t = ""
    else:
        t ='INVALID DUDE'
    return t

## Set time = 3, 6, 12, all

time = 'all'

time_length = data_set(time)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Airline Data Processing

# COMMAND ----------

# DBTITLE 1,LOAD AIRLINE DATA
# Airline Data
# Should accomodate for any of the flights datasets we choose
df_flights = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data{time_length}/").filter(F.col("FL_DATE").between("2015-01-01", "2020-01-01"))
# display(df_flights)

# num_rows = df_flights.count()
# num_cols = len(df_flights.columns)
# print(f"Rows: {num_rows}, Columns: {num_cols}")

# COMMAND ----------

# DBTITLE 1,5 YR FILTER + REMOVE DUPLICATE FLIGHTS
df_flights_no_dupes = df_flights.dropDuplicates()
# display(df_flights_no_dupes)

# num_rows = df_flights_no_dupes.count()
# num_cols = len(df_flights_no_dupes.columns)
# print(f"Rows: {num_rows}, Columns: {num_cols}")

# COMMAND ----------

# DBTITLE 1,FLIGHTS: AIRPORTS TO REMOVE (NON US)
airports_to_remove = ['BQN','STT','GUM','SJU','ISN','PSE','PPG','STX']


# COMMAND ----------

# DBTITLE 1,FLIGHTS: COLS TO KEEP
df_flights_cols_keep = [
    "YEAR", "QUARTER", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "FL_DATE","OP_UNIQUE_CARRIER", "TAIL_NUM", "OP_CARRIER_FL_NUM",
    "ORIGIN_CITY_MARKET_ID", "ORIGIN", "ORIGIN_CITY_NAME", "ORIGIN_STATE_ABR", "ORIGIN_STATE_FIPS", "ORIGIN_STATE_NM", "ORIGIN_WAC",
    "DEST_CITY_MARKET_ID", "DEST", "DEST_CITY_NAME", "DEST_STATE_ABR", "DEST_STATE_FIPS", "DEST_STATE_NM", "DEST_WAC",
    "CRS_DEP_TIME", "DEP_TIME", "DEP_DELAY", "DEP_DEL15", "DEP_TIME_BLK", "TAXI_OUT", "WHEELS_OFF", "WHEELS_ON", "TAXI_IN",
    "CRS_ARR_TIME", "ARR_TIME", "ARR_DELAY", "ARR_DEL15", "ARR_TIME_BLK", "CANCELLED", "DIVERTED", "CRS_ELAPSED_TIME", 
    "ACTUAL_ELAPSED_TIME", "AIR_TIME", "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY", 
    "ACTUAL_ELAPSED_TIME", "DISTANCE", "DISTANCE_GROUP"
]


# COMMAND ----------

# DBTITLE 1,FLIGHTS: FILTERING
## Removed: (df_flights["OP_CARRIER"] == 'UA') &

# Filter: Remove cancelled/diverted flights & non-US origin/dest
# Select: Only columns to keep
# Add Column: Flight ID (use to partition later)
df_flights_filtered = (
    df_flights_no_dupes.filter(
        (df_flights["CANCELLED"] == 0) &
        (df_flights["DIVERTED"] == 0) &
        (~df_flights["ORIGIN"].isin(airports_to_remove)) &
        (~df_flights["DEST"].isin(airports_to_remove))
    )
    .select(df_flights_cols_keep)
    .withColumn("FLIGHT_ID",
        F.concat_ws(
            "_",
            F.col("FL_DATE"),
            F.lpad(F.col("CRS_DEP_TIME").cast("string"), 4, "0"),
            F.col("TAIL_NUM"),
            F.col("OP_CARRIER_FL_NUM"),
            F.col("ORIGIN"),
            F.col("DEST")
        )
    )
)

# display(df_flights_filtered)
# print(f'Size: {len(df_flights_cols_keep)} x {df_flights_filtered.count()}')

# COMMAND ----------

# DBTITLE 1,HOLIDAYS
holidays = [
    # 2015
    ("2015-01-01", "New Year's Day"),
    ("2015-01-19", "Martin Luther King Jr. Day"),
    ("2015-02-16", "Presidents Day"),
    ("2015-05-25", "Memorial Day"),
    ("2015-07-04", "Independence Day"),
    ("2015-09-07", "Labor Day"),
    ("2015-10-12", "Columbus Day"),
    ("2015-11-11", "Veterans Day"),
    ("2015-11-26", "Thanksgiving Day"),
    ("2015-12-24", "Christmas Eve"),
    ("2015-12-25", "Christmas Day"),

    # 2016
    ("2016-01-01", "New Year's Day"),
    ("2016-01-18", "Martin Luther King Jr. Day"),
    ("2016-02-15", "Presidents Day"),
    ("2016-05-30", "Memorial Day"),
    ("2016-07-04", "Independence Day"),
    ("2016-09-05", "Labor Day"),
    ("2016-10-10", "Columbus Day"),
    ("2016-11-11", "Veterans Day"),
    ("2016-11-24", "Thanksgiving Day"),
    ("2016-12-24", "Christmas Eve"),
    ("2016-12-25", "Christmas Day"),

    # 2017
    ("2017-01-01", "New Year's Day"),
    ("2017-01-16", "Martin Luther King Jr. Day"),
    ("2017-02-20", "Presidents Day"),
    ("2017-05-29", "Memorial Day"),
    ("2017-07-04", "Independence Day"),
    ("2017-09-04", "Labor Day"),
    ("2017-10-09", "Columbus Day"),
    ("2017-11-11", "Veterans Day"),
    ("2017-11-23", "Thanksgiving Day"),
    ("2017-12-24", "Christmas Eve"),
    ("2017-12-25", "Christmas Day"),

    # 2018
    ("2018-01-01", "New Year's Day"),
    ("2018-01-15", "Martin Luther King Jr. Day"),
    ("2018-02-19", "Presidents Day"),
    ("2018-05-28", "Memorial Day"),
    ("2018-07-04", "Independence Day"),
    ("2018-09-03", "Labor Day"),
    ("2018-10-08", "Columbus Day"),
    ("2018-11-11", "Veterans Day"),
    ("2018-11-22", "Thanksgiving Day"),
    ("2018-12-24", "Christmas Eve"),
    ("2018-12-25", "Christmas Day"),

    # 2019
    ("2019-01-01", "New Year's Day"),
    ("2019-01-21", "Martin Luther King Jr. Day"),
    ("2019-02-18", "Presidents Day"),
    ("2019-05-27", "Memorial Day"),
    ("2019-07-04", "Independence Day"),
    ("2019-09-02", "Labor Day"),
    ("2019-10-14", "Columbus Day"),
    ("2019-11-11", "Veterans Day"),
    ("2019-11-28", "Thanksgiving Day"),
    ("2019-12-24", "Christmas Eve"),
    ("2019-12-25", "Christmas Day"),
    ("2020-01-01", "New Year's Day"),
]
holiday_df = spark.createDataFrame(holidays, ["holiday_date", "holiday_name"])

holiday_df = holiday_df.withColumn("holiday_date", to_date("holiday_date"))

df_flights_filtered = (
        df_flights_filtered
        .withColumn("FL_DATE", to_date(col("FL_DATE")))
        .join(
            holiday_df,
            (F.abs(F.datediff("fl_date", "holiday_date")) <= 2),
            how="left"
        )
        .withColumn("IS_US_HOLIDAY", F.when(F.col("holiday_name").isNotNull(), F.lit(1)).otherwise(F.lit(0)))
        .drop("holiday_date", "holiday_name")
    )

# COMMAND ----------

# MAGIC %skip
# MAGIC display(df_flights_filtered)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Weather Data Processing

# COMMAND ----------

# DBTITLE 1,LOAD WEATHER DATA
# Weather data
# Should accomodate for any of the flights datasets we choose
df_weather = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data{time_length}/").filter(F.col("DATE").between("2015-01-01", "2020-01-01"))
# display(df_weather)

# num_rows = df_weather.count()
# num_cols = len(df_weather.columns)
# print(f"Rows: {num_rows}, Columns: {num_cols}")

# COMMAND ----------

# DBTITLE 1,WEATHER: COLS TO KEEP
# Keeping columns with <5% nulls

columns_to_keep = [
    "STATION", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION", "NAME", "REPORT_TYPE", "SOURCE", "HourlyAltimeterSetting",
    "HourlyDewPointTemperature", "HourlyDryBulbTemperature", "HourlyPrecipitation", "HourlyPresentWeatherType",
    "HourlyPressureChange", "HourlyPressureTendency", "HourlyRelativeHumidity", "HourlySkyConditions",
    "HourlySeaLevelPressure", "HourlyStationPressure", "HourlyVisibility", "HourlyWetBulbTemperature",
    "HourlyWindDirection", "HourlyWindGustSpeed", "HourlyWindSpeed", "REM", "WindEquipmentChangeDate"
]


# COMMAND ----------

# DBTITLE 1,WEATHER: US ONLY
# Filtering for US locations only
# Keeping columns with <5% nulls
df_weather_filtered = df_weather.filter((df_weather["NAME"].contains("US"))).select(columns_to_keep).cache()
# display(df_weather_filtered)

# num_rows = df_weather_filtered.count()
# num_cols = len(df_weather_filtered.columns)
# print(f"Rows: {num_rows}, Columns: {num_cols}")

# COMMAND ----------

# DBTITLE 1,WEATHER: FX: PARSE PRESENT WEATHER
## FROM CHATGPT  AND REVIEWED BY ME - THIS CODE USES REGEX TO IDENTIFY KEY WEATHER CODES 
## AND RETURN A HUMAN READABLE FIELD AND OHE FIELDS FOR EACH WEATHER TYPE

def parse_present_weather(df_in, col="HourlyPresentWeatherType"):
    # 1) Normalize → tokens
    df = (
        df_in
        .withColumn("wx_raw", F.upper(F.coalesce(F.col(col), F.lit(""))))
        .withColumn("wx1", F.regexp_replace(F.col("wx_raw"), r"\bNULL\b|\*", ""))   # remove 'null'/'*'
        .withColumn("wx2", F.translate(F.col("wx1"), "|", " "))                    # pipes → space
        .withColumn("wx3", F.regexp_replace(F.col("wx2"), r"[^A-Z+:\-\d]+", " "))  # keep codes, drop noise
        .withColumn("wx4", F.regexp_replace(F.col("wx3"), r"\s+", " "))            # collapse spaces
        .withColumn("tokens", F.split(F.trim(F.col("wx4")), r"\s+"))
    )

    # helper: any token matches regex?
    def has(pattern):
        return F.exists(F.col("tokens"), lambda x: x.rlike(pattern))

    # 2) Flags (ints)
    flags = {
    
        ### INTENSITY FLAGS
        "light":   has(r"^\-"),
        "heavy":   has(r"^\+"),

        ### THUNDERSTORM/SQUALL CONDITIONS
        # TS (thunderstorms), TSRA (thunderstorm rain), VCTS (vicinity thunderstorms), SQ (squalls)
        "thunderstorm": 
            has(r"^TS($|[A-Z])") | has(r"\bTSRA\b") | has(r"^VCTS($|[A-Z])") | has(r"(^|\W)SQ(:\d+)?($|\W)"),

        ### RAIN/DRIZZLE
        # RA (rain), DZ (drizzle), SHRA (showers), TSRA (thunderstorm rain)
        "rain_or_drizzle":
            has(r"(^|\W)(RA|DZ)(:\d+)?($|\W)") | 
            has(r"\b(SHRA)\b"),
        
        ### FREEZING CONDITIONS
        # FZ (freezing), FZRA (freezing rain), FZDZ (freezing drizzle)
        "freezing_conditions":
            has(r"^FZ($|[A-Z])") | has(r"\b(FZRA|FZDZ)\b"),
        
        ### SNOW
        # SN (snow), SG (snow grains), SHSN (snow showers), BLSN (blowing snow), DRSN (drifting snow) 
        "snow":
            has(r"(^|\W)(SN|SG)(:\d+)?($|\W)") | 
            has(r"\b(BLSN|DRSN)\b") |
            has(r"\bSHSN\b"),
            
        ### HAIL/ICE
        # GS (small hail), GR (hail), HAIL, PL (ice pellets), IC (ice crystals)
        "hail_or_ice":
            # GS (Small Hail/Graupel) + GR (Hail) + HAIL
            has(r"(^|\W)(GS|GR|PL|IC)(:\d+)?($|\W)") | has(r"\bHAIL\b"),
            
        ### VISIBILITY OBSCURANTS
        # BR (mist), FG (fog), FU (smoke), HZ (haze), DU (dust), SA (sand), VCFG (vicinity fog), and PR (partial)
        "reduced_visibility":
            has(r"(^|\W)(BR|FG|FU|HZ|DU|SA)(:\d+)?($|\W)") | 
            has(r"\bVCFG\b") |
            has(r"(^|\s)PR($|\s)"),

        ### SPATIAL/DRIFTING DESCRIPTORS
        # VC (vicinity), DR (drifting), BL (blowing), MI (shallow), BC (patches)
        "spatial_effects":
            has(r"^VC($|[A-Z])") | 
            has(r"^(DR|BL)($|[A-Z])") |
            has(r"(^|\s)(MI|BC)(:\d+)?($|\s)"),

        ### Unknown
        "unknown_precip":    has(r"(^|\W)UP(:\d+)?($|\W)")
    }

    for name, expr in flags.items():
        df = df.withColumn(name, F.when(expr, F.lit(1)).otherwise(F.lit(0)))

    # 3) Human-readable label (null-safe, no lambdas)
    pieces = [
        F.when(F.col("light")==1, F.lit("light_weather")),
        F.when(F.col("heavy")==1, F.lit("heavy_weather")), 
        F.when(F.col("thunderstorm")==1, F.lit("thunderstorm")),
        F.when(F.col("rain_or_drizzle")==1, F.lit("rain_or_drizzle")),
        F.when(F.col("freezing_conditions")==1, F.lit("freezing_conditions")),
        F.when(F.col("snow")==1, F.lit("snow")),
        F.when(F.col("hail_or_ice")==1, F.lit("hail_or_ice")),
        F.when(F.col("reduced_visibility")==1, F.lit("reduced_visibility")),
        F.when(F.col("spatial_effects")==1, F.lit("spatial_effects")),
        F.when(F.col("unknown_precip")==1, F.lit("unknown_precip"))
    ]
    
    df = (
        df.withColumn("ReadableWeather", F.concat_ws(" ", *pieces))
        .drop("wx_raw", "wx4")
        )

    return df

# COMMAND ----------

# DBTITLE 1,WEATHER: FX: PARSE SKY COND'T
def parse_sky(df_in, col="HourlySkyConditions"):
    # 0) Normalize
    df = (
        df_in
        .withColumn("sky_raw", F.upper(F.coalesce(F.col(col), F.lit(""))))
        .withColumn("sky1", F.regexp_replace("sky_raw", r"\bNULL\b|\*", ""))
        .withColumn("sky2", F.translate("sky1", "|", " "))
        .withColumn("sky3", F.regexp_replace("sky2", r"\s+", " "))
        .withColumn("tokens", F.split(F.trim("sky3"), r"\s+"))
    )

    # 1) Token buckets
    cov_pat  = r"^(FEW|SCT|BKN|OVC):\d{2}$"  # coverage + oktas
    base_pat = r"^\d{2,3}$"                  # base in hundreds of ft (e.g., 25 -> 2500 ft)

    df = (
        df
        .withColumn("cov_tokens",  F.filter("tokens", lambda t: t.rlike(cov_pat)))
        .withColumn("base_tokens", F.filter("tokens", lambda t: t.rlike(base_pat)))
    )

    # 2) Parse arrays
    code_arr = F.transform("cov_tokens",  lambda c: F.regexp_extract(c, r"^(FEW|SCT|BKN|OVC)", 1))
    okta_arr = F.transform("cov_tokens",  lambda c: F.regexp_extract(c, r":(\d{2})", 1).cast("int"))
    base_arr = F.transform("base_tokens", lambda b: (b.cast("int") * F.lit(100)))  # hundreds → ft

    # Array of structs with clear field names; no need for another transform
    layers = F.arrays_zip(
        code_arr.alias("code"),
        okta_arr.alias("okta"),
        base_arr.alias("base_ft_agl")
    )

    df = df.withColumn("layers", layers).drop("cov_tokens","base_tokens","tokens", "sky1","sky2","sky3")

    # 3) Presence flags
    df = (
        df
        .withColumn("has_few", F.exists("layers", lambda l: l.getField("code") == F.lit("FEW")).cast("int"))
        .withColumn("has_sct", F.exists("layers", lambda l: l.getField("code") == F.lit("SCT")).cast("int"))
        .withColumn("has_bkn", F.exists("layers", lambda l: l.getField("code") == F.lit("BKN")).cast("int"))
        .withColumn("has_ovc", F.exists("layers", lambda l: l.getField("code") == F.lit("OVC")).cast("int"))
        .withColumn(
        "HourlyAltimeterSetting",F.coalesce(F.col("HourlyAltimeterSetting"), F.col("HourlySeaLevelPressure"), F.col("HourlyStationPressure"), F.col("HourlySeaLevelPressure"))
    ))

    # 4) Aggregates / features
    size_layers   = F.size("layers")
    oktas         = F.transform("layers", lambda l: l.getField("okta"))
    bases_ft      = F.transform("layers", lambda l: l.getField("base_ft_agl"))

    overall_oktas = F.when(size_layers > 0, F.array_max(oktas)).otherwise(F.lit(0))
    cloud_frac    = overall_oktas.cast("double") / F.lit(8.0)

    lowest_base   = F.when(size_layers > 0, F.array_min(bases_ft))
    highest_base  = F.when(size_layers > 0, F.array_max(bases_ft))
    mean_base     = F.when(
        size_layers > 0,
        F.aggregate(bases_ft, F.lit(0), lambda acc, x: acc + x) / size_layers.cast("double")
    )
    base_range    = highest_base - lowest_base

    bkn_ovc       = F.filter("layers", lambda l: l.getField("code").isin("BKN","OVC"))
    ceiling_ft    = F.when(F.size(bkn_ovc) > 0,
                           F.array_min(F.transform(bkn_ovc, lambda l: l.getField("base_ft_agl"))))
    ceiling_ft    = F.coalesce(ceiling_ft, F.lit(0))

    sky_missing   = (size_layers == 0).cast("int")

    df = (
        df
        .withColumn("overall_cloud_frac_0_1", cloud_frac)
        .withColumn("lowest_cloud_ft", lowest_base)
        .withColumn("highest_cloud_ft", highest_base)
    )

    # # 5) Readable string
    # readable = F.transform(
    #     "layers",
    #     lambda l: F.concat_ws(
    #         "",
    #         l.getField("code"),
    #         F.lit(" "),
    #         (l.getField("base_ft_agl")/F.lit(100)).cast("int"),
    #         F.lit("00 ft")
    #     )
    # )
    # df = df.withColumn("ReadableSky", F.concat_ws(", ", readable))
    
    return df


# COMMAND ----------

df_weather_filtered = parse_sky(df_weather_filtered)
df_weather_filtered = parse_present_weather(df_weather_filtered)
# display(df_weather_filtered)

# COMMAND ----------

# HourlyDewPointTemperature - indicates fog; HourlyDryBulbTemperature - icing conditions; HourlyPrecipitation - repetitive; HourlyRelativeHumidity - repetitive; HourlyVisibility - repetitive; HourlyWetBulbTemperature - repetitive
# Not as informative as HourlyAltimeterSetting: HourlyPressureChange / HourlyPressureTendency / HourlySeaLevelPressure / HourlyStationPressure; HourlyWindDirection - not informative by itself

weather_cols = [
    "STATION", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION", "NAME", "REPORT_TYPE", "SOURCE",
    "overall_cloud_frac_0_1", "lowest_cloud_ft", "highest_cloud_ft", "has_few", "has_sct", "has_bkn", 
    "has_ovc", "HourlyAltimeterSetting", "HourlyWindDirection", "HourlyWindGustSpeed", "HourlyWindSpeed", "light", "heavy", "thunderstorm", "rain_or_drizzle", "freezing_conditions", "snow",
    "hail_or_ice", "reduced_visibility", "spatial_effects", "unknown_precip"
]
df_weather_filtered = df_weather_filtered.select(weather_cols)
# display(df_weather_filtered)

# COMMAND ----------

# MAGIC %skip
# MAGIC row_count = df_weather_filtered.count()
# MAGIC null_counts = df_weather_filtered.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df_weather_filtered.columns])
# MAGIC vertical_table = null_counts.selectExpr("stack({0}, {1}) as (column, null_count)".format(
# MAGIC     len(df_weather_filtered.columns),
# MAGIC     ", ".join([f"'{c}', {c}" for c in df_weather_filtered.columns])
# MAGIC )).withColumn(
# MAGIC     "null_pct", (F.col("null_count") / F.lit(row_count)) * 100
# MAGIC )
# MAGIC display(vertical_table)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stations Data Processing

# COMMAND ----------

# DBTITLE 1,STATIONS: IMPORT
# Stations data      
df_stations = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/stations_data/stations_with_neighbors.parquet/")
# display(df_stations)

# num_rows = df_stations.count()
# num_cols = len(df_stations.columns)
# print(f"Rows: {num_rows}, Columns: {num_cols}")

# COMMAND ----------

# DBTITLE 1,STATIONS: COLS TO KEEP
cols_to_keep = [
    "station_id", "lat", "lon", "distance_to_neighbor", "type", "elevation_ft", "icao_code", "iata_code", "local_code"
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Airport Codes

# COMMAND ----------

# DBTITLE 1,LOAD AIRPORT CODES
# Airport Codes and Metadata
df_airport_codes = spark.read.csv("dbfs:/FileStore/airport_codes.csv", header=True, inferSchema=True)
# display(df_airport_codes)

# num_rows = df_airport_codes.count()
# num_cols = len(df_airport_codes.columns)
# print(f"Rows: {num_rows}, Columns: {num_cols}")

# COMMAND ----------

# DBTITLE 1,AIRPORT CODES: FILTER
df_airport_codes_filtered = df_airport_codes.filter(
    (df_airport_codes["iso_country"] == "US") &
    (df_airport_codes["icao_code"].isNotNull()) &
    (df_airport_codes["iata_code"].isNotNull())
)
# display(df_airport_codes_filtered)

# num_rows = df_airport_codes_filtered.count()
# num_cols = len(df_airport_codes_filtered.columns)
# print(f"Rows: {num_rows}, Columns: {num_cols}")

# COMMAND ----------

display(df_airport_codes_filtered)

# COMMAND ----------

# MAGIC %md
# MAGIC ***
# MAGIC ### DATA SUB-JOINS BEGIN
# MAGIC ***

# COMMAND ----------

# MAGIC %md
# MAGIC ## JOIN: Station to Airport Code

# COMMAND ----------

# DBTITLE 1,FLIGHTS: AIRPORTS TO KEEP
# List of all origin airports
df_airport_origins_list = df_flights_filtered.select("ORIGIN").distinct().rdd.flatMap(lambda x:x).collect()

# COMMAND ----------

# DBTITLE 1,STATION TO AIRPORT: JOIN AND FILTER FILTER RELEVANT AIRPORTS
## Filter for local_code matching airport list
df_airport_station_joined = (
    df_stations.join(
        df_airport_codes_filtered,
        df_stations["neighbor_call"] == df_airport_codes_filtered["icao_code"],
        how="left"
    )
    .filter(
        col("iata_code").isin(df_airport_origins_list)
    )
)
# display(df_airport_station_joined)

# num_rows = df_airport_station_joined.count()
# num_cols = len(df_airport_station_joined.columns)
# print(f"Rows: {num_rows}, Columns: {num_cols}")

# display(df_airport_station_joined)

# total_count = df_airport_station_joined.count()
# distinct_count = df_airport_station_joined.distinct().count()
# print(f"Total rows: {total_count}, Distinct rows: {distinct_count}")

# COMMAND ----------

# DBTITLE 1,CREATE FILTER LIST: ALL STATIONS
# MAGIC %skip
# MAGIC # df_dist_stations = df_airport_station_joined.select("station_id").distinct()
# MAGIC # df_dist_stations_list =  [row.station_id for row in df_dist_stations.collect()]
# MAGIC # display(spark.createDataFrame([(row,) for row in df_dist_stations_list], schema=['station_id']))

# COMMAND ----------

# DBTITLE 1,FILTER: AIRPORT-STATION TO CLOSEST STATION
# Define a window specification partitioned by 'iata_code' and ordered by 'distance_to_neighbor'
window_spec = W.partitionBy("icao_code").orderBy("distance_to_neighbor")

# Assign row numbers based on the window specification
df_min_dist = df_airport_station_joined.withColumn("row_number", F.row_number().over(window_spec))

# Filter to keep only the rows with the row number equal to 1 (indicating the row with the minimum distance for each airport)
df_airport_station_joined_filtered = df_min_dist.filter(df_min_dist["row_number"] == 1).drop("row_number")

#display(df_airport_station_joined_filtered)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Convert Local to UTC

# COMMAND ----------

# DBTITLE 1,AIRPORT: SPLIT COORDS FOR JOIN
# Filtering and transforming for airport codes and metadata
# Use only filtered stations
df_airport_codes_clean = df_airport_station_joined_filtered \
    .filter(col("icao_code").isNotNull() & col("iata_code").isNotNull() & (length(col("iata_code")) == 3)) \
    .withColumn("latitude", split(col("coordinates"), ",")[0]) \
    .withColumn("longitude", split(col("coordinates"), ",")[1]) \
    .filter(col("latitude").isNotNull() & col("longitude").isNotNull()) \
    .select("icao_code", "iata_code", "latitude", "longitude", "elevation_ft", "type", "station_id")



# display(df_airport_codes_clean)

# total_count = df_airport_codes_clean.count()
# num_cols = len(df_airport_codes_clean.columns)
# distinct_count = df_airport_codes_clean.distinct().count()

# print(f"Total rows: {total_count}, Columns: {num_cols}, Distinct rows: {distinct_count}")

# COMMAND ----------

# DBTITLE 1,SUPP TABLE FOR AIRPORTS WITH NO STATION
df_airport_codes_supp = df_airport_codes_filtered \
    .filter(col("icao_code").isNotNull() & col("iata_code").isNotNull() & (length(col("iata_code")) == 3)) \
    .withColumn("latitude", split(col("coordinates"), ",")[0]) \
    .withColumn("longitude", split(col("coordinates"), ",")[1]) \
    .filter(col("latitude").isNotNull() & col("longitude").isNotNull()) \
    .select("icao_code", "iata_code", "latitude", "longitude", "elevation_ft", "type")


# COMMAND ----------

# DBTITLE 1,FLIGHTS: JOIN AIRPORT CODES TO ORIGIN AND DEST
from pyspark.sql.functions import col, coalesce

# Join airport codes to pull ICAO and LAT/LONG for ORIGIN/DEST
df_new_otpw = df_flights_filtered.join(
    df_airport_codes_clean
    .withColumnRenamed("iata_code", "ORIGIN_IATA")
    .withColumnRenamed("latitude", "ORIGIN_LAT")
    .withColumnRenamed("longitude", "ORIGIN_LONG")
    .withColumnRenamed("icao_code", "ORIGIN_ICAO")
    .withColumnRenamed("station_id", "STATION")
    .withColumnRenamed("elevation_ft", "ORIGIN_ELEVATION_FT")
    .withColumnRenamed("type", "ORIGIN_SIZE"),
    df_flights_filtered["ORIGIN"] == col("ORIGIN_IATA"),
    "left"
)

df_new_otpw = df_new_otpw.join(
    df_airport_codes_clean
    .withColumnRenamed("iata_code", "DEST_IATA")
    .withColumnRenamed("latitude", "DEST_LAT")
    .withColumnRenamed("longitude", "DEST_LON")
    .withColumnRenamed("icao_code", "DEST_ICAO")
    .withColumnRenamed("elevation_ft", "DEST_ELEVATION_FT")
    .withColumnRenamed("type", "DEST_SIZE"),
    df_new_otpw["DEST"] == col("DEST_IATA"),
    "left"
)

# Fill missing ORIGIN columns from df_airport_codes_filtered
df_new_otpw = df_new_otpw.join(
    df_airport_codes_supp
    .withColumnRenamed("iata_code", "ORIGIN_IATA_BACKUP")
    .withColumnRenamed("latitude", "ORIGIN_LAT_BACKUP")
    .withColumnRenamed("longitude", "ORIGIN_LONG_BACKUP")
    .withColumnRenamed("icao_code", "ORIGIN_ICAO_BACKUP")
    .withColumnRenamed("elevation_ft", "ORIGIN_ELEVATION_FT_BACKUP")
    .withColumnRenamed("type", "ORIGIN_SIZE_BACKUP"),
    (col("ORIGIN") == col("ORIGIN_IATA_BACKUP")), 
    "left"
)

df_new_otpw = df_new_otpw.withColumn(
    "ORIGIN_IATA", coalesce(col("ORIGIN_IATA"), col("ORIGIN_IATA_BACKUP"))
).withColumn(
    "ORIGIN_LAT", coalesce(col("ORIGIN_LAT"), col("ORIGIN_LAT_BACKUP"))
).withColumn(
    "ORIGIN_LONG", coalesce(col("ORIGIN_LONG"), col("ORIGIN_LONG_BACKUP"))
).withColumn(
    "ORIGIN_ICAO", coalesce(col("ORIGIN_ICAO"), col("ORIGIN_ICAO_BACKUP"))
).withColumn(
    "ORIGIN_ELEVATION_FT", coalesce(col("ORIGIN_ELEVATION_FT"), col("ORIGIN_ELEVATION_FT_BACKUP"))
).withColumn(
    "ORIGIN_SIZE", coalesce(col("ORIGIN_SIZE"), col("ORIGIN_SIZE_BACKUP"))
).drop(
    "ORIGIN_IATA_BACKUP", "ORIGIN_LAT_BACKUP", "ORIGIN_LONG_BACKUP",
    "ORIGIN_ICAO_BACKUP", "ORIGIN_ELEVATION_FT_BACKUP", "ORIGIN_SIZE_BACKUP"
)

# Fill missing DEST columns from df_airport_codes_filtered
df_new_otpw = df_new_otpw.join(
    df_airport_codes_supp
    .withColumnRenamed("iata_code", "DEST_IATA_BACKUP")
    .withColumnRenamed("latitude", "DEST_LAT_BACKUP")
    .withColumnRenamed("longitude", "DEST_LON_BACKUP")
    .withColumnRenamed("icao_code", "DEST_ICAO_BACKUP")
    .withColumnRenamed("elevation_ft", "DEST_ELEVATION_FT_BACKUP")
    .withColumnRenamed("type", "DEST_SIZE_BACKUP"),
    (col("DEST") == col("DEST_IATA_BACKUP")), 
    "left"
)

df_new_otpw = df_new_otpw.withColumn(
    "DEST_IATA", coalesce(col("DEST_IATA"), col("DEST_IATA_BACKUP"))
).withColumn(
    "DEST_LAT", coalesce(col("DEST_LAT"), col("DEST_LAT_BACKUP"))
).withColumn(
    "DEST_LON", coalesce(col("DEST_LON"), col("DEST_LON_BACKUP"))
).withColumn(
    "DEST_ICAO", coalesce(col("DEST_ICAO"), col("DEST_ICAO_BACKUP"))
).withColumn(
    "DEST_ELEVATION_FT", coalesce(col("DEST_ELEVATION_FT"), col("DEST_ELEVATION_FT_BACKUP"))
).withColumn(
    "DEST_SIZE", coalesce(col("DEST_SIZE"), col("DEST_SIZE_BACKUP"))
).drop(
    "DEST_IATA_BACKUP", "DEST_LAT_BACKUP", "DEST_LON_BACKUP",
    "DEST_ICAO_BACKUP", "DEST_ELEVATION_FT_BACKUP", "DEST_SIZE_BACKUP"
)

# display(df_new_otpw)
# total_count = df_new_otpw.count()
# distinct_count = df_new_otpw.distinct().count()
# print(f"Total rows: {total_count}, Distinct rows: {distinct_count}")

# COMMAND ----------

# DBTITLE 1,SKIP: CHECK NULL AIRPORT INFO
# MAGIC %skip
# MAGIC df_ogs = df_new_otpw.filter((col("ORIGIN") == "OGS") | (col("DEST") == "OGS"))
# MAGIC display(df_ogs)

# COMMAND ----------

# DBTITLE 1,FLIGHTS: ASSIGN DATES TO TIMES
# Assuming FL_DATE is CRS_DEP_DATE (Scheduled Departure Date)

cleanup_cols = ["DEP_TIME_CLEAN", "ARR_TIME_CLEAN", "CRS_DEP_TIME_CLEAN", "CRS_ARR_TIME_CLEAN", "DEP_DATE", "ARR_DATE", "CRS_ARR_DATE"]

# Convert 24:00 timestamps to 0:00
df_new_otpw = (
    df_new_otpw
    .withColumn("DEP_TIME_CLEAN", F.when(col("DEP_TIME") == 2400, F.lit(0)).otherwise(col("DEP_TIME")))
    .withColumn("ARR_TIME_CLEAN", F.when(col("ARR_TIME") == 2400, F.lit(0)).otherwise(col("ARR_TIME")))
    .withColumn("CRS_DEP_TIME_CLEAN", F.when(col("CRS_DEP_TIME") == 2400, F.lit(0)).otherwise(col("CRS_DEP_TIME")))
    .withColumn("CRS_ARR_TIME_CLEAN", F.when(col("CRS_ARR_TIME") == 2400, F.lit(0)).otherwise(col("CRS_ARR_TIME")))
)
    ### Edge Cases
    # If dep_time < crs_dep_time and dep_delay > 0
df_new_otpw = (
    df_new_otpw.withColumn("DEP_DATE",
        when(
            ((col("DEP_TIME") < col("CRS_DEP_TIME")) & (col("DEP_DELAY") > 0)) | (col("DEP_TIME") == 2400),
            expr("date_add(FL_DATE, 1)")
        ).otherwise(col("FL_DATE"))
    )

    # If arr < dep, then it is next day
    .withColumn("ARR_DATE",
        when(
            (col("ARR_TIME") < col("DEP_TIME")) | (col("ARR_TIME") == 2400),
            expr("date_add(DEP_DATE, 1)")
        ).otherwise(col("FL_DATE"))
    )

    # If sch arr < sch dep, then it is next day
    .withColumn("CRS_ARR_DATE",
        when(
            (col("CRS_ARR_TIME") < col("CRS_DEP_TIME")) | (col("CRS_ARR_TIME") == 2400),
            expr("date_add(FL_DATE, 1)")
        ).otherwise(col("FL_DATE"))
    )

    ### Normal Cases
    # Departure 
    .withColumn("DEP_DATETIME_LOCAL",
        date_format(
            to_timestamp(
                concat_ws(' ',col("DEP_DATE"),lpad(col("DEP_TIME_CLEAN"), 4, "0")),
            "yyyy-MM-dd HHmm"),
        "yyyy-MM-dd'T'HH:mm:ss")
    )

    # Arrival
    .withColumn("ARR_DATETIME_LOCAL",
        date_format(
            to_timestamp(
                concat_ws(' ', col("ARR_DATE"), lpad(col("ARR_TIME_CLEAN"), 4, "0")),
            "yyyy-MM-dd HHmm"),
        "yyyy-MM-dd'T'HH:mm:ss")
    )

    # Sch Departure
    .withColumn("CRS_DEP_DATETIME_LOCAL",
        date_format(
            to_timestamp(
                concat_ws(' ',col("FL_DATE"),lpad(col("CRS_DEP_TIME_CLEAN"), 4, "0")),
            "yyyy-MM-dd HHmm"),
        "yyyy-MM-dd'T'HH:mm:ss")
    )

    # Sch Arrival
    .withColumn("CRS_ARR_DATETIME_LOCAL",
        date_format(
            to_timestamp(
                concat_ws(' ', col("CRS_ARR_DATE"), lpad(col("CRS_ARR_TIME_CLEAN"), 4, "0")),
            "yyyy-MM-dd HHmm"),
        "yyyy-MM-dd'T'HH:mm:ss")
    )

    # Cleanup
    .drop(*cleanup_cols)
)

# display(df_new_otpw)

# COMMAND ----------

# DBTITLE 1,FLIGHTS:  CONVERT LOCAL TO UTC FOR FLIGHTS-AIRPORT-STATION & ADD GRANULARITY COLS
# Can't broadcast and each worker needs to create its own instance
# Convert local timezone to UTC
from pyspark.sql.functions import lpad, substring, col

@pandas_udf(StringType())
def get_timezone_pandas_udf(lat_series: pd.Series, lon_series: pd.Series) -> pd.Series:
    tf = TimezoneFinder()
    timezones = [
       tf.timezone_at(lat=float(lat), lng=float(lon)) if pd.notnull(lat) and pd.notnull(lon) else None
        for lat, lon in zip(lat_series, lon_series)
    ]
    return pd.Series(timezones)

df_new_otpw = df_new_otpw.withColumn(
    "ORIGIN_TIMEZONE",
    get_timezone_pandas_udf(col("origin_lat"), col("origin_long"))
).withColumn(
    "DEST_TIMEZONE",
    get_timezone_pandas_udf(col("dest_lat"), col("dest_lon"))
)

df_new_otpw = df_new_otpw.withColumn(
    "DEP_DATETIME_UTC",
    convert_timezone(
        col("ORIGIN_TIMEZONE"),
        lit("UTC"),
        to_timestamp(col("DEP_DATETIME_LOCAL"), "yyyy-MM-dd'T'HH:mm:ss")
    )
).withColumn(
    "ARR_DATETIME_UTC",
    convert_timezone(
        col("DEST_TIMEZONE"),
        lit("UTC"),
        to_timestamp(col("ARR_DATETIME_LOCAL"), "yyyy-MM-dd'T'HH:mm:ss")
    )
).withColumn(
    "CRS_DEP_DATETIME_UTC",
    convert_timezone(
        col("ORIGIN_TIMEZONE"),
        lit("UTC"),
        to_timestamp(col("CRS_DEP_DATETIME_LOCAL"), "yyyy-MM-dd'T'HH:mm:ss")
    )
).withColumn(
    "CRS_ARR_DATETIME_UTC",
    convert_timezone(
        col("DEST_TIMEZONE"),
        lit("UTC"),
        to_timestamp(col("CRS_ARR_DATETIME_LOCAL"), "yyyy-MM-dd'T'HH:mm:ss")
    )
).withColumn(
    "CRS_ARR_HOUR", substring(lpad("CRS_ARR_TIME", 4, "0"), 1, 2).cast("int")
).withColumn(
    "ARR_HOUR", substring(lpad("ARR_TIME", 4, "0"), 1, 2).cast("int")
).withColumn(
    "CRS_DEP_HOUR", substring(lpad("CRS_DEP_TIME", 4, "0"), 1, 2).cast("int")
).withColumn(
    "DEP_HOUR", substring(lpad("DEP_TIME", 4, "0"), 1, 2).cast("int")
)

# display(df_new_otpw)

# num_rows = df_new_otpw.count()
# num_cols = len(df_new_otpw.columns)
# print(f"Rows: {num_rows}, Columns: {num_cols}")

# COMMAND ----------

# MAGIC %skip
# MAGIC df_null_times = df_new_otpw.filter(
# MAGIC     col("DEP_DATETIME_UTC").isNull() | col("ARR_DATETIME_UTC").isNull()
# MAGIC )
# MAGIC display(df_null_times)

# COMMAND ----------

# DBTITLE 1,FINAL: COLS TO KEEP
cols_to_keep = [
    "YEAR", "QUARTER", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK", "FL_DATE", "IS_US_HOLIDAY", "OP_UNIQUE_CARRIER", "TAIL_NUM", "OP_CARRIER_FL_NUM",
    "ORIGIN_CITY_MARKET_ID", "ORIGIN", "ORIGIN_CITY_NAME", "ORIGIN_STATE_ABR", "ORIGIN_STATE_FIPS", "ORIGIN_STATE_NM", "ORIGIN_WAC",
    "DEST_CITY_MARKET_ID", "DEST", "DEST_CITY_NAME", "DEST_STATE_ABR", "DEST_STATE_FIPS", "DEST_STATE_NM", "DEST_WAC",
    "CRS_DEP_TIME", "CRS_DEP_HOUR", "DEP_TIME", "DEP_HOUR", "DEP_DELAY", "DEP_DEL15", "DEP_TIME_BLK", "TAXI_OUT", "WHEELS_OFF", "WHEELS_ON", "TAXI_IN",
    "CRS_ARR_TIME", "CRS_ARR_HOUR", "ARR_TIME", "ARR_HOUR", "ARR_DELAY", "ARR_DEL15", "ARR_TIME_BLK", "CANCELLED", "DIVERTED", "CRS_ELAPSED_TIME", 
    "ACTUAL_ELAPSED_TIME", "AIR_TIME", "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY", "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY", "DISTANCE", "DISTANCE_GROUP", "FLIGHT_ID", "ORIGIN_ICAO", "ORIGIN_IATA", "ORIGIN_LAT", "ORIGIN_LONG", "ORIGIN_ELEVATION_FT", "ORIGIN_SIZE", "STATION",
    "DEST_ICAO", "DEST_IATA", "DEST_LAT", "DEST_LON", "DEST_ELEVATION_FT", "DEST_SIZE", "DEP_DATETIME_UTC", "ARR_DATETIME_UTC", 
    "CRS_DEP_DATETIME_UTC", "CRS_ARR_DATETIME_UTC"
]

# COMMAND ----------

# DBTITLE 1,FAS: FILTER TO NECESSARY COLUMNS
df_final_otpw = df_new_otpw[cols_to_keep]
# display(df_final_otpw)

# num_rows = df_final_otpw.count()
# num_cols = len(df_final_otpw.columns)
# print(f"Rows: {num_rows}, Columns: {num_cols}")

# COMMAND ----------

# MAGIC %md
# MAGIC ***
# MAGIC # WEATHER TO FAS JOIN
# MAGIC ***

# COMMAND ----------

# DBTITLE 1,SPARK: ENABLE FEATURES
# These are safe to set; if your workspace locks them, Spark will just ignore.
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")

# COMMAND ----------

# DBTITLE 1,SPARK: CONFIRM ENABLED
print("Spark:", spark.version)
print("AQE:", spark.conf.get("spark.sql.adaptive.enabled"))
print("Skew:", spark.conf.get("spark.sql.adaptive.skewJoin.enabled"))
print("AutoBroadcast:", spark.conf.get("spark.sql.autoBroadcastJoinThreshold"))


# COMMAND ----------

# DBTITLE 1,WEATHER: FILTER TO ONLY NECESSARY STATIONS
# Filter weather data
final_flight_df=df_final_otpw
flight_stations = final_flight_df.select("STATION").distinct().withColumnRenamed("STATION", "JOIN_STATION")

final_weather_df = df_weather_filtered.join(
    flight_stations,
    df_weather_filtered["STATION"] == flight_stations["JOIN_STATION"],
    "leftsemi"
)
# display(final_weather_df)

# num_rows = FINAL_WEATHER_DATA.count()
# num_cols = len(FINAL_WEATHER_DATA.columns)
# print(f"Rows: {num_rows}, Columns: {num_cols}")

# COMMAND ----------

# DBTITLE 1,MEGA JOIN: FX WEATHER TO FLIGHTS
# GOAL: for each flight, find the closest weather information to the local station before hour from takeoff time (ideally less than 5 hours from flight)

## create column minus 4 hours

from pyspark.sql import functions as F, Window as W

def mega_join(flights, weather, offset=1, tolerance=6):

    TOL_SEC = tolerance * 3600  

    ## properly format times
    fl = (
        flights
        # Ensure the flight timestamp is typed correctly
        .withColumn("CRS_DEP_DATETIME_UTC", F.col("CRS_DEP_DATETIME_UTC").cast("timestamp"))
        # Target time = takeoff - Offset
        .withColumn("TARGET_TIME_UTC",
                    F.col("CRS_DEP_DATETIME_UTC") - F.expr(f"INTERVAL {offset} HOURS"))
        .withColumn("T_EPOCH", F.col("TARGET_TIME_UTC").cast("long"))
    )

    ##weather change to epoch
    wx = (
        weather
        .withColumn("DATE", F.col("DATE").cast("timestamp"))
        .withColumn("OBS_EPOCH", F.col("DATE").cast("long")) ## observed weather timestamp
    )

    ## Define key value pair for reparition by range
    fl_part = fl.repartitionByRange("STATION", "T_EPOCH").alias("f")
    wx_part = wx.repartitionByRange("STATION", "OBS_EPOCH").alias("w")

    ## Join conditions
    join_pred = (
            (F.col("f.STATION") == F.col("w.STATION")) &
            (F.col("w.OBS_EPOCH") <= F.col("f.T_EPOCH")) &
            ((F.col("f.T_EPOCH") - F.col("w.OBS_EPOCH")) <= TOL_SEC)
        )
    
    ## Identify potential candidates
    cand = fl_part.join(wx_part, join_pred, "left")

    ## Partition
    w_part = (
        W.partitionBy(F.col("f.FLIGHT_ID"))
        .orderBy((F.col("f.T_EPOCH") - F.col("w.OBS_EPOCH")).asc())
    )

    closest_match = (
        cand
        .where(F.col("w.OBS_EPOCH").isNotNull())
        .withColumn("rank", F.row_number().over(w_part))
        .where(F.col("rank") == 1)
        .drop("rank")
    )

    # final join
    wx_cols_to_attach = [c for c in weather.columns if c not in ("STATION", "DATE")]

    result = (
        fl.alias("f")
        .join(
            closest_match.select(
                F.col("f.FLIGHT_ID").alias("FID"),
                F.col("w.STATION").alias("WX_STATION"),
                F.col("w.DATE").alias("MATCHED_OBS_TIME_UTC"),
                *[F.col(f"w.{c}") for c in wx_cols_to_attach]
            ),
            on=F.col("f.FLIGHT_ID") == F.col("FID"),
            how="left"
        )
        .drop("FID")
    )

    matched_pct = (result.where(F.col("MATCHED_OBS_TIME_UTC").isNotNull()).count(),
                result.count())
    print("Matched / Total flights:", matched_pct)
    
    return result

# COMMAND ----------

# DBTITLE 1,JOIN FLIGHTS TO WEATHER
start_time = timer.time()
result = mega_join(final_flight_df, final_weather_df, offset=1, tolerance=8)
end_time = timer.time()
print(f"mega_join execution time: {end_time - start_time:.2f} seconds")
#  (1338506, 1338511) - 3M

# COMMAND ----------

# DBTITLE 1,SKIP: display join
# MAGIC %skip
# MAGIC #display(result)

# COMMAND ----------

# DBTITLE 1,SAVE TO PATH
spark.conf.set("spark.databricks.delta.optimizeWrite.enabled", "true")
spark.conf.set("spark.databricks.delta.autoCompact.enabled", "true")

# Rewrites the path
save_path = f"dbfs:/student-groups/Group_02_01/fasw{time_length}/final_join"
dbutils.fs.rm(save_path, True)

start_time = timer.time()
if time_length == '':
    num_partitions = 100
    partition_keys = ["FL_DATE","OP_UNIQUE_CARRIER"]
    (
        result
        .repartition(num_partitions, *partition_keys)
        .write
        .format("parquet")
        .mode("overwrite")
        .save(save_path)
    )
else:
    result.write.format("parquet").mode("overwrite").save(save_path)
end_time = timer.time()
print(f"FASW join saved as parquet to: {save_path}.")
print(f"Write execution time: {end_time - start_time:.2f} seconds")

# COMMAND ----------

# DBTITLE 1,SKIP: QC DATASET
# MAGIC %skip
# MAGIC ### check origin,utc depart time match
# MAGIC
# MAGIC station_check = result\
# MAGIC .select("STATION","ORIGIN","NAME","DEP_DATETIME_UTC","MATCHED_OBS_TIME_UTC")\
# MAGIC .withColumn("diff_dep_time_hr", (F.unix_timestamp(F.col("DEP_DATETIME_UTC")) - F.unix_timestamp(F.col("MATCHED_OBS_TIME_UTC"))) / 3600)
# MAGIC
# MAGIC display(station_check)
# MAGIC
# MAGIC ## confirm duplicates
# MAGIC
# MAGIC num_total = result.count()
# MAGIC num_distinct = result.distinct().count()
# MAGIC num_duplicates = num_total - num_distinct
# MAGIC print(f"Number of duplicate rows: {num_duplicates}")

# COMMAND ----------

# DBTITLE 1,SKIP: DUPE CHECK
# MAGIC %skip
# MAGIC dup_flight_ids = (
# MAGIC     result.groupBy("FLIGHT_ID")
# MAGIC     .count()
# MAGIC     .filter(F.col("count") > 1)
# MAGIC )
# MAGIC
# MAGIC display(dup_flight_ids)

# COMMAND ----------

# MAGIC %skip
# MAGIC ## DELETES FOLDERS
# MAGIC dbutils.fs.rm("dbfs:/student-groups/Group_02_01/splits_3m",recurse=True)
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC
# MAGIC %skip
# MAGIC ### VERSION 1 (Split by Year)
# MAGIC ### Filter by Year if we have all 5 years
# MAGIC df_train = df.filter((F.year("FL_DATE") >= 2015) & (F.year("FL_DATE") <= 2017))
# MAGIC df_val   = df.filter(F.year("FL_DATE") == 2018)
# MAGIC df_test  = df.filter(F.year("FL_DATE") == 2019)
# MAGIC print("Train:", df_train.count(), "rows")
# MAGIC print("Validation:", df_val.count(), "rows")
# MAGIC print("Test:", df_test.count(), "rows")
# MAGIC
# MAGIC # ### VERSION 2 (Split by Ratio, Final Dataset MUST be already sorted in chronological order)
# MAGIC # from pyspark.sql.window import Window
# MAGIC # from pyspark.sql.functions import row_number, col
# MAGIC # train_ratio, val_ratio = 0.6, 0.2
# MAGIC # total_count = df_otpw_final.count()
# MAGIC # train_threshold = int(total_count * train_ratio)
# MAGIC # val_threshold = train_threshold + int(total_count * val_ratio)
# MAGIC # df_indexed = df_otpw_final.withColumn("row_num", row_number().over(Window.orderBy("FL_DATE")))
# MAGIC # train_df = df_indexed.filter(col("row_num") <= train_threshold).drop("row_num")
# MAGIC # val_df = df_indexed.filter((col("row_num") > train_threshold) & (col("row_num") <= val_threshold)).drop("row_num")
# MAGIC # test_df = df_indexed.filter(col("row_num") > val_threshold).drop("row_num")

# COMMAND ----------

# MAGIC %md
# MAGIC - FCA iata codes were missing in data because local_code in the airport codes was coded as the ICAO without the K (GPI)
# MAGIC     - Changed cell 25: STATION TO AIRPORT: FILTER RELEVANT AIRPORTS to filter IATA code instead of local_code
# MAGIC     - This also fixed issue with null lats/longs
# MAGIC - Changed Flight UUID to Flight ID (standard)
# MAGIC - Included more columns (that have >5% nulls) so that there is a better heatmap
# MAGIC - Minor performance code updates
# MAGIC - Commented code that we can potentially delete as they are more for quality checks
# MAGIC
# MAGIC Other notes:
# MAGIC - We technically don’t need DEST IATA, LONG, LAT, etc. since we’re only joining based on departure, but we can still keep it
# MAGIC