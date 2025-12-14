# Databricks notebook source
# MAGIC %md
# MAGIC MLP with Rolling Windows

# COMMAND ----------

# DBTITLE 1,IMPORT
import os
import pandas as pd
from pyspark.sql.functions import col
from pyspark.sql import functions
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, FeatureHasher, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from xgboost.spark import SparkXGBClassifier
import mlflow

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

#Checkpoint location
dbfs_path = "dbfs:/student-groups/Group_02_01"

#RUN FOR 1 YEAR
time = 'all'

# Define your existing paths
time_length = data_set(time)
input_path = f"{root}/fasw{time_length}/processed_rolling_windows/"

# COMMAND ----------

# MAGIC %skip
# MAGIC dbutils.fs.rm(f"{root}/experiments/mlp", recurse=True)
# MAGIC dbutils.fs.mkdirs(f"{root}/experiments/mlp")
# MAGIC

# COMMAND ----------

root = "dbfs:/student-groups/Group_02_01"
data_BASE_DIR = root
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# COMMAND ----------

subdirs = [f for f in dbutils.fs.ls(f"{root}/fasw{time_length}/processed_rolling_windows") if f.isDir()]
windows = len(subdirs)
N = int(windows / 2)
N

# COMMAND ----------

time_length = data_set(time)
train_input_path = f"{root}/fasw{time_length}/processed_rolling_windows/window_1_train"

val_input_path = f"{root}/fasw{time_length}/processed_rolling_windows/window_1_val"


df1 = spark.read.format("parquet").load(train_input_path)
dfv1 = spark.read.format("parquet").load(val_input_path)
display(df1)

# COMMAND ----------

df1.columns

# COMMAND ----------

# DBTITLE 1,ENCODE ORIGIN DEST
# MAGIC %skip
# MAGIC from pyspark.sql import functions as F
# MAGIC
# MAGIC # Calculate encodings from training data only
# MAGIC origin_encoding = df1.groupBy("ORIGIN").agg(F.mean("ARR_DEL15").alias("origin_delay_rate"))
# MAGIC dest_encoding = df1.groupBy("DEST").agg(F.mean("ARR_DEL15").alias("dest_delay_rate"))
# MAGIC
# MAGIC # Get overall mean for missing airports
# MAGIC overall_delay_rate = df1.select(F.mean("ARR_DEL15")).first()[0]
# MAGIC
# MAGIC # Join to train
# MAGIC df1 = df1.join(origin_encoding, on="ORIGIN", how="left")
# MAGIC df1 = df1.join(dest_encoding, on="DEST", how="left")
# MAGIC df1 = df1.fillna({'origin_delay_rate': overall_delay_rate, 'dest_delay_rate': overall_delay_rate})
# MAGIC
# MAGIC # Join to val (using train's encodings!)
# MAGIC dfv1 = dfv1.join(origin_encoding, on="ORIGIN", how="left")
# MAGIC dfv1 = dfv1.join(dest_encoding, on="DEST", how="left")
# MAGIC dfv1 = dfv1.fillna({'origin_delay_rate': overall_delay_rate, 'dest_delay_rate': overall_delay_rate})
# MAGIC
# MAGIC # # Add to your numerical features
# MAGIC # numerical_raw.extend(['origin_delay_rate', 'dest_delay_rate'])

# COMMAND ----------

# DBTITLE 1,FEATURE GROUPS

numerical_log = [
    'CRS_ELAPSED_TIME_log',
    'DISTANCE_log',
    'ORIGIN_ELEVATION_FT_log',
    'DEST_ELEVATION_FT_log',
    'lowest_cloud_ft_log',
    'HourlyWindGustSpeed_log',
    'crs_time_to_next_flight_diff_mins_log',
    'actual_to_crs_time_to_next_flight_diff_mins_clean_log'
]

numerical_raw = [
    #'ORIGIN_LAT',
    #'ORIGIN_LONG',
    #'DEST_LAT',
    #'DEST_LON',
    'overall_cloud_frac_0_1',
    'highest_cloud_ft',
    'HourlyAltimeterSetting',
    'HourlyWindSpeed',
    'origin_pagerank',
    'dest_pagerank',
    'origin_out_degree',
    'dest_in_degree',
    'prev_flight_arr_delay_clean',
    # 'origin_delay_rate',
    # 'dest_delay_rate'
]


categorical_ohe = [
    'OP_UNIQUE_CARRIER',
    'ORIGIN_STATE_ABR',
    'DEST_STATE_ABR',
    'ORIGIN_SIZE',
    'DEST_SIZE',
    'QUARTER',
    'MONTH',
    'DAY_OF_WEEK',
    'CRS_DEP_TIME_BLOCK',
    'CRS_ARR_TIME_BLOCK',
    'HourlyWindCardinalDirection'
]


binary_features = [
    'IS_US_HOLIDAY',
    'has_few',
    'has_sct',
    'has_bkn',
    'has_ovc',
    'light',
    'heavy',
    'thunderstorm',
    'rain_or_drizzle',
    'freezing_conditions',
    'snow',
    'hail_or_ice',
    'reduced_visibility',
    'spatial_effects',
    'unknown_precip'
]


### not using
categorical_high_card = [
    'ORIGIN',      # Use ORIGIN_idx - hundreds of airports
    'DEST'         # Use DEST_idx - hundreds of airports
]

exclude_features = [
    'YEAR',  # probably constant in your dataset
    'TAIL_NUM',  # too high cardinality
    'CRS_DEP_DATETIME_UTC',  # use time blocks instead
    'ARR_DEL15',  # TARGET
    'CARRIER_DELAY',  # target leakage
    'NAS_DELAY',  # target leakage
    'SECURITY_DELAY',  # target leakage
    'LATE_AIRCRAFT_DELAY'  # target leakage
]

# COMMAND ----------

# All numerical features (log + raw)
all_numerical = numerical_log + numerical_raw  # ~21 features

# After OHE, you'll get output columns like:
ohe_output_cols = [f"{col}_ohe" for col in categorical_ohe]  # ~12 OHE vector

# COMMAND ----------

scaled_num_vector_length = len(
    df1.select('scaled_num_vector').first()['scaled_num_vector']
)
ohe_vector_length = len(
    df1.select('ohe_vector').first()['ohe_vector']
)
print(f"scaled_num_vector length: {scaled_num_vector_length}")
print(f"ohe_vector length: {ohe_vector_length}")

# COMMAND ----------

# 1. Assemble numerical features
num_assembler = VectorAssembler(
    inputCols=all_numerical,  # your 21 features
    outputCol="num_vector2"
)

# 2. Scale numerical features  
scaler = StandardScaler(
    inputCol="num_vector2",
    outputCol="scaled_num_vector2"
)

# 3. OHE categorical features
ohe_stages = []
ohe_output_cols = []
for col in categorical_ohe:
    indexer = StringIndexer(inputCol=col, outputCol=f"{col}_idx2", handleInvalid="keep")
    encoder = OneHotEncoder(inputCol=f"{col}_idx2", outputCol=f"{col}_ohe2")
    ohe_stages.extend([indexer, encoder])
    ohe_output_cols.append(f"{col}_ohe2")

# 4a. Assemble binary features into vector
binary_assembler = VectorAssembler(
    inputCols=binary_features,
    outputCol="binary_vector2",
    handleInvalid="keep"
)

# 4b. Assemble final vector (all vectors now)
final_assembler = VectorAssembler(
    inputCols=["scaled_num_vector2"] + ohe_output_cols + ["binary_vector2"],
    outputCol="features2",
    handleInvalid="keep"
)

# COMMAND ----------

df1.columns

# COMMAND ----------

# Build the full pipeline
stages = ohe_stages + [num_assembler, scaler, binary_assembler, final_assembler]
pipeline = Pipeline(stages=stages)

# Fit on training data
pipeline_model = pipeline.fit(df1)

# # Check your input dimension
# sample_row = train_ready.select("features2").first()
# input_dim = len(sample_row["features2"])
# print(f"MLP input dimension: {input_dim}")

# COMMAND ----------

# DBTITLE 1,MLP Model
# PySpark ML components
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

# MLflow for experiment tracking
import mlflow
import mlflow.spark

# Standard Python
from datetime import datetime
from pyspark.sql import functions as F

# Use your existing variables: time_length, root, input_path are already defined above

# Count number of windows by counting items with '_train' in the name
all_items = dbutils.fs.ls(input_path)
train_items = [f for f in all_items if '_train' in f.name]
N = len(train_items)
print(f"Found {N} training windows for time={time} ({time_length})")

# Create timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Set experiment
experiment_name = f"/Workspace/Shared/Team_2_1/mlp/mlp{time_length}"

#experiment_id = mlflow.create_experiment(
   # name=experiment_name,
    #artifact_location=f"{root}/experiments/mlp_{time}"
#)
mlflow.set_experiment(experiment_name)

# Define evaluators (matching XGBoost)
target_col = "ARR_DEL15"

recall_evaluator = MulticlassClassificationEvaluator(
    labelCol=target_col, 
    predictionCol="prediction", 
    metricName="recallByLabel", 
    metricLabel=1
)
precision_evaluator = MulticlassClassificationEvaluator(
    labelCol=target_col,
    predictionCol="prediction",
    metricName="precisionByLabel",
    metricLabel=1
)
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol=target_col, 
    predictionCol="prediction", 
    metricName="f1"
)
f2_evaluator = MulticlassClassificationEvaluator(
    labelCol=target_col,
    predictionCol="prediction",
    metricName="fMeasureByLabel",
    metricLabel=1,
    beta=2.0
)
pr_auc_evaluator = BinaryClassificationEvaluator(
    labelCol=target_col, 
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR" 
)
roc_auc_evaluator = BinaryClassificationEvaluator(
    labelCol=target_col,
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
acc_evaluator = MulticlassClassificationEvaluator(
    labelCol=target_col,
    predictionCol="prediction",
    metricName="accuracy"
)

# Loop through rolling windows
for i in range(N):
    train_path = f"{input_path}window_{i+1}_train"
    val_path = f"{input_path}window_{i+1}_val"
    
    print(f"\n{'='*60}")
    print(f"PROCESSING WINDOW {i+1}/{N}")
    print(f"{'='*60}")
    
    # Load data
    train_df = spark.read.parquet(train_path)
    val_df = spark.read.parquet(val_path)

    # # Fit pipeline on this window's training data
    # pipeline_model = pipeline.fit(train_df)
    train_ready = pipeline_model.transform(train_df)
    val_ready = pipeline_model.transform(val_df)
    
    # Get input dimension from transformed data
    sample_row = train_ready.select("features2").first()
    input_dim = sample_row[0].size
    print(f"Input dimension: {input_dim}")
    
    # Define MLP architecture
    layers = [input_dim, 32, 2]
    
    # Create MLP classifier
    mlp = MultilayerPerceptronClassifier(
        featuresCol="features2",
        labelCol=target_col,
        layers=layers,
        maxIter=100,
        stepSize=0.005,
        blockSize=128,
        seed=42,
        tol=1e-6
    )
    
    # MLflow run for this window
    with mlflow.start_run(run_name=f"MLP_window_{i+1}"):
        
        # Log parameters
        mlflow.log_param("window", i+1)
        mlflow.log_param("time_length", time_length)
        mlflow.log_param("layers", str(layers))
        mlflow.log_param("max_iter", 300)
        mlflow.log_param("step_size", 0.01)
        mlflow.log_param("block_size", 128)
        mlflow.log_param("input_dim", input_dim)
        
        # Train
        print(f"Training MLP on window {i+1}...")
        model = mlp.fit(train_ready)
        print("✓ Training complete")
        
        # Predictions
        print("Making predictions...")
        train_pred = model.transform(train_ready)
        val_pred = model.transform(val_ready)
        
        # Calculate metrics - TRAIN
        train_recall = recall_evaluator.evaluate(train_pred)
        train_precision = precision_evaluator.evaluate(train_pred)
        train_f1 = f1_evaluator.evaluate(train_pred)
        train_f2 = f2_evaluator.evaluate(train_pred)
        train_pr_auc = pr_auc_evaluator.evaluate(train_pred)
        train_roc_auc = roc_auc_evaluator.evaluate(train_pred)
        train_acc = acc_evaluator.evaluate(train_pred)
        
        # Calculate metrics - VAL
        val_recall = recall_evaluator.evaluate(val_pred)
        val_precision = precision_evaluator.evaluate(val_pred)
        val_f1 = f1_evaluator.evaluate(val_pred)
        val_f2 = f2_evaluator.evaluate(val_pred)
        val_pr_auc = pr_auc_evaluator.evaluate(val_pred)
        val_roc_auc = roc_auc_evaluator.evaluate(val_pred)
        val_acc = acc_evaluator.evaluate(val_pred)
        
        # Log metrics - TRAIN
        mlflow.log_metric("train_recall", train_recall)
        mlflow.log_metric("train_precision", train_precision)
        mlflow.log_metric("train_f1_score", train_f1)
        mlflow.log_metric("train_f2_score", train_f2)
        mlflow.log_metric("train_pr_auc", train_pr_auc)
        mlflow.log_metric("train_roc_auc", train_roc_auc)
        mlflow.log_metric("train_accuracy", train_acc)
        
        # Log metrics - VAL
        mlflow.log_metric("recall", val_recall)
        mlflow.log_metric("precision", val_precision)
        mlflow.log_metric("f1_score", val_f1)
        mlflow.log_metric("f2_score", val_f2)
        mlflow.log_metric("pr_auc", val_pr_auc)
        mlflow.log_metric("roc_auc", val_roc_auc)
        mlflow.log_metric("accuracy", val_acc)
        
        # Log model
        mlflow.spark.log_model(model, "mlp_model")
        
        # Print results for this window
        print(f"\n{'='*60}")
        print(f"RESULTS - WINDOW {i+1}")
        print(f"{'='*60}")
        print(f"{'Metric':<20} {'Train':<15} {'Validation':<15}")
        print(f"{'-'*60}")
        print(f"{'Accuracy':<20} {train_acc:<15.4f} {val_acc:<15.4f}")
        print(f"{'Recall':<20} {train_recall:<15.4f} {val_recall:<15.4f}")
        print(f"{'Precision':<20} {train_precision:<15.4f} {val_precision:<15.4f}")
        print(f"{'F1 Score':<20} {train_f1:<15.4f} {val_f1:<15.4f}")
        print(f"{'F2 Score':<20} {train_f2:<15.4f} {val_f2:<15.4f}")
        print(f"{'PR-AUC':<20} {train_pr_auc:<15.4f} {val_pr_auc:<15.4f}")
        print(f"{'ROC-AUC':<20} {train_roc_auc:<15.4f} {val_roc_auc:<15.4f}")
        print(f"{'='*60}\n")

print(f"\n{'='*60}")
print(f"COMPLETED ALL {N} WINDOWS")
print(f"{'='*60}")
mlflow.end_run()

# COMMAND ----------

print(f"MLP layers config: {layers}")
print(f"Expected first layer (input): {layers[0]}")
print(f"Actual feature dimension: {train_ready.select('features2').first()[0].size}")

# COMMAND ----------

# MAGIC %md
# MAGIC To do:
# MAGIC - what is ml flow things
# MAGIC - update to loop thru rolling windows
# MAGIC - record changes somewhere for training

# COMMAND ----------

# ============================================
# LOAD BEST MODEL FROM MLFLOW
# ============================================

import mlflow
import mlflow.pytorch
from mlflow import spark as mlflow_spark

best_model_uri = 'runs:/33d37261dc5c4e62abe5609cd34d61f2/mlp_model'

best_model = mlflow_spark.load_model(best_model_uri)

print("\nLoaded best model from MLflow:")


# COMMAND ----------

test_path = "dbfs:/student-groups/Group_02_01/fasw/processed_train_test/test"

# COMMAND ----------

test_df = spark.read.parquet(test_path)
print(f"  Raw test rows: {test_df.count():,}")

# Use the same pipeline_model you fit earlier on df1
test_ready = pipeline_model.transform(test_df)
print(f"  ✓ Test data preprocessed\n")


# COMMAND ----------

test_pred = best_model.transform(test_ready)



# COMMAND ----------

target_col = "ARR_DEL15"

from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator

recall_evaluator = MulticlassClassificationEvaluator(
    labelCol=target_col, 
    predictionCol="prediction",
    metricName="recallByLabel",
    metricLabel=1
)
precision_evaluator = MulticlassClassificationEvaluator(
    labelCol=target_col,
    predictionCol="prediction",
    metricName="precisionByLabel",
    metricLabel=1
)
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol=target_col,
    predictionCol="prediction",
    metricName="f1"
)
f2_evaluator = MulticlassClassificationEvaluator(
    labelCol=target_col,
    predictionCol="prediction",
    metricName="fMeasureByLabel",
    beta=2.0,
    metricLabel=1
)
pr_auc_evaluator = BinaryClassificationEvaluator(
    labelCol=target_col,
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)
roc_auc_evaluator = BinaryClassificationEvaluator(
    labelCol=target_col,
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
acc_evaluator = MulticlassClassificationEvaluator(
    labelCol=target_col,
    predictionCol="prediction",
    metricName="accuracy"
)


# COMMAND ----------

rec = recall_evaluator.evaluate(test_pred)
prec = precision_evaluator.evaluate(test_pred)
f1 = f1_evaluator.evaluate(test_pred)
f2 = f2_evaluator.evaluate(test_pred)
pr_auc = pr_auc_evaluator.evaluate(test_pred)
roc_auc = roc_auc_evaluator.evaluate(test_pred)
acc = acc_evaluator.evaluate(test_pred)

print(rec, prec, f1, f2, pr_auc, roc_auc, acc)


# COMMAND ----------

print("\n================= TEST METRICS =================")
print(f"{'Metric':<15} {'Value':>10}")
print("----------------------------------------")
print(f"{'Accuracy':<15} {acc:>10.4f}")
print(f"{'Recall':<15} {rec:>10.4f}")
print(f"{'Precision':<15} {prec:>10.4f}")
print(f"{'F1 Score':<15} {f1:>10.4f}")
print(f"{'F2 Score':<15} {f2:>10.4f}")
print(f"{'PR-AUC':<15} {pr_auc:>10.4f}")
print(f"{'ROC-AUC':<15} {roc_auc:>10.4f}")
print("==============================================\n")
