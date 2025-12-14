# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Tokenizer Transformer

# COMMAND ----------

# DBTITLE 1,IMPORTS
# ============================================
# IMPORTS
# ============================================

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from pyspark.sql.functions import col
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array

from sklearn.metrics import (
    precision_score, recall_score, f1_score, fbeta_score,
    average_precision_score, roc_auc_score
)

import mlflow
import mlflow.pytorch

print("✓ All imports successful")

# COMMAND ----------

# DBTITLE 1,CONFIG ML FLOW
# ============================================
# MLFLOW CONFIGURATION
# ============================================

# Root path configuration
ROOT = "dbfs:/student-groups/Group_02_01"
time = 'all'  # Options: 3, 6, 12, 'all'

def data_set(time):
    if time == 3:
        return "_3m", "3_months"
    elif time == 6:
        return "_6m", "6_months"
    elif time == 12:
        return "_1y", "1_year"
    elif time == 'all':
        return "", "full_dataset"
    else:
        raise ValueError("time must be 3, 6, 12, or 'all'")

time_length, timeframe_label = data_set(time)
INPUT_PATH = f"{ROOT}/fasw{time_length}/processed_rolling_windows/no_scaling"

# MLflow experiment
EXPERIMENT_NAME = f"/Shared/Team_2_1/ft_transformer_{timeframe_label}"
mlflow.set_experiment(EXPERIMENT_NAME)

print("=" * 80)
print("CONFIGURATION")
print("=" * 80)
print(f"Experiment: {EXPERIMENT_NAME}")
print(f"Timeframe: {timeframe_label}")
print(f"Data path: {INPUT_PATH}")
print("=" * 80)

# COMMAND ----------

# DBTITLE 1,DEFINE FOCAL LOSS
import torch.nn.functional as TF

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean"):
        """
        alpha: weighting for the positive class (0–1). Higher => more weight on positives.
        gamma: focusing parameter. Higher => more focus on hard examples.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: raw model outputs, shape (N,) or (N,1)
        targets: 0/1 floats, same shape as logits
        """
        # make sure shapes match
        targets = targets.view_as(logits)

        # standard BCE per example (no reduction)
        bce = TF.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # probabilities for the true class
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # focal term
        focal_factor = (1 - p_t) ** self.gamma

        # alpha weighting (more weight for positives)
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        else:
            alpha_t = 1.0

        loss = alpha_t * focal_factor * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


# COMMAND ----------

# DBTITLE 1,DEFINE FEATURES
# ============================================
# FEATURE GROUPS
# ============================================

# Numerical features (log-transformed)
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

# Numerical features (raw)
numerical_raw = [
    'overall_cloud_frac_0_1',
    'highest_cloud_ft',
    'HourlyAltimeterSetting',
    'HourlyWindSpeed',
    'origin_pagerank',
    'dest_pagerank',
    'origin_out_degree',
    'dest_in_degree',
    'prev_flight_arr_delay_clean'
]

# Categorical features
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

# Binary features
binary_features = [
    'IS_US_HOLIDAY',
    'has_few', 'has_sct', 'has_bkn', 'has_ovc',
    'light', 'heavy', 'thunderstorm', 'rain_or_drizzle',
    'freezing_conditions', 'snow', 'hail_or_ice',
    'reduced_visibility', 'spatial_effects', 'unknown_precip'
]

all_numerical = numerical_log + numerical_raw

print(f"✓ Feature groups defined:")
print(f"  Numerical: {len(all_numerical)}")
print(f"  Categorical: {len(categorical_ohe)}")
print(f"  Binary: {len(binary_features)}")

# Count rolling windows
subdirs = [f for f in dbutils.fs.ls(INPUT_PATH) if f.isDir()]
train_windows = [w for w in subdirs if 'train' in w.name]
N_WINDOWS = len(train_windows)
print(f"  Rolling windows: {N_WINDOWS}")

# COMMAND ----------

# DBTITLE 1,PREPROCESS FEATURES
# ============================================
# BUILD PREPROCESSING PIPELINE
# ============================================

def build_preprocessing_pipeline(numerical_log, numerical_raw, categorical_ohe, binary_features):
    """Build Spark ML Pipeline for feature preprocessing."""
    stages = []
    all_numerical = numerical_log + numerical_raw
    
    # Scale numerical features (mean=0, std=1)
    for num_col in all_numerical:
        assembler = VectorAssembler(
            inputCols=[num_col],
            outputCol=f"{num_col}_vec",
            handleInvalid="keep"
        )
        scaler = StandardScaler(
            inputCol=f"{num_col}_vec",
            outputCol=f"{num_col}_scaled2",
            withMean=True,
            withStd=True
        )
        stages.extend([assembler, scaler])
    
    # Index categorical features
    for cat_col in categorical_ohe:
        indexer = StringIndexer(
            inputCol=cat_col,
            outputCol=f"{cat_col}_idx2",
            handleInvalid="keep"
        )
        stages.append(indexer)
    
    return Pipeline(stages=stages)

preprocessing_pipeline = build_preprocessing_pipeline(
    numerical_log, numerical_raw, categorical_ohe, binary_features
)

print(f"✓ Preprocessing pipeline built ({len(preprocessing_pipeline.getStages())} stages)")

# COMMAND ----------

# DBTITLE 1,FIT PIPELINE
# ============================================
# FIT PIPELINE ON SAMPLE
# ============================================

print("Fitting preprocessing pipeline...")

# Load and sample window 1 for fitting
train_path_1 = f"{INPUT_PATH}/window_1_train"
df_train_1 = spark.read.parquet(train_path_1)
df_sample = df_train_1.sample(fraction=0.1, seed=42)
sample_count = df_sample.count()

print(f"  Fitting on {sample_count:,} samples (10% of window 1)")

# Fit pipeline
fitted_pipeline = preprocessing_pipeline.fit(df_sample)
print("✓ Pipeline fitted")

# Extract categorical cardinalities
print("\nExtracting categorical cardinalities...")
cat_cardinalities = []
for cat_col in categorical_ohe:
    for stage in fitted_pipeline.stages:
        if hasattr(stage, 'getInputCol') and stage.getInputCol() == cat_col:
            n_labels = len(stage.labels) + 1  # +1 for unknown
            cat_cardinalities.append(n_labels)
            break

print(f"Initial cardinalities: {cat_cardinalities}")

# Fix time-based features for rolling windows
cat_cardinalities_fixed = []
for i, cat_col in enumerate(categorical_ohe):
    if cat_col == 'QUARTER':
        cat_cardinalities_fixed.append(4)  # Always 4 quarters
    elif cat_col == 'MONTH':
        cat_cardinalities_fixed.append(12)  # Always 12 months
    else:
        cat_cardinalities_fixed.append(cat_cardinalities[i])

cat_cardinalities = cat_cardinalities_fixed

print(f"Final cardinalities (fixed): {cat_cardinalities}")
print("✓ Categorical cardinalities extracted")

# COMMAND ----------

# DBTITLE 1,DATA EXTRACTION
# ============================================
# DATA EXTRACTION FUNCTION
# ============================================
from pyspark.sql import functions as SF
def prepare_for_pytorch(df, numerical_cols, categorical_cols, binary_cols):
    """
    Prepare DataFrame for PyTorch:
    - Extract scalars from DenseVectors (numerical features)
    - Cast categorical indices to int
    - Cast binary features to float
    """
    select_exprs = []
    
    # Extract numerical features from DenseVectors
    for col in numerical_cols:
        select_exprs.append(
            vector_to_array(SF.col(f"{col}_scaled2"))[0]
            .cast('double')
            .alias(f"{col}_num")
        )
    
    # Categorical indices
    for col in categorical_cols:
        select_exprs.append(
            F.col(f"{col}_idx2")
            .cast('int')
            .alias(f"{col}_cat")
        )
    
    # Binary features
    for col in binary_cols:
        select_exprs.append(
            F.col(col)
            .cast('float')
            .alias(f"{col}_bin")
        )
    
    # Target
    select_exprs.append(
        SF.col('ARR_DEL15')
        .cast('float')
        .alias('target')
    )
    
    return df.select(*select_exprs)

print("✓ Data extraction function defined")

# COMMAND ----------

# ============================================
# FT-TRANSFORMER MODEL
# ============================================

class FTTransformer(nn.Module):
    """
    Feature Tokenizer Transformer for tabular data.
    
    Architecture:
    1. Each feature → Token (embedding)
    2. Multi-head attention across tokens (3 blocks)
    3. CLS token → Classification output
    """
    
    def __init__(
        self,
        n_numerical,
        n_binary,
        categorical_cardinalities,
        d_model=192,
        n_heads=8,
        n_blocks=3,
        dropout=0.2,
        attention_dropout=0.2
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # ============================================
        # FEATURE TOKENIZATION
        # ============================================
        
        # Numerical features → single token
        self.numerical_tokenizer = nn.Linear(n_numerical, d_model)
        
        # Binary features → single token
        self.binary_tokenizer = nn.Linear(n_binary, d_model)
        
        # Categorical features → separate tokens (embeddings)
        self.categorical_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, d_model)
            for cardinality in categorical_cardinalities
        ])
        
        # CLS token for classification
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # ============================================
        # TRANSFORMER BLOCKS
        # ============================================
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_blocks
        )
        
        # ============================================
        # CLASSIFICATION HEAD
        # ============================================
        
        self.output_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
    
    def forward(self, numerical, categorical, binary):
        """
        Args:
            numerical: (batch_size, n_numerical)
            categorical: (batch_size, n_categorical)
            binary: (batch_size, n_binary)
        
        Returns:
            logits: (batch_size,)
        """
        batch_size = numerical.size(0)
        
        # ============================================
        # TOKENIZE FEATURES
        # ============================================
        
        tokens = []
        
        # CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens.append(cls_tokens)
        
        # Numerical token
        num_token = self.numerical_tokenizer(numerical).unsqueeze(1)
        tokens.append(num_token)
        
        # Categorical tokens
        for i, embedding in enumerate(self.categorical_embeddings):
            cat_token = embedding(categorical[:, i]).unsqueeze(1)
            tokens.append(cat_token)
        
        # Binary token
        bin_token = self.binary_tokenizer(binary).unsqueeze(1)
        tokens.append(bin_token)
        
        # Concatenate all tokens: (batch_size, n_tokens, d_model)
        x = torch.cat(tokens, dim=1)
        
        # ============================================
        # TRANSFORMER PROCESSING
        # ============================================
        
        x = self.transformer(x)
        
        # ============================================
        # CLASSIFICATION FROM CLS TOKEN
        # ============================================
        
        cls_output = x[:, 0, :]  # Take CLS token
        logits = self.output_layer(cls_output).squeeze(-1)
        
        return logits

print("✓ FTTransformer model defined")

# COMMAND ----------

# ============================================
# TRAINING CONFIGURATION
# ============================================

CONFIG = {
    # Training hyperparameters
    'epochs': 15,
    'batch_size': 4096,
    'val_batch_size': 8192,
    'learning_rate': 3e-4,
    'weight_decay': 1e-4,
    
    # FT-Transformer architecture
    'd_model': 192,
    'n_heads': 8,
    'n_blocks': 3,
    'dropout': 0.2,
    'attention_dropout': 0.2,
    
    # Data
    'sample_fraction': None,  # Set to 0.1 for 10% sampling (faster testing)
    
    # Baseline to beat
    'baseline_pr_auc': 0.6833,
    'baseline_f1': 0.6343
}

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 80)
print("TRAINING CONFIGURATION")
print("=" * 80)
print(f"Device: {device}")
for key, value in CONFIG.items():
    print(f"  {key:25s}: {value}")
print("=" * 80)

# COMMAND ----------

# DBTITLE 1,ROLLING WINDOW TRAINING WITH TRAIN + VAL METRICS
# ============================================
# ROLLING WINDOW TRAINING WITH TRAIN + VAL METRICS
# ============================================

import numpy as np

print("=" * 80)
print("STARTING ROLLING WINDOW TRAINING - FT-TRANSFORMER")
print("=" * 80)

all_results = []

for window_idx in range(1, N_WINDOWS + 1):
    
    print(f"\n{'='*80}")
    print(f"WINDOW {window_idx}/{N_WINDOWS}")
    print(f"{'='*80}")
    
    # ============================================
    # START MLFLOW RUN
    # ============================================
    with mlflow.start_run(run_name=f"ft_transformer_window_{window_idx}") as run:
        
        # Log hyperparameters
        mlflow.log_params(CONFIG)
        mlflow.log_param("window", window_idx)
        mlflow.set_tag("model_type", "FTTransformer")
        mlflow.set_tag("architecture", "transformer")
        
        # ============================================
        # LOAD DATA
        # ============================================
        
        train_path = f"{INPUT_PATH}/window_{window_idx}_train"
        val_path = f"{INPUT_PATH}/window_{window_idx}_val"
        
        print(f"\n[Loading Data]")
        df_train = spark.read.parquet(train_path)
        df_val = spark.read.parquet(val_path)
        
        if CONFIG['sample_fraction']:
            df_train = df_train.sample(fraction=CONFIG['sample_fraction'], seed=42)
            df_val = df_val.sample(fraction=CONFIG['sample_fraction'], seed=42)
        
        train_count = df_train.count()
        val_count = df_val.count()
        print(f"  Train: {train_count:,} samples")
        print(f"  Val: {val_count:,} samples")
        
        # ============================================
        # PREPROCESS
        # ============================================
        
        print(f"\n[Preprocessing]")
        df_train_processed = fitted_pipeline.transform(df_train)
        df_val_processed = fitted_pipeline.transform(df_val)
        
        df_train_ready = prepare_for_pytorch(
            df_train_processed, all_numerical, categorical_ohe, binary_features
        )
        df_val_ready = prepare_for_pytorch(
            df_val_processed, all_numerical, categorical_ohe, binary_features
        )
        
        print("  ✓ Features prepared")
        
        # ============================================
        # CONVERT TO PYTORCH
        # ============================================
        
        print(f"\n[Converting to PyTorch]")
        train_pdf = df_train_ready.toPandas()
        val_pdf = df_val_ready.toPandas()
        
        train_dataset = TensorDataset(
            torch.tensor(train_pdf[[f"{col}_num" for col in all_numerical]].values, dtype=torch.float32),
            torch.tensor(train_pdf[[f"{col}_cat" for col in categorical_ohe]].values, dtype=torch.long),
            torch.tensor(train_pdf[[f"{col}_bin" for col in binary_features]].values, dtype=torch.float32),
            torch.tensor(train_pdf['target'].values, dtype=torch.float32)
        )
        
        val_dataset = TensorDataset(
            torch.tensor(val_pdf[[f"{col}_num" for col in all_numerical]].values, dtype=torch.float32),
            torch.tensor(val_pdf[[f"{col}_cat" for col in categorical_ohe]].values, dtype=torch.long),
            torch.tensor(val_pdf[[f"{col}_bin" for col in binary_features]].values, dtype=torch.float32),
            torch.tensor(val_pdf['target'].values, dtype=torch.float32)
        )
        
        delay_rate = train_dataset.tensors[3].mean().item()
        print(f"  Delay rate: {delay_rate*100:.1f}%")
        
        mlflow.log_metrics({
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset),
            'delay_rate': delay_rate
        })
        
        del train_pdf, val_pdf
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=CONFIG['batch_size'], 
            shuffle=True,
            num_workers=4,          # try 4–8 depending on cores
            pin_memory=True,        # speeds up host→GPU copies
            persistent_workers=True # avoids worker respawn overhead
            )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=CONFIG['val_batch_size'], 
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        
        print(f"  ✓ Train batches: {len(train_loader)}")
        print(f"  ✓ Val batches: {len(val_loader)}")
        
        # ============================================
        # INITIALIZE MODEL
        # ============================================
        
        print(f"\n[Initializing FT-Transformer]")
        model = FTTransformer(
            n_numerical=len(all_numerical),
            n_binary=len(binary_features),
            categorical_cardinalities=cat_cardinalities,
            d_model=CONFIG['d_model'],
            n_heads=CONFIG['n_heads'],
            n_blocks=CONFIG['n_blocks'],
            dropout=CONFIG['dropout'],
            attention_dropout=CONFIG['attention_dropout']
        ).to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")
        print(f"  Architecture: {CONFIG['n_blocks']} blocks × {CONFIG['n_heads']} heads")
        print(f"  Model dimension: {CONFIG['d_model']}")
        mlflow.log_param("total_parameters", total_params)
        
        # We already computed delay_rate above:
        # delay_rate = train_dataset.tensors[3].mean().item()

        # # Option 1: fixed alpha (simple, works well)
        # alpha = 0.25
        # gamma = 2.0

        # Option 2 (optional): make alpha depend on class balance
        pos_frac = delay_rate  # already computed
        alpha = min(0.85, max(0.5, 1.0 - pos_frac))  # e.g. 0.8-ish for 18% positives
        gamma = 2.0

        print(f"  Using FocalLoss with alpha={alpha:.2f}, gamma={gamma:.1f}")
        mlflow.log_param("focal_alpha", alpha)
        mlflow.log_param("focal_gamma", gamma)

        criterion = FocalLoss(alpha=alpha, gamma=gamma).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=CONFIG['learning_rate'], 
            weight_decay=CONFIG['weight_decay']
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=CONFIG['epochs']
        )
        
        use_amp = True
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        torch.set_float32_matmul_precision("high")  # PyTorch 2.x


        # ============================================
        # TRAINING LOOP
        # ============================================
        
        print(f"\n[Training for {CONFIG['epochs']} epochs]")
        print(f"{'Epoch':>5} | {'Loss':>8} | {'Train PR':>9} | {'Val PR':>9} | {'Gap':>8} | {'Val F1':>8} | {'Val F2':>8}")
        print("-" * 80)
        
        best_pr_auc = 0
        best_metrics = None
        
        for epoch in range(CONFIG['epochs']):
            
            # ============================================
            # TRAIN PHASE
            # ============================================
            model.train()
            train_loss = 0
            train_preds = []
            train_targets = []

            for num_feat, cat_feat, bin_feat, target in train_loader:
                num_feat = num_feat.to(device, non_blocking=True)
                cat_feat = cat_feat.to(device, non_blocking=True)
                bin_feat = bin_feat.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                    logits = model(num_feat, cat_feat, bin_feat)
                    loss = criterion(logits, target)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()

                # Collect predictions on CPU for metrics
                preds = torch.sigmoid(logits).detach().cpu().numpy()
                train_preds.extend(preds)
                train_targets.extend(target.detach().cpu().numpy())
            scheduler.step()
            
            train_loss /= len(train_loader)
            
            # Calculate training metrics
            train_preds = np.array(train_preds)
            train_targets = np.array(train_targets)
            train_pred_classes = (train_preds > 0.5).astype(int)
            
            train_metrics = {
                'train_loss': train_loss,
                'train_recall': recall_score(train_targets, train_pred_classes, zero_division=0),
                'train_precision': precision_score(train_targets, train_pred_classes, zero_division=0),
                'train_f1': f1_score(train_targets, train_pred_classes, zero_division=0),
                'train_f2': fbeta_score(train_targets, train_pred_classes, beta=2, zero_division=0),
                'train_pr_auc': average_precision_score(train_targets, train_preds),
                'train_roc_auc': roc_auc_score(train_targets, train_preds)
            }
            
            # ============================================
            # VALIDATION PHASE
            # ============================================
            model.eval()
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for num_feat, cat_feat, bin_feat, target in val_loader:
                    num_feat = num_feat.to(device, non_blocking=True)
                    cat_feat = cat_feat.to(device, non_blocking=True)
                    bin_feat = bin_feat.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)

                    with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                        logits = model(num_feat, cat_feat, bin_feat)

                    preds = torch.sigmoid(logits).detach().cpu().numpy()
                    all_preds.extend(preds)
                    all_targets.extend(target.detach().cpu().numpy())
                        
            # Calculate validation metrics
            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)
            pred_classes = (all_preds > 0.5).astype(int)
            
            val_metrics = {
                'val_recall': recall_score(all_targets, pred_classes, zero_division=0),
                'val_precision': precision_score(all_targets, pred_classes, zero_division=0),
                'val_f1': f1_score(all_targets, pred_classes, zero_division=0),
                'val_f2': fbeta_score(all_targets, pred_classes, beta=2, zero_division=0),
                'val_pr_auc': average_precision_score(all_targets, all_preds),
                'val_roc_auc': roc_auc_score(all_targets, all_preds),
                'learning_rate': optimizer.param_groups[0]['lr']
            }
            
            # ============================================
            # LOG METRICS TO MLFLOW
            # ============================================
            
            all_metrics = {**train_metrics, **val_metrics}
            mlflow.log_metrics(all_metrics, step=epoch)
            
            # Calculate gap
            gap = train_metrics['train_pr_auc'] - val_metrics['val_pr_auc']
            
            # Print progress
            print(f"{epoch+1:>5} | {train_loss:>8.4f} | {train_metrics['train_pr_auc']:>9.4f} | {val_metrics['val_pr_auc']:>9.4f} | {gap:>+8.4f} | {val_metrics['val_f1']:>8.4f} | {val_metrics['val_f2']:>8.4f}")
            
            # Track best model
            if val_metrics['val_pr_auc'] > best_pr_auc:
                best_pr_auc = val_metrics['val_pr_auc']
                best_metrics = {**train_metrics, **val_metrics}
            
         
        
        # ============================================
        # LOG BEST METRICS & MODEL
        # ============================================
        
        mlflow.log_metrics({
            'best_val_pr_auc': best_pr_auc,
            'best_val_f1': best_metrics['val_f1'],
            'best_val_f2': best_metrics['val_f2'],
            'best_train_pr_auc': best_metrics['train_pr_auc'],
            'best_train_f1': best_metrics['train_f1'],
            'best_train_val_gap': best_metrics['train_pr_auc'] - best_pr_auc
        })
        
        # Save model to MLflow
        mlflow.pytorch.log_model(model, "model")
        
        # Save results
        all_results.append({
            'window': window_idx,
            'best_pr_auc': best_pr_auc,
            'best_metrics': best_metrics,
            'run_id': run.info.run_id
        })
        
        # Print window summary
        train_val_gap = best_metrics['train_pr_auc'] - best_pr_auc
        overfitting_status = "⚠️ Overfitting" if train_val_gap > 0.1 else "✓ Good"
        beat_baseline = "🎉 BEAT!" if best_pr_auc > CONFIG['baseline_pr_auc'] else ""
        
        print("\n" + "=" * 80)
        print(f"WINDOW {window_idx} COMPLETE")
        print("=" * 80)
        print(f"  Best Val PR-AUC:   {best_pr_auc:.4f} (baseline: {CONFIG['baseline_pr_auc']}) {beat_baseline}")
        print(f"  Best Val F1:       {best_metrics['val_f1']:.4f} (baseline: {CONFIG['baseline_f1']})")
        print(f"  Best Val F2:       {best_metrics['val_f2']:.4f}")
        print(f"  Best Train PR-AUC: {best_metrics['train_pr_auc']:.4f}")
        print(f"  Train/Val Gap:     {train_val_gap:+.4f} {overfitting_status}")
        print(f"  MLflow Run ID:     {run.info.run_id}")
        print("=" * 80)
        
        # Cleanup
        del train_dataset, val_dataset, train_loader, val_loader, model
        torch.cuda.empty_cache()

print("\n✓ All windows trained!")

# COMMAND ----------

# DBTITLE 1,TRAINING CURVES AND VIZ
# ============================================
# TRAINING CURVES & VISUALIZATIONS
# ============================================

import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.tracking import MlflowClient

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (18, 10)

print("=" * 80)
print("GENERATING TRAINING CURVES")
print("=" * 80)

client = MlflowClient()

# Retrieve metrics from MLflow
all_runs_data = []
for window_result in all_results:
    run_id = window_result['run_id']
    
    metrics_history = {}
    metric_names = ['train_loss', 'train_pr_auc', 'val_pr_auc', 'val_f1', 'val_f2', 
                    'val_precision', 'val_recall', 'train_f1']
    
    for metric_name in metric_names:
        metric_history = client.get_metric_history(run_id, metric_name)
        metrics_history[metric_name] = [(m.step, m.value) for m in metric_history]
    
    all_runs_data.append({
        'window': window_result['window'],
        'run_id': run_id,
        'metrics': metrics_history
    })

print(f"✓ Retrieved metrics from {len(all_runs_data)} runs")

# ============================================
# PLOT 1: TRAINING CURVES (6 SUBPLOTS)
# ============================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('FT-Transformer Training Metrics Across All Windows', fontsize=16, fontweight='bold')

# Plot 1: Training Loss
ax = axes[0, 0]
for run_data in all_runs_data:
    if run_data['metrics']['train_loss']:
        epochs, values = zip(*run_data['metrics']['train_loss'])
        ax.plot(epochs, values, marker='o', label=f"Window {run_data['window']}", linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Training Loss', fontsize=12)
ax.set_title('Training Loss by Epoch', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Train vs Val PR-AUC
ax = axes[0, 1]
baseline_pr_auc = CONFIG['baseline_pr_auc']
ax.axhline(y=baseline_pr_auc, color='red', linestyle='--', linewidth=2, label=f'Baseline ({baseline_pr_auc:.4f})')
for run_data in all_runs_data:
    if run_data['metrics']['train_pr_auc']:
        epochs_t, train_vals = zip(*run_data['metrics']['train_pr_auc'])
        epochs_v, val_vals = zip(*run_data['metrics']['val_pr_auc'])
        ax.plot(epochs_t, train_vals, marker='o', linestyle='-', label=f"W{run_data['window']} Train", linewidth=2)
        ax.plot(epochs_v, val_vals, marker='s', linestyle='--', label=f"W{run_data['window']} Val", linewidth=2, alpha=0.7)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('PR-AUC', fontsize=12)
ax.set_title('PR-AUC: Train vs Val', fontsize=14, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 3: Val F1 Score
ax = axes[0, 2]
baseline_f1 = CONFIG['baseline_f1']
ax.axhline(y=baseline_f1, color='red', linestyle='--', linewidth=2, label=f'Baseline ({baseline_f1:.4f})')
for run_data in all_runs_data:
    if run_data['metrics']['val_f1']:
        epochs, values = zip(*run_data['metrics']['val_f1'])
        ax.plot(epochs, values, marker='o', label=f"Window {run_data['window']}", linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('Validation F1 Score by Epoch', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: Val F2 Score
ax = axes[1, 0]
for run_data in all_runs_data:
    if run_data['metrics']['val_f2']:
        epochs, values = zip(*run_data['metrics']['val_f2'])
        ax.plot(epochs, values, marker='o', label=f"Window {run_data['window']}", linewidth=2)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('F2 Score', fontsize=12)
ax.set_title('Validation F2 Score by Epoch', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Precision vs Recall
ax = axes[1, 1]
for run_data in all_runs_data:
    if run_data['metrics']['val_precision']:
        epochs_p, precision = zip(*run_data['metrics']['val_precision'])
        epochs_r, recall = zip(*run_data['metrics']['val_recall'])
        ax.plot(epochs_p, precision, marker='o', linestyle='-', label=f"W{run_data['window']} Prec", linewidth=2)
        ax.plot(epochs_r, recall, marker='s', linestyle='--', label=f"W{run_data['window']} Rec", linewidth=2, alpha=0.7)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Precision vs Recall by Epoch', fontsize=14, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 6: Train/Val Gap
ax = axes[1, 2]
for run_data in all_runs_data:
    if run_data['metrics']['train_pr_auc'] and run_data['metrics']['val_pr_auc']:
        epochs_t, train_vals = zip(*run_data['metrics']['train_pr_auc'])
        epochs_v, val_vals = zip(*run_data['metrics']['val_pr_auc'])
        gaps = [t - v for t, v in zip(train_vals, val_vals)]
        ax.plot(epochs_t, gaps, marker='o', label=f"Window {run_data['window']}", linewidth=2)
ax.axhline(y=0.1, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Overfitting threshold')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Train - Val Gap', fontsize=12)
ax.set_title('Train/Val PR-AUC Gap (Overfitting Check)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
display(plt.gcf())
print("\n✓ Training curves generated")

# COMMAND ----------

# ============================================
# PERFORMANCE COMPARISON
# ============================================

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Extract best metrics
windows = [r['window'] for r in all_results]
best_pr_aucs = [r['best_pr_auc'] for r in all_results]
best_f1s = [r['best_metrics']['val_f1'] for r in all_results]
best_f2s = [r['best_metrics']['val_f2'] for r in all_results]
best_recalls = [r['best_metrics']['val_recall'] for r in all_results]
best_precisions = [r['best_metrics']['val_precision'] for r in all_results]

# Chart 1: PR-AUC and F1
ax = axes[0]
x = np.arange(len(windows))
width = 0.35

bars1 = ax.bar(x - width/2, best_pr_aucs, width, label='PR-AUC', color='steelblue', alpha=0.8)
bars2 = ax.bar(x + width/2, best_f1s, width, label='F1 Score', color='coral', alpha=0.8)

ax.axhline(y=CONFIG['baseline_pr_auc'], color='blue', linestyle='--', linewidth=2, alpha=0.5)
ax.axhline(y=CONFIG['baseline_f1'], color='red', linestyle='--', linewidth=2, alpha=0.5)

ax.set_xlabel('Window', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Best PR-AUC and F1 Score per Window', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'W{w}' for w in windows])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

# Chart 2: Precision, Recall, F2
ax = axes[1]
x = np.arange(len(windows))
width = 0.25

bars1 = ax.bar(x - width, best_precisions, width, label='Precision', color='green', alpha=0.8)
bars2 = ax.bar(x, best_recalls, width, label='Recall', color='orange', alpha=0.8)
bars3 = ax.bar(x + width, best_f2s, width, label='F2 Score', color='purple', alpha=0.8)

ax.set_xlabel('Window', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Best Precision, Recall, and F2 per Window', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'W{w}' for w in windows])
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
display(plt.gcf())
print("✓ Performance comparison charts generated")

# COMMAND ----------

# ============================================
# RESULTS SUMMARY TABLE
# ============================================

results_df = pd.DataFrame({
    'Window': windows,
    'PR-AUC': [f"{pr:.4f}" for pr in best_pr_aucs],
    'F1': [f"{f1:.4f}" for f in best_f1s],
    'F2': [f"{f2:.4f}" for f in best_f2s],
    'Precision': [f"{p:.4f}" for p in best_precisions],
    'Recall': [f"{r:.4f}" for r in best_recalls],
    'Beat Baseline': ['✓' if pr > CONFIG['baseline_pr_auc'] else '✗' for pr in best_pr_aucs],
    'MLflow Run ID': [r['run_id'] for r in all_results]
})

print("=" * 80)
print("DETAILED RESULTS TABLE")
print("=" * 80)
print(results_df.to_string(index=False))
print("=" * 80)

print(f"\n✓ FT-Transformer training complete!")
print(f"✓ All metrics logged to MLflow experiment: {EXPERIMENT_NAME}")
print(f"✓ View runs in Databricks MLflow UI")