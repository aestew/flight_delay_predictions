# flight_delay_predictions
End To End ML PIpeline for Flight Delay Predictions - Data Pipeline and Transformer Models

## Flight Delay

75 million rows of DOT flight delay data and 132 million rows of weather data from NOAA were joined using Apache Spark and analyzed to evaluate flight delay predictions under significant class imbalance and temporal distribution drift. Flight delay predictions require modeling complex interactions, including flight schedule, airline, flight network effects, temporal dependencies and external issues including maintenance, staffing and security. Here, we use three model types, graph and temporal features to compare predictions addressing imbalance through downsampling and focal loss.

## Write-Up

- **Feature Tokenizer Transformer Model Performance in Predicting Flight Delays**  
  Medium article detailing the modeling approach, class imbalance handling, and transformer results:  
  https://medium.com/@aestew/feature-tokenizer-transformer-model-performance-in-predicting-flight-delays-96126989e133


## Pipeline, Features & Modeling Contributions

### Amy Steward — Pipeline Co-Lead, Features & Transformer Models
- **Co-designed and implemented the large-scale Spark join** between **~75M DOT flight records and 130M+ NOAA weather observations**, including join logic, time alignment, and nearest-station matching to produce model-ready datasets.
- Contributed to **pipeline design and correctness**, including checkpointing decisions, reruns after schema/data changes, QC validation, and downstream stability.
- Built and integrated **graph-based airport network features (PageRank)** and **temporal lag features** with strict **leakage prevention logic** directly into the pipeline.
- Built, tuned, and evaluated **MLP and Feature Tokenizer Transformer models**, including GPU-based training and **focal loss** for class imbalance.
- Led **generalization analysis and blind testing**, and authored neural modeling results, tables, and figures for the final report and presentation.

### Kristen Lin — Pipeline Lead
- Led the design and implementation of the **end-to-end Spark data pipeline**, including ingestion, preprocessing, and scalable checkpointed workflows.
- Architected **time-aware cross-validation** (forward chaining → rolling windows) and model-ready dataset generation across multiple data horizons.
- Drove repeated **pipeline refactors and optimizations** to improve performance, stability, and reprocessing efficiency, including QC, null handling, and timestamp normalization.
- Authored and maintained **pipeline diagrams, ERDs, and documentation** reflecting evolving pipeline architecture.
