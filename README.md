# flight_delay_predictions
End To End ML PIpeline for Flight Delay Predictions - Data Pipeline and Transformer Models

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
