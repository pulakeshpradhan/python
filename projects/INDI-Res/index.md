# INDI-Res  
**A Time Series of Reservoir Area, Water Level, and Storage in India Derived from High-Resolution Multi-Satellite Observations**

---

## Overview

**INDI-Res** is a geospatial data and analysis project that provides a consistent, high-resolution time series of **reservoir surface area, water level, and storage dynamics across India**.  
The dataset is derived from **multi-satellite Earth observation data**, enabling long-term monitoring of surface water resources at national and sub-national scales.

This project is designed to support:
- Hydrological and water resource assessments  
- Climate variability and drought studies  
- Agricultural water management and irrigation planning  
- Reservoir operation and policy-relevant analysis  

---

## Key Features

- 📡 **Multi-satellite integration** (optical and/or radar-based observations)
- 🗺️ **High spatial resolution** reservoir surface area mapping
- 📈 **Time series of water level and storage estimates**
- 🇮🇳 **Nationwide coverage across India**
- 🔁 **Reproducible research pipeline** using Python and Jupyter notebooks

---

## Repository Structure

```text
INDI-Res/
│
├── README.md
├── LICENSE
├── pyproject.toml
├── environment.yml
├── .gitignore
│
├── data/
│   ├── raw/              # Original satellite and ancillary datasets
│   ├── interim/          # Preprocessed but non-final data
│   ├── processed/        # Final reservoir area, level, and storage products
│   └── external/         # Third-party datasets (e.g., DEM, reservoir boundaries)
│
├── notebooks/
│   ├── 00_exploration/           # Initial data inspection and QA
│   ├── 01_preprocessing/         # Satellite data preprocessing
│   ├── 02_feature_engineering/   # Area–elevation–storage relationships
│   ├── 03_modeling/              # Water level and storage estimation
│   ├── 04_evaluation/            # Validation and uncertainty analysis
│   └── README.md
│
├── src/
│   └── indi_res/
│       ├── __init__.py
│       ├── config.py
│       ├── data/
│       │   ├── load.py
│       │   ├── preprocess.py
│       │   └── utils.py
│       ├── features/
│       │   ├── build_features.py
│       │   └── scaling.py
│       ├── models/
│       │   ├── train.py
│       │   ├── predict.py
│       │   └── evaluate.py
│       ├── visualization/
│       │   └── plots.py
│       └── utils/
│           └── io.py
│
├── scripts/
│   ├── run_preprocessing.py
│   ├── train_model.py
│   └── evaluate_model.py
│
├── experiments/
│   ├── exp_001_baseline/
│   │   ├── config.yaml
│   │   └── results.json
│   └── exp_002_multisatellite/
│
├── results/
│   ├── figures/
│   ├── tables/
│   └── metrics/
│
├── docs/
│   ├── methodology.md
│   ├── data_description.md
│   └── model_details.md
│
└── tests/
    ├── test_data.py
    ├── test_models.py
    └── test_features.py
