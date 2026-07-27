Soil Fertility Index Forecasting

> Predicting agricultural soil fertility 6 months ahead using machine learning and time-series models.

## Overview

Agricultural productivity depends heavily on soil fertility, which varies seasonally based on environmental conditions, weather patterns, and soil composition. Traditional soil testing is expensive, time-consuming, and provides only a snapshot of current conditions.

This project uses machine learning to forecast soil fertility index values **6 months ahead** based on historical environmental data, enabling farmers and agricultural planners to:

- **Predict future soil fertility** before the growing season
- **Optimize fertilization schedules** based on forecasted conditions  
- **Identify sites at risk** of low fertility in advance
- **Reduce soil testing costs** through predictive modeling

The system combines satellite imagery, climate data, soil properties, and environmental indicators to generate accurate fertility forecasts across different geographical locations.

---

## Key Features

- **Multi-horizon time-series forecasting** with configurable lookback and forecast windows
- **Multiple model architectures** (XGBoost, LSTM, BiLSTM with Attention)
- **Composite fertility index** derived from Nitrogen, pH, NDVI, organic matter, and soil moisture
- **Spatial-temporal modeling** incorporating coordinates, elevation, and seasonal patterns
- **Robust preprocessing pipeline** handling outliers, missing values, and feature engineering
- **Reproducible training** with configurable random seeds
- **Comprehensive evaluation metrics** (RMSE, MAE, R², MAPE)

---

## Performance Highlights

### Task 1: Single-Step Prediction (XGBoost)

Using Random Forest and XGBoost with 5-fold time-series cross-validation:

```
✓ XGBoost is available

RF CV R²:   0.6790 ± 0.0742
RF CV MAE:  0.0323 ± 0.0044
RF CV RMSE: 0.0445 ± 0.0055

XGB CV R²:   0.8728 ± 0.0525
XGB CV MAE:  0.0177 ± 0.0049
XGB CV RMSE: 0.0276 ± 0.0059

XGB - Test: MAE=0.0132  RMSE=0.0222  R²=0.9286  MAPE=2.70%
```

### Task 2: Multi-Step Forecasting (3 & 6 Months Ahead)

Best results across different configurations:

| Configuration | Model | Test R² | Test RMSE | Test MAE |
|--------------|-------|---------|-----------|----------|
| **Lookback=18, Horizon=6** | BiLSTM+Attention | 0.9536 | 0.0226 | 0.0187 |
| **Lookback=18, Horizon=3** | XGBoost | 0.9691 | 0.0186 | 0.0150 |
| **Lookback=12, Horizon=6** | LSTM | 0.9772 | 0.0160 | 0.0128 |
| **Lookback=12, Horizon=3** | BiLSTM+Attention | 0.9676 | 0.0191 | 0.0149 |

**Optimal Configuration:** Lookback=12 months, Horizon=6 months → **Test R² = 0.9772**

---

## Installation

### Prerequisites

- Python 3.8+
- pip or conda package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/soil-fertility-forecasting.git
cd soil-fertility-forecasting

# Create virtual environment
python -m venv capstone_venv
source capstone_venv/bin/activate  # On Windows: capstone_venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Data Preprocessing

```bash
# Run the preprocessing notebook
jupyter notebook preprocess_combined.ipynb
```

This notebook:
- Loads raw agricultural data
- Handles missing values and outliers
- Creates derived features (NDVI, green fraction, log transformations)
- Normalizes pH values to optimal ranges
- Extracts geographical coordinates
- Splits data into train/validation/test sets

**Output:** Cleaned datasets saved to `csv/` directory

### 2. Task 1: Single-Step Prediction

```bash
# Train XGBoost and Random Forest models
jupyter notebook xgboost.ipynb
```

**Output:** 
- Trained models saved to `models/xgb_model.pkl`, `models/rf_model.joblib`
- Cross-validation metrics printed to console

### 3. Task 2: Time-Series Forecasting

#### Step A: Prepare Sequential Data

```bash
python prepare_task2_data.py \
  --input csv/preprocessed_data.csv \
  --lookback 12 \
  --horizon 6 \
  --output task2_processed_data_l12_h6
```

**Parameters:**
- `--lookback`: Number of historical months to use as input (e.g., 12 = past year)
- `--horizon`: Number of months ahead to predict (e.g., 6 = half year ahead)

**Output:**
- `task2_processed_data_l12_h6.csv` – Sequence data
- `task2_processed_data_l12_h6_metadata.csv` – Sample metadata
- `task2_processed_data_l12_h6_scaler.pkl` – Fitted StandardScaler

#### Step B: Train Models

```bash
python task2_train_model.py \
  --lookback 12 \
  --horizon 6 \
  --epochs 60 \
  --batch-size 64 
```

**Options:**
- `--lookback`, `--horizon`: Must match data preparation step
- `--tune-bilstm`: Enable Bayesian hyperparameter tuning for BiLSTM
- `--epochs`: Training epochs for neural network models
- `--batch-size`: Batch size for training

**Output:**
- `task2_model_metrics_l12_h6.json` – Performance metrics
- `task2_model_predictions_l12_h6.csv` – Predictions vs actual values
- Trained models saved to `models/` directory

---

## Project Structure

```
Capstone_Code/
├── preprocess_combined.ipynb        # Data cleaning and feature engineering
├── xgboost.ipynb                    # Task 1: Single-step prediction models
├── prepare_task2_data.py            # Task 2: Sequential data preparation
├── task2_train_model.py             # Task 2: Train LSTM/BiLSTM/XGBoost
├── task2_validate.py                # Validation utilities
├── requirements.txt                 # Python dependencies
│
├── csv/                             # Preprocessed datasets
│   ├── X_train_clean.csv
│   ├── X_val_clean.csv
│   ├── X_test_clean.csv
│   └── y_train.csv, y_val.csv, y_test.csv
│
├── models/                          # Trained model artifacts
│   ├── xgb_model.pkl
│   ├── rf_model.joblib
│   └── lstm_l12_h6_best.h5
│
├── task2_processed/                 # Sequential data for time-series models
│   ├── task2_processed_data_l12_h3.csv
│   ├── task2_processed_data_l12_h6.csv
│   └── ...
│
├── task2_model_metrics_*.json       # Model performance results
├── task2_model_predictions_*.csv    # Prediction outputs
│
└── eda_plots/                       # Exploratory data analysis visualizations
```

---

## Dataset Schema

### Input Features

The model uses the following features:

| Feature | Description | Type |
|---------|-------------|------|
| `Rain_log` | Log-transformed rainfall (mm) | Numeric |
| `Temp` | Temperature (°C) | Numeric |
| `LST` | Land Surface Temperature (°C) | Numeric |
| `SoilMoisture` | Volumetric soil moisture (%) | Numeric |
| `NDVI` | Normalized Difference Vegetation Index | Numeric (0-1) |
| `green_fraction` | Vegetation greenness fraction | Numeric (0-1) |
| `Clay` | Clay content (%) | Numeric |
| `Nitrogen_log` | Log-transformed soil nitrogen | Numeric |
| `pH` | Soil pH level | Numeric (0-14) |
| `BulkDensity` | Soil bulk density (g/cm³) | Numeric |
| `Elevation` | Elevation above sea level (m) | Numeric |
| `Slope_log` | Log-transformed terrain slope | Numeric |
| `AOD` | Aerosol Optical Depth | Numeric |
| `NO2_log` | Log-transformed NO₂ concentration | Numeric |
| `SO2_log` | Log-transformed SO₂ concentration | Numeric |
| `Month_Sin`, `Month_Cos` | Cyclical month encoding | Numeric (-1 to 1) |
| `longitude`, `latitude` | Geographical coordinates | Numeric |
| `Silt` | Silt content (%) | Numeric |

### Target Variable

**Fertility Index** – Composite metric calculated as:

```python
Fertility Index = 0.25 × Nitrogen + 0.25 × pH + 0.20 × Soil_Moisture + 0.15 × Clay + 0.15 × Silt
```

Normalized to 0-1 scale using Min-Max scaling.

---

## Model Architecture

### Task 1: XGBoost Regressor

```python
XGBRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    tree_method='hist',
    random_state=42,
    n_jobs=-1,
)
```

### Task 2: BiLSTM with Attention

```
Input: (batch, timesteps, features)
    ↓
Bidirectional LSTM (128 units, return_sequences=True)
    ↓
Dropout (0.3)
    ↓
Bidirectional LSTM (64 units, return_sequences=True)
    ↓
Attention Layer (compute weighted context)
    ↓
Dense (64, relu) → Dropout (0.3)
    ↓
Dense (32, relu) → Dropout (0.2)
    ↓
Dense (1, linear) → Output
```

---

## Validation Strategy

### Task 1 (Single-Step)
- **5-Fold Time Series Cross-Validation** with `TimeSeriesSplit`
- Train/Val/Test split: 70% / 15% / 15%
- Temporal ordering preserved

### Task 2 (Multi-Step)
- **Rolling window validation** with consistent temporal gaps
- Lookback windows from `t-lag` to `t-1`
- Forecast target at `t+horizon`
- No data leakage between splits

---

## Reproducibility

All experiments use fixed random seeds:

```python
python task2_train_model.py --seed 42
```

Seeds are set for:
- NumPy random operations
- Python random module
- TensorFlow/Keras backend
- XGBoost internal sampling

---

## Usage Examples

### Example 1: Train with Different Window Sizes

```bash
# Short lookback, near-term forecast (3 months ahead)
python prepare_task2_data.py --lookback 6 --horizon 3
python task2_train_model.py --lookback 6 --horizon 3

# Long lookback, long-term forecast (6 months ahead)
python prepare_task2_data.py --lookback 18 --horizon 6
python task2_train_model.py --lookback 18 --horizon 6
```

### Example 2: Hyperparameter Tuning

```bash
python task2_train_model.py \
  --lookback 12 \
  --horizon 6 \
  --tune-bilstm \
  --epochs 150
```

This runs Bayesian optimization over:
- LSTM units (32, 64, 128, 256)
- Dropout rates (0.1, 0.2, 0.3, 0.4)
- Learning rates (0.001, 0.0005, 0.0001)

### Example 3: Export Predictions

```bash
python task2_train_model.py \
  --lookback 12 \
  --horizon 6 \
  --output-predictions predictions.csv
```

Output format:
```csv
site_id,date,actual,predicted,model
site_001,2026-01,0.742,0.738,BiLSTM+Attention
site_001,2026-02,0.756,0.751,BiLSTM+Attention
...
```

---

## Interpreting Results

### Metrics Explained

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **R²** | Coefficient of determination | % of variance explained (higher = better) |
| **RMSE** | Root Mean Squared Error | Average prediction error in original units |
| **MAE** | Mean Absolute Error | Average absolute deviation |
| **MAPE** | Mean Absolute Percentage Error | Average error as % of actual value |

### Decision Thresholds

| Fertility Index | Status | Action |
|-----------------|--------|--------|
| **> 0.70** | Optimal | Maintain current practices |
| **0.50 – 0.70** | Moderate | Targeted fertilization |
| **< 0.50** | Low | Intensive soil amendment |

---

## Technology Stack

- **XGBoost** – Gradient boosting for tabular data
- **TensorFlow/Keras** – Deep learning framework for LSTM/BiLSTM
- **scikit-learn** – Preprocessing, metrics, cross-validation
- **Pandas** – Data manipulation and time-series handling
- **NumPy** – Numerical operations
- **Matplotlib/Seaborn** – Visualization
- **SHAP** – Model interpretability and feature importance

---

## Research Context

This project was developed as a capstone research project at **PES University** under the guidance of **Prof. Nivedita Kasturi**.

**Project Title:**  
*Temporal Dynamics of Urban Fertility: A Machine Learning Framework for Urban Site Recommendation and Fertility Forecasting using Multi-Source Environmental Data*

**Key Contributions:**
- Developed composite fertility index from agronomic research
- Compared tree-based vs. deep learning approaches for time-series forecasting
- Achieved 97.7% R² accuracy on 6-month ahead predictions
- Demonstrated attention mechanism effectiveness for long-term dependencies

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

---

## License

MIT License

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{urban-fertility-forecasting-2026,
  title={Temporal Dynamics of Urban Fertility: A Machine Learning Framework for Urban Site Recommendation and Fertility Forecasting using Multi-Source Environmental Data},
  author={Abhranshu Sarkar, Chinthan K, Akhilesh M, Chethana K R, Nivedita Kasturi},
  year={2026},
  institution={PES University},
  type={Capstone Research Project}
}
```

---

## Contact

For questions or collaboration opportunities:

- **Email:** abhranshusarkar@outlook.com
- **GitHub:** [@sarkar-abhranshu](https://github.com/sarkar-abhranshu)
- **Email:** chink2425@gmail.com
- **Github:** [Chinthan k](https://github.com/chin123k)

---

## Acknowledgments

- Prof. Nivedita Kasturi for project guidance
- Open-source community for excellent ML libraries
