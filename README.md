# Sales Forecasting - Machine Learning Project

## Intern Details
| Field | Value |
|-------|-------|
| **Full Name** | Gaurav |
| **Email** | gjangir477@gmail.com |
| **Codtech Intern ID** | CT-2712 |
| **Company** | Codtech IT Solutions |
| **Project** | Sales Forecasting |

---

## Project Overview
This project implements time series forecasting models to predict future sales using real-world data from Kaggle's Walmart Sales Forecast dataset.

## Objectives
- Predict weekly sales using historical data
- Compare multiple forecasting models (ARIMA, Exponential Smoothing)
- Generate future sales forecasts for business planning

## Dataset
- **Source**: Kaggle - Walmart Sales Forecast
- **Size**: 421,570 rows (aggregated to 143 weekly records)
- **Features**: Date, Weekly_Sales, IsHoliday
- **Date Range**: February 2010 - October 2012

## Methodology

### 1. Data Loading & Preprocessing
- Loaded Walmart sales data from Kaggle
- Aggregated daily data to weekly frequency
- Handled missing values and outliers

### 2. Exploratory Data Analysis (EDA)
- Time series visualization
- Distribution analysis
- Rolling statistics (mean, standard deviation)
- Trend and seasonality identification

### 3. Models Implemented

| Model | Description | Use Case |
|-------|-------------|----------|
| **ARIMA** | AutoRegressive Integrated Moving Average | Trend + stationary data |
| **Exponential Smoothing** | Holt-Winters method | Seasonal patterns |

### 4. Model Evaluation
- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **R²** (Coefficient of Determination)
- **MAPE** (Mean Absolute Percentage Error)

## Results

| Model | RMSE | MAE | MAPE | Best For |
|-------|------|-----|------|----------|
| ARIMA | 2,076,852 | 1,620,371 | 3.55% | Short-term trends |
| Exp Smoothing | 2,703,222 | 2,265,129 | 4.93% | Seasonal patterns |

**Best Model**: ARIMA (lowest RMSE)

## Files Generated
- `sales_forecasting.py` - Main analysis script
- `raw_data.csv` - Raw sales data
- `model_comparison.csv` - Model performance metrics
- `future_forecast.csv` - 12-week future forecast
- `eda_analysis.png` - Exploratory data analysis plots
- `arima_forecast.png` - ARIMA model visualization
- `exp_smoothing_forecast.png` - Exponential Smoothing visualization
- `forecast_comparison.png` - Model comparison plot
- `future_forecast.png` - Future sales forecast

## How to Run
```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels kagglehub
python sales_forecasting.py
```

## Dependencies
- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- kagglehub (for data download)
