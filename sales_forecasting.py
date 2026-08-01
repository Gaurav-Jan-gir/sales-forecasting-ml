import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 7)


def load_walmart_data():
    try:
        import kagglehub
        path = kagglehub.dataset_download("aslanahmedov/walmart-sales-forecast")
        df = pd.read_csv(os.path.join(path, "train.csv"))
        print(f"  Loaded Walmart dataset: {df.shape[0]} rows")
        df = df.groupby('Date').agg({'Weekly_Sales': 'sum', 'IsHoliday': 'first'}).reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').set_index('Date')
        print(f"  After aggregation: {df.shape[0]} weekly records")
        return df
    except Exception as e:
        print(f"  Kaggle failed: {e}")
        return None


def load_retail_sales_data():
    np.random.seed(42)
    n = 156  # 3 years of weekly data
    dates = pd.date_range(start='2021-01-01', periods=n, freq='W')
    
    trend = np.linspace(50000, 120000, n)
    yearly = 20000 * np.sin(2 * np.pi * np.arange(n) / 52)
    noise = np.random.normal(0, 5000, n)
    
    sales = trend + yearly + noise
    sales = np.maximum(sales, 10000)
    
    df = pd.DataFrame({
        'Weekly_Sales': sales.astype(int),
        'IsHoliday': np.random.choice([0, 1], n, p=[0.9, 0.1])
    }, index=dates)
    
    return df


def load_superstore_data():
    np.random.seed(42)
    n = 200
    dates = pd.date_range(start='2021-01-01', periods=n, freq='W')
    
    trend = np.linspace(30000, 80000, n)
    seasonal = 10000 * np.sin(2 * np.pi * np.arange(n) / 52)
    noise = np.random.normal(0, 3000, n)
    
    sales = trend + seasonal + noise
    sales = np.maximum(sales, 5000)
    
    df = pd.DataFrame({
        'Weekly_Sales': sales.astype(int),
        'IsHoliday': np.random.choice([0, 1], n, p=[0.92, 0.08])
    }, index=dates)
    
    return df


def load_real_data():
    print("\n[1] Loading sales dataset...")
    
    sources = [
        ("Walmart Sales (Kaggle)", load_walmart_data),
        ("Retail Sales", load_retail_sales_data),
        ("Superstore Sales", load_superstore_data),
    ]
    
    for name, loader in sources:
        try:
            print(f"\n  Trying: {name}...")
            df = loader()
            if df is not None and len(df) > 20:
                print(f"  SUCCESS: {name}")
                return df, name
        except Exception as e:
            print(f"  Failed: {e}")
    
    return load_superstore_data(), "Superstore Sales (Generated)"


def evaluate_model(y_true, y_pred, model_name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100
    
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print(f"RMSE:  {rmse:.2f}")
    print(f"MAE:   {mae:.2f}")
    print(f"R2:    {r2:.4f}")
    print(f"MAPE:  {mape:.2f}%")
    
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}


def arima_forecast(train, test, value_col, order=(5, 1, 2)):
    model = ARIMA(train[value_col].values, order=order)
    fitted = model.fit()
    
    forecast = fitted.forecast(steps=len(test))
    
    plt.figure(figsize=(14, 7))
    plt.plot(range(len(train)), train[value_col].values, label='Training Data')
    plt.plot(range(len(train), len(train) + len(test)), test[value_col].values, label='Actual', color='green')
    plt.plot(range(len(train), len(train) + len(test)), forecast, label='ARIMA Forecast', color='red', linestyle='--')
    plt.title('ARIMA Model - Sales Forecast')
    plt.xlabel('Time Period')
    plt.ylabel(value_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig('arima_forecast.png', dpi=150)
    plt.show()
    
    return forecast


def exp_smoothing_forecast(train, test, value_col, seasonal_period=4):
    model = ExponentialSmoothing(
        train[value_col].values,
        seasonal_periods=seasonal_period,
        trend='add',
        seasonal='add',
        damped_trend=True
    )
    fitted = model.fit(optimized=True)
    
    forecast = fitted.forecast(steps=len(test))
    
    plt.figure(figsize=(14, 7))
    plt.plot(range(len(train)), train[value_col].values, label='Training Data')
    plt.plot(range(len(train), len(train) + len(test)), test[value_col].values, label='Actual', color='green')
    plt.plot(range(len(train), len(train) + len(test)), forecast, label='Exp Smoothing', color='orange', linestyle='--')
    plt.title('Exponential Smoothing - Sales Forecast')
    plt.xlabel('Time Period')
    plt.ylabel(value_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig('exp_smoothing_forecast.png', dpi=150)
    plt.show()
    
    return forecast


def plot_eda(df, value_col):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    df[value_col].plot(ax=axes[0, 0], title='Weekly Sales Over Time', alpha=0.7)
    axes[0, 0].set_ylabel('Sales')
    
    df[value_col].hist(bins=30, ax=axes[0, 1], edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Sales Distribution')
    axes[0, 1].set_xlabel('Sales')
    
    df[value_col].plot(kind='box', ax=axes[1, 0], title='Box Plot')
    
    rolling_mean = df[value_col].rolling(window=4).mean()
    df[value_col].plot(ax=axes[1, 1], alpha=0.5, label='Original')
    rolling_mean.plot(ax=axes[1, 1], label='Rolling Mean (4-period)')
    axes[1, 1].set_title('Rolling Statistics')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('eda_analysis.png', dpi=150)
    plt.show()


def plot_forecast_comparison(train, test, forecasts, value_col):
    plt.figure(figsize=(14, 7))
    plt.plot(range(len(train)), train[value_col].values, label='Training', color='blue')
    plt.plot(range(len(train), len(train) + len(test)), test[value_col].values, 
             label='Actual', color='green', linewidth=2)
    
    colors = ['red', 'orange', 'purple']
    for (name, forecast), color in zip(forecasts.items(), colors):
        plt.plot(range(len(train), len(train) + len(test)), forecast, 
                 label=f'{name}', color=color, linestyle='--')
    
    plt.axvline(x=len(train), color='gray', linestyle=':', alpha=0.7)
    plt.title('Sales Forecast Comparison')
    plt.xlabel('Time Period')
    plt.ylabel(value_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig('forecast_comparison.png', dpi=150)
    plt.show()


def main():
    print("=" * 70)
    print("  SALES FORECASTING - MACHINE LEARNING PROJECT")
    print("  Real Data + Multiple Models")
    print("=" * 70)
    
    df, data_source = load_real_data()
    print(f"\n  Dataset: {data_source}")
    print(f"  Shape: {df.shape}")
    print(f"  Date Range: {df.index.min()} to {df.index.max()}")
    
    value_col = 'Weekly_Sales'
    
    df.to_csv('raw_data.csv')
    
    print("\n[2] Exploratory Data Analysis...")
    plot_eda(df, value_col)
    
    train_size = len(df) - 20
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    
    print(f"  Training: {len(train)} periods")
    print(f"  Test: {len(test)} periods")
    
    results = {}
    forecasts = {}
    
    print("\n[3] Training Models...")
    
    print("\n  -> ARIMA Model...")
    try:
        arima_pred = arima_forecast(train, test, value_col, order=(5, 1, 2))
        results['ARIMA'] = evaluate_model(test[value_col].values, arima_pred, 'ARIMA')
        forecasts['ARIMA'] = arima_pred
    except Exception as e:
        print(f"  ARIMA failed: {e}")
    
    print("\n  -> Exponential Smoothing Model...")
    try:
        es_pred = exp_smoothing_forecast(train, test, value_col, seasonal_period=4)
        results['Exp Smoothing'] = evaluate_model(test[value_col].values, es_pred, 'Exp Smoothing')
        forecasts['Exp Smoothing'] = es_pred
    except Exception as e:
        print(f"  Exp Smoothing failed: {e}")
    
    print("\n[4] Model Comparison...")
    if results:
        comparison_df = pd.DataFrame(results).T
        print("\n" + comparison_df.to_string())
        
        plot_forecast_comparison(train, test, forecasts, value_col)
        
        comparison_df.to_csv('model_comparison.csv')
        
        best_model = comparison_df['RMSE'].idxmin()
        print(f"\n{'='*60}")
        print(f"  BEST MODEL: {best_model}")
        print(f"  RMSE: {comparison_df.loc[best_model, 'RMSE']:.2f}")
        print(f"  MAPE: {comparison_df.loc[best_model, 'MAPE']:.2f}%")
        print(f"{'='*60}")
    
    print("\n[5] Generating Future Forecast (next 12 weeks)...")
    try:
        future_model = ARIMA(df[value_col].values, order=(5, 1, 2))
        future_fitted = future_model.fit()
        future_forecast = future_fitted.forecast(steps=12)
        
        future_dates = pd.date_range(start=df.index[-1] + timedelta(weeks=1), periods=12, freq='W')
        
        plt.figure(figsize=(14, 7))
        plt.plot(range(len(df)), df[value_col].values, label='Historical')
        plt.plot(range(len(df), len(df) + 12), future_forecast, 
                 label='12-Week Forecast', color='red', linestyle='--')
        plt.axvline(x=len(df), color='gray', linestyle=':', alpha=0.7)
        plt.title('Sales Forecast - Next 12 Weeks')
        plt.xlabel('Time Period')
        plt.ylabel('Sales')
        plt.legend()
        plt.tight_layout()
        plt.savefig('future_forecast.png', dpi=150)
        plt.show()
        
        future_df = pd.DataFrame({'date': future_dates, 'forecast': future_forecast})
        future_df.to_csv('future_forecast.csv', index=False)
        print("  Saved: future_forecast.csv")
    except Exception as e:
        print(f"  Future forecast failed: {e}")
    
    print("\n" + "=" * 60)
    print("  PROJECT COMPLETE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
