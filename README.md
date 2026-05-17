# Daily Temperature Time Series Forecasting

A lightweight univariate time series forecasting project  
using the **official train/test split** from the Delhi Climate dataset.

## Dataset
- Source: [Daily Climate Time Series Data](https://www.kaggle.com/datasets/sumanthvrao/daily-climate-time-series-data)
- Train: 2013–2016  
- Test: 2017 (held-out)

Features:
- `date`
- `meantemp` (target)

## Problem Statement
Predict today's average temperature  
based on **yesterday's temperature only**.

## Methodology
- Feature: lag-1 (yesterday's temperature)
- No random split (used official test set)
- Model: Random Forest Regressor
- Evaluation: MAE, RMSE

## Results
| Metric | Value |
|------|------|
| MAE  | ~X.X °C |
| RMSE | ~X.X °C |

## Observations
The predicted curve closely follows the overall trend of the actual temperature.  
However, a small **phase shift** can be observed —  
the forecast slightly lags behind rapid temperature changes.

This is expected behavior for a lag-1 autoregressive model:  
it learns smooth patterns well, but reacts with delay to sudden jumps.

## Visualizations
- Actual vs Predicted Temperature
- Error Distribution

## Usage
- For learning

### Possible Improvements
- Adding multi-step lag features
- Incorporating humidity and wind speed
- Using rolling statistics or seasonal indicators