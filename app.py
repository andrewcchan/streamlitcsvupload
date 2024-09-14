import streamlit as st
import plotly.express as px
import statsmodels.api as sm
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# create random x and y values
st.title("Time Series Forecasting")


# Generate data
N = 1000
P = 200
start_date = pd.to_datetime('2024-01-01')
dates = pd.date_range(start=start_date, periods=N, freq='D')
values = [30 * np.sin(2 * np.pi * i / 100) + 10 * np.random.randn() for i, _ in enumerate(dates)]

# Create DataFrame
df = pd.DataFrame({'date': dates, 'value': values})
df['type'] = ['train' for i in range(N)]
# Split data into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Fit model 
mse_values = []
for p in range(2, 5):
    for d in range(2):
        for q in range(2, 5):
            model = SARIMAX(train['value'], order=(p, d, q), seasonal_order=(1, 1, 1, 12))
            results = model.fit()
            mse_values.append([p, d, q, mean_squared_error(test['value'], results.forecast(steps=len(test)))])
mse_df = pd.DataFrame(mse_values, columns=['p', 'd', 'q', 'mse'])
st.write("mse_df")
st.write(mse_df)

min_mse_row = mse_df.loc[mse_df['mse'].idxmin()]
st.write("selected min_mse_row", min_mse_row)
model = SARIMAX(train['value'], order=(int(min_mse_row['p']), int(min_mse_row['d']), int(min_mse_row['q'])), seasonal_order=(1, 1, 1, 12))
results = model.fit()

# Predict next 20 time steps
future_dates = pd.date_range(start=train['date'].iloc[-1] + pd.Timedelta(days=1), periods=P, freq='D')
future_df = pd.DataFrame({'date': future_dates})
future_df.set_index('date', inplace=True)
future_predictions = results.forecast(P)
future_df['date'] = future_dates
future_df['value'] = future_predictions.values
future_df['type'] = ['prediction' for i in range(P)]

# Combine original data with predictions
combined_df = pd.concat([df, future_df])

# Plot results
colors = ['step count'] * N + ['predicted'] * P
fig = px.scatter(combined_df, x='date', y='value', color='type',color_discrete_sequence=["#1f77b4", "#ff7f0e"])
fig.update_layout(title='Time Series Prediction', xaxis_title='Date', yaxis_title='Value')


st.plotly_chart(fig, use_container_width=True)


# https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html


