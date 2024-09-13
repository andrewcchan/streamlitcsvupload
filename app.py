import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import forecast as fc
import statsmodels.api as sm


N = 1000
# create random x and y values
st.title("Fitbit Step Forecasting")
start_date = pd.to_datetime('2024-01-01')
dates = pd.date_range(start=start_date, periods=N , freq='D')
y = [30 * np.sin(2 * np.pi * i / 100) + 10 * np.random.randn() for i, _ in enumerate(dates)]
y_hat = [30 * np.sin(2 * np.pi * i / 100) for i, _ in enumerate(dates)]
df = pd.DataFrame({'Date':dates, 'step count':y, 'predicted step count':y_hat})
fig = px.scatter(df, x='Date', y=['step count', 'predicted step count'], color_discrete_map={'predicted step count':'orange'})
fig.update_yaxes(title_text='Step count')
st.plotly_chart(fig, use_container_width=True)

# Forecast
# Forecast using SARIMAX
endog = df['step count']
model = sm.tsa.statespace.SARIMAX(endog)
results = model.fit(disp=False)
forecast = results.forecast(steps=100)
forecast_df = pd.DataFrame({'Date':pd.date_range(start=dates[-1]+pd.Timedelta(days=1), periods=100, freq='D'), 'step count':forecast})
combined_df = pd.concat([df, forecast_df])
colors = ['step count'] * N + ['predicted'] * 100
fig = px.scatter(combined_df, x='Date', y='step count', color=colors)
fig.update_yaxes(title_text='Step count')
st.plotly_chart(fig, use_container_width=True)


# https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html


