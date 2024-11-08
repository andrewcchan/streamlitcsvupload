import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import plotly.express as px
import matplotlib.pyplot as plt
from darts.models import ExponentialSmoothing
from darts import TimeSeries
from darts.metrics import mape
from darts.models import AutoARIMA
import streamlit as st
from darts.models import NaiveDrift

# create random x and y values
st.title("Time Series Forecasting")

# Generate data
N = 365
P = 20
Amplitude = st.slider('Amplitude', min_value=1, max_value=100, value=30)
Freq = st.slider('1/Frequency', min_value=1, max_value=500, value=100)

st.write(f"Amplitude: {Amplitude}")
st.write(f"1/Frequency: {Freq}")

# Rest of the code remains the same...
start_date = pd.to_datetime('2024-01-01')
dates = pd.date_range(start=start_date, periods=N, freq='D')
values = [Amplitude * np.sin(2 * np.pi * i / Freq) + 10 * np.random.randn() for i, _ in enumerate(dates)]

# Create DataFrame
df = pd.DataFrame({'date': dates, 'value': values})

# Split data into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]

# Create a TimeSeries, specifying the time and value columns
series = TimeSeries.from_dataframe(df, "date", "value")

train_size = int(len(df) * 0.8)
train, val = series[:train_size], series[train_size:]

model=AutoARIMA()
model.fit(train)
forecast = model.predict(len(val))
st.write(f"model {model} obtains MAPE: {mape(val, forecast):.2f}%")

model2=NaiveDrift()
model2.fit(train)
forecast2 = model2.predict(len(val))
st.write(f"model {model2} obtains MAPE: {mape(val, forecast2):.2f}%")

# plot
fig = px.line(series.pd_dataframe())
fig.add_scatter(x=forecast.data_array()['date'].values, y=forecast.values().squeeze(),name='AutoARIMA')
fig.add_scatter(x=forecast.data_array()['date'].values, y=forecast2.values().squeeze(),name='NaiveDrift')
st.plotly_chart(fig, use_container_width=True)
