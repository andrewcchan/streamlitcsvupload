import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
# create random x and y values
st.title("Fitbit Step Forecasting")
start_date = pd.to_datetime('2024-01-01')
dates = pd.date_range(start=start_date, periods=1000, freq='D')
y = [30 * np.sin(2 * np.pi * i / 100) + 10 * np.random.randn() for i, _ in enumerate(dates)]
y_hat = [30 * np.sin(2 * np.pi * i / 100) for i, _ in enumerate(dates)]
df = pd.DataFrame({'Date':dates, 'step count':y, 'predicted step count':y_hat})
fig = px.scatter(df, x='Date', y=['step count', 'predicted step count'], color_discrete_map={'predicted step count':'orange'})
fig.update_yaxes(title_text='Step count')
st.plotly_chart(fig, use_container_width=True)


    # https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html