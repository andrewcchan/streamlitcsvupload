import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import plotly.express as px
from darts import TimeSeries
from darts.models import ExponentialSmoothing


# Generate data
N = 365
P = 20
start_date = pd.to_datetime('2024-01-01')
dates = pd.date_range(start=start_date, periods=N, freq='D')
values = [30 * np.sin(2 * np.pi * i / 100) + 10 * np.random.randn() for i, _ in enumerate(dates)]

# Create DataFrame
df = pd.DataFrame({'date': dates, 'value': values})

# Split data into training and testing sets
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]


# Create a TimeSeries, specifying the time and value columns
series = TimeSeries.from_dataframe(df, "date", "value")

train_size = int(len(df) * 0.8)
train, val = series[:train_size], series[train_size:]


model = ExponentialSmoothing()
model.fit(train)
prediction = model.predict(len(val), num_samples=1000)
fig = px.line(series.pd_dataframe())
fig.add_scatter(x=prediction.time_index, y=prediction.low_quantile(0.5), mode="lines", name="forecast lower")
fig.add_scatter(x=prediction.time_index, y=prediction.high_quantile(0.6), mode="lines", name="forecast upper", fillcolor='rgba(0,0,0,0)', fill='tonexty')
fig.show()

