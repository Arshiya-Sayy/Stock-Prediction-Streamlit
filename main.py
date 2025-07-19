import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("📈 Stock Prediction App")

stocks = ("AAPL", "GOOG", "MSFT")
selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Years of Prediction:", 1 , 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)

    # Flatten multiindex columns
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [' '.join(col).strip() if col[1] else col[0] for col in data.columns.values]
    
    return data

data_load_state = st.text("Load Data.....")
data = load_data(selected_stock)
data_load_state.text("Loading data....done!")

# Handle dynamic column names
date_col = "Date"
open_col = f"Open {selected_stock}"
close_col = f"Close {selected_stock}"

required_cols = [date_col, open_col, close_col]
missing = [col for col in required_cols if col not in data.columns]
if missing:
    st.error(f"Missing columns in data: {missing}")
    st.stop()

# Drop rows with missing essential data
data.dropna(subset=required_cols, inplace=True)

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data[date_col], y=data[open_col], name='Stock Open'))
    fig.add_trace(go.Scatter(x=data[date_col], y=data[close_col], name='Stock Close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Forecasting with Prophet
df_train = data[[date_col, close_col]].rename(columns={date_col: "ds", close_col: "y"})

# Safely convert columns
df_train['ds'] = pd.to_datetime(df_train['ds'], errors='coerce')
df_train['y'] = pd.to_numeric(df_train['y'], errors='coerce')
df_train.dropna(inplace=True)

if df_train.empty:
    st.error("No valid data available for forecasting. Try another stock.")
    st.stop()

st.subheader("Cleaned Training Data")
st.write(df_train.head())

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader('Forecast data')
st.write(forecast.tail())

st.write('forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)