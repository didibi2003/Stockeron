# Install necessary libraries


import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import hashlib
import types

def hash_function(obj):
    if isinstance(obj, types.FunctionType):
        return hashlib.sha256(obj.__code__.co_code).digest()
    else:
        raise TypeError("Unsupported object type for hashing.")

# Define Streamlit app
def main():
    st.title("Stock Price Prediction App")
    
    # Sidebar - Input parameters
    st.sidebar.header("Input Parameters")
    ticker = st.sidebar.text_input("Enter Stock Ticker Symbol", value='AAPL')
    num_years = st.sidebar.slider("Number of Years of Historical Data", 1, 10, 5)
    
    # Load data
    data = load_data(ticker, num_years)
    
    if data is not None:
        st.write("### Historical Stock Data")
        st.write(data.head())
        
        # Preprocess data
        X_train, X_test, y_train, y_test = preprocess_data(data)
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        evaluate_model(model, X_test, y_test)
        
        # Prediction
        st.write("### Make Prediction")
        prediction_date = st.date_input("Select Date for Prediction", pd.to_datetime('today'))
        prediction = predict(model, data, prediction_date)
        st.write(f"Predicted Closing Price on {prediction_date}: ${prediction:.2f}")

# Load historical stock data
@st.cache_data(hash_funcs={types.FunctionType: hash_function})

def load_data(ticker, num_years):
    end_date = pd.to_datetime('today')
    start_date = end_date - pd.DateOffset(years=num_years)
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        return data.copy()
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# Preprocess data
def preprocess_data(data):
    data['Date'] = data.index
    data['Date'] = data['Date'].astype('int64') // 10**9 # Convert to Unix timestamp
    X = data[['Date']]
    y = data['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Train model
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Evaluate model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.write("### Model Evaluation")
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R^2 Score: {r2:.2f}")
# Make prediction
def predict(model, data, date):
    prediction_date = pd.to_datetime(date)
    unix_timestamp = prediction_date.timestamp()
    prediction_date = prediction_date.strftime('%Y-%m-%d')
    prediction_input = np.array([[unix_timestamp]])
    prediction = model.predict(prediction_input)[0]
    return prediction

if __name__ == '__main__':
    main()
