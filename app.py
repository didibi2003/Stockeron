pip install yfinance

import streamlit as st
import yfinance as yf
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score

# Function to load or download SP500 data
def load_data():
    if os.path.exists("sp500.csv"):
        sp500 = pd.read_csv("sp500.csv", index_col=0)
    else:
        sp500 = yf.Ticker("^GSPC")
        sp500 = sp500.history(period="max")
        sp500.to_csv("sp500.csv")
    sp500.index = pd.to_datetime(sp500.index)
    return sp500

# Function to train the RandomForestClassifier model
def train_model(data):
    # Remove unnecessary columns
    data = data.copy()
    del data["Dividends"]
    del data["Stock Splits"]
    
    # Create target variable
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    
    # Filter data from 1990-01-01
    data = data.loc["1990-01-01":].copy()
    
    # Features
    predictors = ["Close", "Volume", "Open", "High", "Low"]
    
    # Split data into train and test
    train = data.iloc[:-100]
    test = data.iloc[-100:]
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
    model.fit(train[predictors], train["Target"])
    
    return model

# Function to make predictions
def predict(data, model):
    data = data.copy()
    predictors = ["Close", "Volume", "Open", "High", "Low"]
    preds = model.predict(data[predictors])
    preds = pd.Series(preds, index=data.index, name="Predictions")
    return preds

# Main function to run the Streamlit app
def main():
    st.title('SP500 Predictions App')
    
    # Load data
    sp500 = load_data()
    
    # Train model
    model = train_model(sp500)
    
    # Make predictions
    predictions = predict(sp500, model)
    
    # Display predictions
    st.write(predictions)

if __name__ == '__main__':
    main()
