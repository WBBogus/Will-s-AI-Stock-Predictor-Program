# Will-s-AI-Stock-Predictor-Program
The AI Stock Predictor Program is a machine learning-based tool that forecasts stock prices using historical data and inflation trends. This program leverages Yahoo Finance (yfinance) for stock data retrieval and Facebook Prophet for time series forecasting, providing users with an estimate of future stock prices along with an accuracy metric. This project was completed as a way for me to start learning machine learning so please forgive the excessive comments, it's how I learn the best.

# Features
- Fetches historical stock data from Yahoo Finance.
- Uses Prophet to predict stock prices based on past trends.
- Incorporates inflation rates as an additional regressor for improved accuracy.
- Evaluates prediction accuracy using Mean Absolute Percentage Error (MAPE).
- Allows users to specify the stock ticker symbol and prediction period.
- Visualizes forecasted stock prices with Matplotlib.

# How It Works
- The user inputs a stock ticker symbol (e.g., AAPL for Apple).
- The program retrieves historical stock price data for the specified prediction period.
- The data is formatted for Prophet, removing time zone issues.
- A machine learning model is trained with stock price and inflation rate data.
- The model predicts future stock prices and compares them to actual values.
- The accuracy of the prediction is displayed.
- A graph of the predicted vs. actual stock prices is generated.

# Requirements
- pandas
- yfinance
- matplotlib
- prophet
- sklearn

# Example Input
Please enter the stock symbol of the company you wish to predict: AAPL  
Please enter the time period you wish to predict (e.g., 1y, 2y, 3y): 2y

# Example Output
Stock Predictor Accuracy for AAPL: 87.45%
