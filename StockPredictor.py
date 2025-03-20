# Will Boguslawski
# Stock Predictor Program

import pandas as pd # to create dataframes
import yfinance as yf # to get stock data
import matplotlib.pyplot as plt # to create plot graph
from prophet import Prophet # to create ML model
import datetime as dt # to get current date and time
from sklearn.metrics import mean_absolute_percentage_error # to calculate error

# Define the stock ticker symbol (e.g., AAPL = Apple), its an abbreviation
print("\nWelcome to Will's Stock Predictor Program!")
ticker = input("Please enter the stock symbol of the company you wish to predict: ")
predictionPeriod = input("Please enter the time period you wish to predict (e.g., 1y, 2y, 3y): ")

# Create an object that represents company stock ticker and allows retrieve various types of financial data related to that stock.
stock = yf.Ticker(ticker)
data = stock.history(period=predictionPeriod) # get the last how many years of data.

# Since data returns a DataFrame where the date is the index instead of being a regular column. We need to move the index back into a column.
# this is important because Prophet requires a ds (date) column explicitly. If the date remains as an index, Prophet won't be able to recognize it.

# yfinance provides many columns (Open, High, Low, Volume, etc.), but we only need Date and Close Price for prediction. 
# Prophet has strict requirements for column names. ds is the dateime column and y is the numerical value prophet will predict which is stock price.
# so we must rename the Date and Close columns to ds and y respectively.

# Prophet does not support time zones in the ds column, yfinance returns timestamps with time zones which causes errors.
# to fix this we must remove the timezone so Prophet accepts the ds column.

# Reset index and prepare data for Prophet
data.reset_index(inplace=True) # moves the index (which contains the dates) back into a regular column.
dataFrame = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'}) 
dataFrame['ds'] = dataFrame['ds'].dt.tz_localize(None)  # Remove timezone

# avg inflation rate over past 10 years
inflation_rate_data = [0.02] * len(dataFrame)  

# Print the first few rows to verify
print(data.head())

# Add inflation rate as additional regressor
dataFrame['inflation_rate'] = inflation_rate_data

# Where the model is created and trained
model = Prophet(yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False)

# Add regressor to model
model.add_regressor('inflation_rate')

model.fit(dataFrame)

# Splitting Data for Accuracy Testing
# 90% of the stock data is used for training, 10% is used for testing (validation).
split_idx = int(len(dataFrame) * 0.9)  # Finds the index where the split should happen (90% of the dataset)
test_df = dataFrame.iloc[split_idx:]  # Selects the last 10% of the data as the test set

# Creates future timestamps (for the same number of days as the test set), neccesaary for Prophet to make predictions.
future = model.make_future_dataframe(periods=len(test_df))

future['inflation_rate'] = inflation_rate_data[-1]  # Use the last known inflation rate value for the future
future['inflation_rate'] = future['inflation_rate'].ffill()  # Forward fill to match length

# Trained Prophet model analyzes past trends to predict future prices.
forecast = model.predict(future)

# Extracting & Comparing Predictions
# forecast[['ds', 'yhat']] selects date (ds) and predicted stock price (yhat).
# .iloc[-len(test_df):] ensures we only take predictions for the test period
predicted = forecast[['ds', 'yhat']].iloc[-len(test_df):]
actual = test_df[['ds', 'y']] # Extracts the actual stock prices for the test period

# Merges the actual stock prices (y) with predicted prices (yhat)
# Allows direct comparison between actual & predicted values
merged_df = pd.merge(dataFrame, forecast[['ds', 'yhat']], on='ds', how='inner')

# Calculate Mean Absolute Percentage Error (MAPE)
# Measures how far the predicted prices are from actual prices
mape = mean_absolute_percentage_error(merged_df['y'], merged_df['yhat']) * 100  # Convert to percentage
accuracy = 100 - mape  # Calculate accuracy, higher is better

# Print the accuracy before showing the graph
print(f"\n📈 Stock Predictor Accuracy for {ticker}: {accuracy:.2f}%")

# Plot the forecasted stock prices
model.plot(forecast)
plt.title(f"{ticker} Stock Price Prediction\nAccuracy: {accuracy:.2f}%")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()