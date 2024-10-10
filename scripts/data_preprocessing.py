import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

def preprocess_data(ticker):
    # Load data
    df = pd.read_csv(f'data/{ticker}.csv')
    
    # Sort by date
    df = df.sort_values('Date')
    
    # Reset index
    df.reset_index(drop=True, inplace=True)
    
    # Handle missing values
    df.ffill(inplace=True)
    
    # Feature selection (using 'Close' price)
    data = df[['Close']].values
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Save scaler
    np.save('models/scaler.npy', scaler)
    
    # Create sequences
    sequence_length = 60
    X = []
    y = []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    
    # Reshape X for LSTM input
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    # Split into training and testing sets
    training_size = int(len(X) * 0.8)
    X_train, X_test = X[:training_size], X[training_size:]
    y_train, y_test = y[:training_size], y[training_size:]
    
    # Save preprocessed data
    np.save('data/X_train.npy', X_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_train.npy', y_train)
    np.save('data/y_test.npy', y_test)
    
    print('Data preprocessing completed.')

if __name__ == "__main__":
    ticker = 'AAPL'
    preprocess_data(ticker)

