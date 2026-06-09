import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class StockDataLoader:
    def __init__(self, filepath, sequence_length=200):
        self.filepath = filepath
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def load_and_preprocess(self):
        # Read the historical data pipeline
        df = pd.read_csv(self.filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Select target features: Open, High, Low, Close, Volume
        data = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        scaled_data = self.scaler.fit_transform(data)
        return scaled_data

    def create_sequences(self, scaled_data):
        x, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            x.append(scaled_data[i : i + self.sequence_length])
            y.append(scaled_data[i + self.sequence_length][3]) # Target index 3 is Close Price
        return np.array(x), np.array(y)

    def split_data(self, x, y):
        train_size = int(0.80 * len(x))
        val_size = int(0.10 * len(x))
        
        x_train, y_train = x[:train_size], y[:train_size]
        x_val, y_val = x[train_size : train_size + val_size], y[train_size : train_size + val_size]
        x_test, y_test = x[train_size + val_size:], y[train_size + val_size:]
        
        return x_train, y_train, x_val, y_val, x_test, y_test

    def inverse_transform_predictions(self, preds):
        temp = np.zeros((preds.shape[0], 5))
        temp[:, 3] = preds.flatten()
        return self.scaler.inverse_transform(temp)[:, 3]
