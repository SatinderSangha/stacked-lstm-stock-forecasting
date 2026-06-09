import os
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_loader import StockDataLoader
from model import StockForecastingModels

def calculate_metrics(actual, pred):
    mse = mean_squared_error(actual, pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, pred)
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    accuracy = 100 - mape
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "Accuracy": accuracy}

if __name__ == "__main__":
    # Ensure data path exists
    data_path = 'fang.us.txt'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Please upload the tracking asset file '{data_path}' to root directory.")

    # Ingest and package sequences 
    loader = StockDataLoader(filepath=data_path, sequence_length=200)
    scaled_data = loader.load_and_preprocess()
    x, y = loader.create_sequences(scaled_data)
    x_train, y_train, x_val, y_val, x_test, y_test = loader.split_data(x, y)

    input_shape = (x_train.shape[1], x_train.shape[2])

    # 1. Evaluate Attention LSTM
    print("\n--- Training Attention Augmented LSTM Pipeline ---")
    att_model = StockForecastingModels.build_attention_lstm(input_shape)
    att_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    att_model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_val, y_val), verbose=1)
    
    att_preds = att_model.predict(x_test)
    att_preds_rescaled = loader.inverse_transform_predictions(att_preds)

    # 2. Evaluate Baseline Vanilla LSTM
    print("\n--- Training Baseline Vanilla LSTM Pipeline ---")
    vanilla_model = StockForecastingModels.build_vanilla_lstm(input_shape)
    vanilla_model.compile(optimizer='adam', loss='mean_squared_error')
    vanilla_model.fit(x_train, y_train, batch_size=32, epochs=20, validation_data=(x_val, y_val), verbose=1)
    
    vanilla_preds = vanilla_model.predict(x_test)
    vanilla_preds_rescaled = loader.inverse_transform_predictions(vanilla_preds)

    # Extract actual rescaled baseline prices
    actual_prices = loader.inverse_transform_predictions(y_test.reshape(-1, 1))

    # Evaluate performance criteria benchmarks
    att_metrics = calculate_metrics(actual_prices, att_preds_rescaled)
    vanilla_metrics = calculate_metrics(actual_prices, vanilla_preds_rescaled)

    print("\n================ FINAL ARCHITECTURE PERFORMANCE EVALUATION ================")
    print(f"Attention LSTM System -> MAE: {att_metrics['MAE']:.4f} | RMSE: {att_metrics['RMSE']:.4f} | Accuracy: {att_metrics['Accuracy']:.2f}%")
    print(f"Vanilla LSTM Baseline -> MAE: {vanilla_metrics['MAE']:.4f} | RMSE: {vanilla_metrics['RMSE']:.4f} | Accuracy: {vanilla_metrics['Accuracy']:.2f}%")
