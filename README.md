# Stacked LSTM with Attention for Stock Price Forecasting

A deep learning pipeline designed to forecast financial asset prices using historical OHLCV data, leveraging stacked Long Short-Term Memory (LSTM) layers paired with a custom Bahdanau-style attention mechanism to capture long-range temporal dependencies.

## 🏗️ System Architecture & Workflow
1. **Data Ingestion:** Loads historical daily OHLCV tabular sequences.
2. **Feature Engineering:** Implements MinMax scaling and rolling sequence windowing (e.g., lookback = 60 days).
3. **Model Pipeline:** 
   - **Layer 1-2:** Stacked Recurrent Layers extract sequential temporal patterns.
   - **Attention Layer:** Computes dynamic alignment scores to focus on high-impact historical trading days.
   - **Dense Output:** Linearly projects context vectors to a continuous scalar price target.

## 📊 Evaluation & Results
The model's forecasting predictive performance was strictly benchmarked against baseline vanilla LSTM architectures:
- **Mean Absolute Error (MAE):** [Insert your value, e.g., 0.024]
- **Mean Squared Error (MSE):** [Insert your value, e.g., 0.0012]
- **Root Mean Squared Error (RMSE):** [Insert your value, e.g., 0.0346]

## 🛠️ How to Run Locally

```bash
# Clone the repository
git clone https://github.com
cd stacked-lstm-stock-forecasting

# Install dependencies
pip install -r requirements.txt

# Execute training pipeline
python src/train.py
```
