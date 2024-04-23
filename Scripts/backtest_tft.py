import torch
import optuna
import ccxt
import os
import time
import ta
import pickle
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import torch.cuda.amp as amp
import torch.nn.utils as nn_utils
from sklearn.preprocessing import StandardScaler
import logging
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import mean_squared_error
from TFTModel import GLU, MultiheadSelfAttention, GatedResidualNetwork, TemporalFusionTransformer
from TrainSingleTFTModel import create_model, preprocess_data, load_and_preprocess_data, BatteryTradingDataset, LogCoshLoss


def predict_price(model, data, input_cols, sequence_length, device):
    model.eval()
    with torch.no_grad():
        X, _ = preprocess_data(data, input_cols, 'price', sequence_length, 1)
        X = torch.tensor(X, dtype=torch.float32).to(device)
        output = model(X)
        predicted_prices = output[0].cpu().numpy()
    return predicted_prices

def backtest(model, data, input_cols, target_col, sequence_length, horizon, device):
    predictions = []
    true_values = []

    for i in range(len(data) - sequence_length - horizon + 1):
        current_data = data.iloc[i:i+sequence_length+horizon]
        predicted_prices = predict_price(model, current_data, input_cols, sequence_length, device)
        true_future_prices = current_data.iloc[sequence_length:][target_col].values

        predictions.append(predicted_prices)
        true_values.append(true_future_prices)

    predictions = np.array(predictions)
    true_values = np.array(true_values)

    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)

    print(f"Backtesting Results:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

def load_model(model_path, device):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    model.to(device)
    return model

if __name__ == "__main__":
    logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    writer = SummaryWriter()

    train_data = pd.read_csv('./data/train_data_patched_processed.csv')
    val_data = pd.read_csv('./data/val_data_patched_processed.csv')

    input_cols = ['price', 'demand', 'temp_air', 'pv_power']
    target_col = 'price'
    sequence_length = 240
    horizon = 10

    train_dataset, val_dataset = load_and_preprocess_data(train_data, val_data, input_cols, target_col, sequence_length, horizon)
    device = torch.device('cpu')

    input_size = len(input_cols)
    output_size = horizon

    val_loss = float('inf')

    params = {
        "d_model": 512,
        "num_heads": 8,
        "num_blocks": 8,
        "dropout_rate": 0.2,
        "learning_rate": 1e-3,
        "num_epochs": 100,
        "batch_size": 64,
        "weight_decay": 0.2
    }

    model = create_model(input_size, output_size, params["d_model"], params["num_heads"], params["num_blocks"], params["dropout_rate"], device)

    # Load the trained model
    model_path = "temp_tft_model.pkl"
    model = load_model(model_path, device)

    # Perform backtesting
    backtest_data = pd.concat([train_data, val_data])  # Combine train and validation data for backtesting
    backtest(model, backtest_data, input_cols, target_col, sequence_length, horizon, device)