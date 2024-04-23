# Imports
from itertools import islice

from tqdm import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.pandas import PandasDataset
import pandas as pd
from estimator import LagLlamaEstimator
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import plotly.io as pio
from gluonts.dataset.common import ListDataset

LAGLLAMA = "epoch183.ckpt"

def get_lag_llama_predictions(dataset, val_data, prediction_length, context_length=32, num_samples=20, device="cuda", batch_size=64, nonnegative_pred_samples=True):
    ckpt = torch.load(LAGLLAMA, map_location=device)
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    estimator = LagLlamaEstimator(
        ckpt_path=LAGLLAMA,
        prediction_length=prediction_length,
        context_length=context_length,
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        nonnegative_pred_samples=nonnegative_pred_samples,
        rope_scaling={
            "type": "linear",
            "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
        },
        batch_size=batch_size,
        num_parallel_samples=num_samples,
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=num_samples
    )
    forecasts = list(tqdm(forecast_it, total=len(dataset), desc="Forecasting batches"))
    tss = list(tqdm(ts_it, total=len(dataset), desc="Ground truth"))

    return forecasts, tss

if __name__ == '__main__':
    prediction_length = 50
    context_length = 240
    num_samples = 10
    device = torch.device('cpu')
    val_data = pd.read_csv('./data/val_data_patched_processed.csv')
    val_data['timestamp'] = pd.to_datetime(val_data['timestamp'], format='%Y-%m-%d %H:%M:%S')
    val_data = val_data.set_index('timestamp')
    val_data['item_id'] = 'time_series_1'
    val_data['price'] = val_data['price'].astype('float32')

    # Create a rolling window dataset
    dataset = []
    for i in range(len(val_data) - context_length - prediction_length + 1):
        dataset.append(ListDataset([{
            "start": val_data.index[i], 
            "target": val_data.iloc[i:i+context_length+prediction_length]['price'].tolist(),
            "item_id": "time_series_1"
        }], freq=val_data.index.freq))

    all_forecasts = []
    all_forecast_dates = pd.DatetimeIndex([])

    fig = go.Figure()

    # Iterate through the rolling window and make predictions
    for i, test_slice in enumerate(dataset):
        forecasts, tss = get_lag_llama_predictions(test_slice, val_data, prediction_length, context_length, num_samples, device)
        forecast_samples = forecasts[0].samples
        forecast_means = np.mean(forecast_samples, axis=0)
        forecast_medians = np.median(forecast_samples, axis=0)
        forecast_dates = pd.date_range(start=test_slice.index[context_length+i], periods=prediction_length, freq=val_data.index.freq)

        # Append the forecasts and dates
        all_forecasts.append(forecast_samples)
        all_forecast_dates = all_forecast_dates.append(forecast_dates)

        # Plot the probability distributions as shaded areas
        for j in range(num_samples):
            fig.add_trace(go.Scatter(
                x=forecast_dates,
                y=forecast_samples[j],
                mode='lines',
                line=dict(color='rgba(68, 68, 68, 0.1)'),
                showlegend=False
            ))

    # Compute the overall mean and median forecasts
    all_forecasts = np.concatenate(all_forecasts, axis=1)
    mean_forecasts = np.mean(all_forecasts, axis=0)
    median_forecasts = np.median(all_forecasts, axis=0)

    # Plot the actuals
    fig.add_trace(go.Scatter(
        x=val_data.index,
        y=val_data['price'],
        mode='lines',
        name='Actual',
        line=dict(color='blue')
    ))

    # Overlay the mean and median forecasts
    fig.add_trace(go.Scatter(
        x=all_forecast_dates,
        y=mean_forecasts,
        mode='lines',
        name='Forecast Mean',
        line=dict(color='orange')
    ))
    fig.add_trace(go.Scatter(
        x=all_forecast_dates,
        y=median_forecasts,
        mode='lines',
        name='Forecast Median',
        line=dict(color='green')
    ))

    # Set up the layout for higher resolution
    fig.update_layout(
        title='Forecast Distributions vs Actuals',
        xaxis_title='Timestamp',
        yaxis_title='Price',
        autosize=False,
        width=1920,  # Higher width for more detail
        height=1080,  # Higher height for more detail
        font=dict(size=18)  # Larger font for readability
    )

    # Show the figure
    fig.show()

    # Save the figure with high resolution
    fig.write_image("forecast_distributions_vs_actuals.png", scale=3)  # Increase the scale for higher resolution