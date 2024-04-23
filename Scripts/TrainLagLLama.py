# Imports
from itertools import islice
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tqdm import tqdm
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.pandas import PandasDataset
import pandas as pd
from estimator import LagLlamaEstimator

LAGLLAMA = "lagllama697.ckpt"
# Function for Lag-Llama inference
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
        validation_data=val_data
    )
    predictor = estimator.train(train, val_data=test, cache_data=True, shuffle_buffer_length=1000)

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
    num_samples = 40
    device = torch.device('cuda')

    train_data = pd.read_csv('./data/train_data_patched_processed.csv')
    val_data = pd.read_csv('./data/val_data_patched_processed.csv')

    # Add the minimum price value to every price value
    train_data['price'] = train_data['price'].apply(lambda x: x+1100.0)
    val_data['price'] = val_data['price'].apply(lambda x: x+1100.0)

    train_data['timestamp'] = pd.to_datetime(train_data['timestamp'], format='%Y-%m-%d %H:%M:%S')
    val_data['timestamp'] = pd.to_datetime(val_data['timestamp'], format='%Y-%m-%d %H:%M:%S')

    train_data = train_data.set_index('timestamp')
    val_data = val_data.set_index('timestamp')

    train_data['item_id'] = 'time_series_1'
    val_data['item_id'] = 'time_series_1'

    print(train_data['price'].dtype)
    print(val_data['price'].dtype)

    train_data['price'] = train_data['price'].astype('float32')
    val_data['price'] = val_data['price'].astype('float32')

    train = PandasDataset.from_long_dataframe(train_data[['price', 'item_id']], item_id='item_id', target='price')
    test = PandasDataset.from_long_dataframe(val_data[['price', 'item_id']], item_id='item_id', target='price')

    ckpt = torch.load(LAGLLAMA, map_location='cuda')
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
    

    estimator = LagLlamaEstimator(
        ckpt_path=LAGLLAMA,
        prediction_length=prediction_length,
        context_length=context_length,
        nonnegative_pred_samples=True,
        aug_prob=0,
        lr=5e-4,
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        time_feat=estimator_args["time_feat"],
        batch_size=128,
        num_parallel_samples=num_samples,
        trainer_kwargs={"max_epochs": 2000},  # lightning trainer arguments
        device=device,
    )
    predictor = estimator.train(train, val_data=test, cache_data=True, shuffle_buffer_length=1000)

    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test,
        predictor=predictor,
        num_samples=num_samples
    )

    forecasts = list(tqdm(forecast_it, total=len(test), desc="Forecasting batches"))
    forecasts = [forecast.samples - 1100.0 for forecast in forecasts]


    tss = list(tqdm(ts_it, total=len(test), desc="Ground truth"))
    tss = [ts.values - 1100.0 for ts in tss]

    # Plot the forecasts
    plt.figure(figsize=(20, 15))
    date_formatter = mdates.DateFormatter('%b, %d')
    plt.rcParams.update({'font.size': 15})

    for idx, (forecast, ts) in islice(enumerate(zip(forecasts, tss)), 9):
        ax = plt.subplot(3, 3, idx+1)
        plt.plot(ts[-4 * prediction_length:].to_timestamp(), label="target")
        forecast.plot(color='g')
        plt.xticks(rotation=60)
        ax.xaxis.set_major_formatter(date_formatter)
        ax.set_title(forecast.item_id)

    plt.gcf().tight_layout()
    plt.legend()
    plt.show()

    # Evaluate the forecasts
    evaluator = Evaluator()
    agg_metrics, ts_metrics = evaluator(iter(tss), iter(forecasts))
    print(agg_metrics)
