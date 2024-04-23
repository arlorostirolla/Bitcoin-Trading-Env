import torch, optuna, ccxt, os, time, ta, pickle
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

class BatteryTradingDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class LogCoshLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(LogCoshLoss, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        loss = torch.log(torch.cosh(y_pred - y_true))
        
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss
        
class GLU(nn.Module):
    def __init__(self, input_size, output_size):
        super(GLU, self).__init__()
        self.fc_linear = nn.Linear(input_size, output_size)
        self.fc_gates = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        linear = self.fc_linear(x)
        gates = self.sigmoid(self.fc_gates(x))
        return linear * gates

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiheadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, "Number of heads must be a factor of the model dimension"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.fc_query = nn.Linear(d_model, d_model)
        self.fc_key = nn.Linear(d_model, d_model)
        self.fc_value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim=-1)

    def split_heads(self, x):
        batch_size, seq_len, _ = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_len, _ = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, x):
        query = self.split_heads(self.fc_query(x))
        key = self.split_heads(self.fc_key(x))
        value = self.split_heads(self.fc_value(x))

        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = self.softmax(scores)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, value)

        context = self.combine_heads(context)
        output = self.fc_out(context)
        return output

class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.1):
        super(GatedResidualNetwork, self).__init__()

        self.fc_input = nn.Linear(input_size, hidden_size)
        self.fc_hidden = nn.Linear(hidden_size, output_size)
        self.fc_gates_input = nn.Linear(input_size, output_size)
        self.fc_gates_hidden = nn.Linear(hidden_size, output_size)

        self.relu = nn.ReLU()
        self.glu = GLU(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(output_size)

    def forward(self, x):
        input_x = x

        x = self.fc_input(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_hidden(x)
        x = self.glu(x)

        gates = self.fc_gates_input(input_x) + self.fc_gates_hidden(x)
        x = input_x + self.dropout(gates * x)

        x = self.layer_norm(x)
        return x

class TemporalFusionTransformer(nn.Module):
    def __init__(self, num_inputs, num_outputs, d_model, num_heads, num_blocks, dropout_rate=0.1):
        super(TemporalFusionTransformer, self).__init__()

        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.d_model = d_model

        self.input_encoding = nn.Linear(num_inputs, d_model)

        self.attention_blocks = nn.ModuleList()
        self.grn_blocks = nn.ModuleList()

        for _ in range(num_blocks):
            self.attention_blocks.append(MultiheadSelfAttention(d_model, num_heads))
            self.grn_blocks.append(GatedResidualNetwork(d_model, d_model, d_model, dropout_rate))

        self.fc_out = nn.Linear(d_model, num_outputs)

    def forward(self, x):
        x = self.input_encoding(x)

        for attn_block, grn_block in zip(self.attention_blocks, self.grn_blocks):
            x_attn = attn_block(x)
            x = grn_block(x + x_attn)

        x = self.fc_out(x)
        return x
    
##############################
#     Model Functions        #
##############################

def create_model(input_size, output_size, d_model, num_heads, num_blocks, dropout_rate, device):
    model = TemporalFusionTransformer(input_size, output_size, d_model, num_heads, num_blocks, dropout_rate)
    model.to(device)
    return model

def predict_price(model, data, input_cols, sequence_length, device):
    model.eval()
    with torch.no_grad():
        X, _ = preprocess_data(data, input_cols, 'price', sequence_length, 1)
        X = torch.tensor(X, dtype=torch.float32).to(device)
        output = model(X)
        predicted_price = output[0].item()
    return predicted_price

###########################
#      Preprocessing      #
###########################

def convert_timestamp(data, timestamp_col):
    if not pd.api.types.is_datetime64_any_dtype(data[timestamp_col]):
        data[timestamp_col] = pd.to_datetime(data[timestamp_col])

    data['day_of_year'] = data[timestamp_col].dt.dayofyear
    data['hour'] = data[timestamp_col].dt.hour
    data['minute'] = data[timestamp_col].dt.minute
    return data

def preprocess_data(data, input_cols, target_col, sequence_length, horizon):
    X, y = [], []
    for i in range(len(data) - sequence_length - horizon + 1):
        X.append(data[i:i+sequence_length][input_cols].values)
        y.append(data[i+sequence_length:i+sequence_length+horizon][target_col].values)
    return np.array(X), np.array(y)

def load_and_preprocess_data(train_data, val_data, input_cols, target_col, sequence_length, horizon):
    train_data = convert_timestamp(train_data, 'timestamp')
    val_data = convert_timestamp(val_data, 'timestamp')

    input_cols += ['day_of_year', 'hour', 'minute']
    
    train_data[input_cols] = StandardScaler().fit_transform(train_data[input_cols])
    val_data[input_cols] = StandardScaler().fit_transform(val_data[input_cols])
    
    train_X, train_y = preprocess_data(train_data, input_cols, target_col, sequence_length, horizon)
    val_X, val_y = preprocess_data(val_data, input_cols, target_col, sequence_length, horizon)
    
    train_dataset = BatteryTradingDataset(train_X, train_y)
    val_dataset = BatteryTradingDataset(val_X, val_y)
    
    return train_dataset, val_dataset

###############################
#        Optimization         #
###############################

def train_tft_model(model, train_loader, val_loader, num_epochs, learning_rate, device, max_norm=1.0, weight_decay=1e-2, trial=None):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = LogCoshLoss(reduction='mean')
    scaler = amp.GradScaler()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        iterations = 0
        print('\n', end='')
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            with amp.autocast():
                outputs = model(batch_x)
                if torch.isnan(outputs).any():
                    print('\r' + "NaN found in model output, skipping iteration {iterations}.", end='')
                    continue
                batch_y = batch_y.view(outputs.size(0), outputs.size(2))
                loss = criterion(outputs[:, -1, :], batch_y)
                if torch.isnan(loss).any():
                    print('\r' + f"NaN found in loss, skipping iteration {iterations}.", end='')
                    continue
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            nn_utils.clip_grad_value_(model.parameters(), clip_value=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            
            print('\r' + f'{iterations} iterations training for Epoch {epoch}', end='')
            iterations += 1
        print('\n', end='')
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        iterations = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                
                print('\r' + f'{iterations} iterations validation for Epoch {epoch}', end='')
                iterations += 1
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                if torch.isnan(outputs).any():
                    print('\r'+f"NaN found in model output, skipping iteration {iterations}.", end='')
                    continue
                batch_y = batch_y.view(outputs.size(0), outputs.size(2))
                loss = criterion(outputs[:, -1, :], batch_y)
                if torch.isnan(loss).any():
                    print('\r'+f"NaN found in loss, skipping iteration {iterations}.", end='')
                    continue
                val_loss += loss.item()
            val_loss /= len(val_loader)
        print('\n', end="")
        print('\r' + f"Epoch {epoch + 1}/{num_epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}", end="")
        
        # Log the training progress
        logging.info(f"Epoch {epoch + 1}/{num_epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        
        if trial:
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return model, val_loss

def objective(trial, train_dataset, val_dataset, input_size, output_size, device, best_val_loss, temp_best_model_path):
    try:
        
        d_model = 256
        num_heads = 16
        
        num_blocks = trial.suggest_int("num_blocks", 1, 10)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.4)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1)  # Updated range
        batch_size = trial.suggest_int("batch_size", 16, 128, step=2)
        num_epochs = trial.suggest_int("num_epochs", 10, 10)
        weight_decay = trial.suggest_float("weight_decay", 0, 0.6)  # Updated range
        
        logging.info(f"Trial parameters: d_model={d_model}, num_heads={num_heads}, num_blocks={num_blocks}, dropout_rate={dropout_rate}, learning_rate={learning_rate}, batch_size={batch_size}, num_epochs={num_epochs}, weight_decay={weight_decay}")
        
        num_workers = 12
        if np.isnan(train_dataset.X).any() or np.isnan(train_dataset.y).any():
            logging.warning("NaN values found in the training dataset.")
        if np.isnan(val_dataset.X).any() or np.isnan(val_dataset.y).any():
            logging.warning("NaN values found in the validation dataset.")
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
        
        model = create_model(input_size, output_size, d_model, num_heads, num_blocks, dropout_rate, device)
        model, val_loss = train_tft_model(model, train_loader, val_loader, num_epochs, learning_rate, device, weight_decay=weight_decay, trial=trial)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            with open(temp_best_model_path, "wb") as file:
                pickle.dump(model, file)
        
        return val_loss
    
    except Exception as e:
        logging.exception(f"Exception occurred during trial: {str(e)}")
        print(str(e))
        return float('inf') 

    except Exception as e:
        logging.exception(f"Exception occurred during trial: {str(e)}")
        print(str(e))
        return float('inf')

def get_possible_num_heads(d_model):
    return [n for n in range(4, (d_model + 1)//3) if d_model % n == 0]

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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define model hyperparameters
    input_size = len(input_cols)
    output_size = horizon

    # Initialize the best validation loss and temporary best model path
    best_val_loss = float('inf')
    temp_best_model_path = "temp_best_model.pkl"

    # Perform hyperparameter optimization using TPE with pruning
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())
    try:
        study.optimize(lambda trial: objective(trial, train_dataset, val_dataset, input_size, output_size, device, best_val_loss, temp_best_model_path), n_trials=100)
    except Exception as e:
        logging.exception(f"Exception occurred during Optuna study: {str(e)}")
        print(f"Exception occurred during Optuna study: {str(e)}")

    # Print best hyperparameters and validation loss
    print("Best hyperparameters:", study.best_params)
    print("Best validation loss:", study.best_value)

    visualization_dir = './visualizations/'
    os.makedirs(visualization_dir, exist_ok=True)
    optuna.visualization.plot_optimization_history(study).write_image(os.path.join(visualization_dir, 'optimization_history.png'))
    optuna.visualization.plot_param_importances(study).write_image(os.path.join(visualization_dir, 'param_importances.png'))

    # Create a fresh model with the best hyperparameters
    best_params = study.best_params
    model = create_model(input_size, output_size, best_params["d_model"], best_params["num_heads"], best_params["num_blocks"], best_params["dropout_rate"], device)

    # Train the fresh model with the best hyperparameters
    num_workers = 12
    train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=best_params["batch_size"], num_workers=num_workers)
    model, _ = train_tft_model(model, train_loader, val_loader, best_params["num_epochs"], best_params["learning_rate"], device, weight_decay=best_params["weight_decay"])

    # Save the final best model
    best_model_path = "best_model.pkl"
    with open(best_model_path, "wb") as file:
        pickle.dump(model, file)
    print(f"Final best model saved as: {best_model_path}")

