import torch, optuna, ccxt, os, time, ta
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


class TimeSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        super(TimeSeriesDataset, self).__init__()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    
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

        # Compute the dot product attention (scaled)
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = self.softmax(scores)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, value)

        # Combine the heads and pass through the output linear layer
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

def train_tft_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    model.to(device)
    # Define the optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss(reduction='mean')


    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for i, (batch_x, batch_y) in enumerate(train_loader):
            # Send batch data to device
            batch_x = batch_x.to(device) 
            batch_y = batch_y.to(device)
            # Forward pass
            outputs = model(batch_x) 
            # Reshape batch_y to match outputs size
            batch_y = batch_y.unsqueeze(1).repeat(1, outputs.size(1), 1) 
            # Compute loss and backward pass
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Accumulate training loss
            train_loss += loss.item() 
            
        # Compute average training loss
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                # Send batch data to device
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                # Forward pass
                outputs = model(batch_x)
                # Reshape batch_y to match outputs size
                batch_y = batch_y.unsqueeze(1).repeat(1, outputs.size(1), 1)
                # Compute loss
                loss = criterion(outputs, batch_y)
                # Accumulate validation loss
                val_loss += loss.item()
                
            # Compute average validation loss
            val_loss /= len(val_loader)
         # Log the metrics to Tensorboard
        


        # Print training and validation loss for current epoch
        print(f"Epoch {epoch+1}/{num_epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
    
    return model, val_loss

def objective(trial, data, device):
    # Perform hyperparameter optimization using Optuna
    train_loader, val_loader = data

    # Define hyperparameters to optimize
    d_model = trial.suggest_int("d_model", 64, 256)
    num_heads = trial.suggest_int("num_heads", 1, 8)
    num_blocks = trial.suggest_int("num_blocks", 1, 4)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    num_epochs = trial.suggest_int("num_epochs", 10, 500)

    # Ensure the d_model is divisible by the number of heads
    if d_model % num_heads != 0:
        raise optuna.TrialPruned()

    # Create the model
    num_inputs = train_loader.dataset.X.shape[-1]
    num_outputs = train_loader.dataset.y.shape[-1]
    model = TemporalFusionTransformer(num_inputs, num_outputs, d_model, num_heads, num_blocks, dropout_rate)

    # Train and evaluate the model
    model, val_loss = train_tft_model(model, train_loader, val_loader, num_epochs, learning_rate, device)
    return val_loss


def fetch_historical_data(symbol, timeframe, output_file):
    exchange = ccxt.binance({
        "rateLimit": 1200, # Adjust the rate limit according to the exchange's requirements
        "enableRateLimit": True,
    })

    all_candles = []
    limit = 1000
    since = exchange.parse8601('2017-01-01T00:00:00Z')
    until = exchange.parse8601('2023-04-19T00:00:00Z')

    while since < until:
        candles = exchange.fetch_ohlcv(symbol, timeframe, limit=limit, since=since)
        if not candles:
            break
        all_candles += candles
        since = candles[-1][0] + exchange.parse_timeframe(timeframe) * 1000
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    
    df.dropna(inplace=True)
    df.to_csv(output_file, index=False)

def create_dataset(data):
    X, y = [], []
    n_steps_in, n_steps_out = 30, 5

    for i in range(len(data)):
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out

        if out_end_ix > len(data):
            break

        seq_X, seq_y = data[i:end_ix, :-1], data[end_ix:out_end_ix, -1]
        X.append(seq_X)
        y.append(seq_y)

    X = np.array(X)
    y = np.array(y)
    print(f"Created {len(X)} input/output sequences")
    return X, y

def load_and_preprocess_data(filename):
    df = pd.read_csv(filename)
    df['timestamp'] = pd.to_datetime(df['timestamp'], format="%Y-%m-%d")
    df.set_index('timestamp', inplace=True)

    # Add technical indicators
    df = ta.add_all_ta_features(df, "open", "high", "low", "close", "volume", fillna=True)

    # Normalize the data
    scaler = MinMaxScaler()
    df[df.columns] = scaler.fit_transform(df[df.columns])

    # Split the data into train and validation sets
    train_size = int(len(df) * 0.7)
    train_df, val_df = df[:train_size], df[train_size:]

    # Convert the data to PyTorch tensors
    train_X, train_y = create_dataset(train_df.values)
    val_X, val_y = create_dataset(val_df.values)
    print(f"X shape: {train_X.shape}, X data type: {train_X.dtype}")
    print(f"y shape: {train_y.shape}, y data type: {train_y.dtype}")

    train_dataset = TimeSeriesDataset(train_X, train_y)
    val_dataset = TimeSeriesDataset(val_X, val_y)

    return train_dataset, val_dataset

def generate_sliding_window(data, input_seq_len, output_seq_len):
    X = []
    y = []
    for i in range(len(data) - input_seq_len - output_seq_len):
        input_seq = data[i:i+input_seq_len]
        if all(input_seq.shape == input_seq[0].shape for input_seq in input_seq):
            X.append(np.array(input_seq))
            output_seq = data[i+input_seq_len:i+input_seq_len+output_seq_len][:,-1]
            y.append(np.array(output_seq))
    X = np.array(X)
    y = np.array(y)
    print("X shape:", X.shape, ", X data type:", X.dtype)
    print("y shape:", y.shape, ", y data type:", y.dtype)
    return X, y

# Add a `trade` function to use the trained model for making trading decisions
def trade(model, data, input_seq_len, output_seq_len, device, threshold):
    model.eval()
    with torch.no_grad():
        X, y = generate_sliding_window(data, input_seq_len, output_seq_len)
        dataset = TimeSeriesDataset(X, y)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        decisions = []

        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            output = model(batch_x)
            pred = output.cpu().numpy().flatten()
            decision = np.sign(pred[-1] - pred[-2])
            if decision > threshold:
                decision = 1  # Buy
            elif decision < -threshold:
                decision = -1  # Sell
            else:
                decision = 0  # Hold

            decisions.append(decision)

    return decisions

if __name__ == "__main__":
    symbol = "BTC/USDT"
    timeframe = "1d"  # Daily candles
    output_file = "btc_historical_data.csv"
    if not os.path.exists(output_file):
        fetch_historical_data(symbol, timeframe, output_file)
    train, val = load_and_preprocess_data(output_file)

    # Generate input and output sequences
    input_seq_len = 30
    output_seq_len = 5
    X_train, y_train = generate_sliding_window(train, input_seq_len, output_seq_len)
    X_val, y_val = generate_sliding_window(val, input_seq_len, output_seq_len)

    # Create DataLoader objects
    train_loader = DataLoader(train, batch_size=30, shuffle=True)
    val_loader = DataLoader(val, batch_size=30, shuffle=False)

    # Set device
    device = torch.device("cuda")
    best_model_path = "./best_model.pt"
    # Perform hyperparameter optimization
    study = optuna.create_study(direction="minimize")
    study.set_user_attr('initial_best_value', float('inf'))
    study.optimize(lambda trial: objective(trial, (train_loader, val_loader), device), n_trials=1000)

    # Print best hyperparameters
    print("Best trial:")
    print("  Value: {}".format(study.best_trial.value))
    print("  Params: ")
    for key, value in study.best_trial.params.items():
        print("    {}: {}".format(key, value))
