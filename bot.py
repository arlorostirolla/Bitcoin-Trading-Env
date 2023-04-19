import torch, optuna, ccxt, os, time, ta
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

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
    
    return model

def objective(trial, data, device):
    # Perform hyperparameter optimization using Optuna
    train_loader, val_loader = data

    # Define hyperparameters to optimize
    d_model = trial.suggest_int("d_model", 64, 256)
    num_heads = trial.suggest_int("num_heads", 1, 8)
    num_blocks = trial.suggest_int("num_blocks", 1, 4)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    num_epochs = trial.suggest_int("num_epochs", 10, 100)

    # Ensure the d_model is divisible by the number of heads
    if d_model % num_heads != 0:
        return np.inf

    # Create the model
    num_inputs = train_loader.dataset.X.shape[-1]
    num_outputs = train_loader.dataset.y.shape[-1]
    model = TemporalFusionTransformer(num_inputs, num_outputs, d_model, num_heads, num_blocks, dropout_rate)

    # Train and evaluate the model
    val_loss = train_tft_model(model, train_loader, val_loader, num_epochs, learning_rate, device)

    return val_loss
