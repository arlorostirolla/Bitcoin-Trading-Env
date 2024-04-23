import torch 
import torch.nn as nn

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
