import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.hidden_size = 32
        self.num_layers = 2
        self.num_heads = 1
        
        self.embedding = nn.Linear(1, self.hidden_size)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=self.num_layers)
        

        self.fc = nn.Linear(self.hidden_size, 100315)

    def forward(self, x):

        x = x.squeeze(1).unsqueeze(-1)
        

        x = self.embedding(x)
        

        out = self.transformer_encoder(x)
        

        out = out[:, -1, :]
        

        out = self.fc(out)
        return out

