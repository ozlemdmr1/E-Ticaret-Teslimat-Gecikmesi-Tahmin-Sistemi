import torch
import torch.nn as nn

class ShipDelayNN(nn.Module):
    def __init__(self, input_dim):
        super(ShipDelayNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)  # BCEWithLogitsLoss kullanılacağı için çıkışta Sigmoid yok
        )
        
    def forward(self, x):
        return self.network(x)