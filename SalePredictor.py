import torch.nn as nn
import torch

class SalePredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SalePredictor, self).__init__()
        self.shared_block = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
        )
        self.final_block = nn.Sequential(
            nn.Linear(4 + 4 * 14, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)  # output_dim variables
        )

    def forward(self, x):
        base_input = x[:, :4]
        match_inputs = x[:, 4:].view(-1, 14, 8)  # 14 games with 14 features each
        shared_out = [self.shared_block(match_inputs[:, i, :]) for i in range(14)]
        shared_out = torch.cat(shared_out, dim=1)
        combined_input = torch.cat((base_input, shared_out), dim=1)
        output = self.final_block(combined_input)
        return output