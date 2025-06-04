import torch
import torch.nn as nn
import torch.nn.functional as F


class Lobe(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_hidden_layers: int,
        hidden_layer_width: int,
        output_softmax: bool = False,
    ):
        super().__init__()
        self.hidden_layers = nn.ModuleList()
        self.output_layers = nn.ModuleList()
        current_dim = input_dim
        for _ in range(n_hidden_layers):
            self.hidden_layers.append(nn.Linear(current_dim, hidden_layer_width))
            self.output_layers.append(nn.Linear(hidden_layer_width, output_dim))
            current_dim = hidden_layer_width
        self.output_softmax = output_softmax

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        predictions = []
        h = x
        for hidden, output in zip(self.hidden_layers, self.output_layers):
            h = F.elu(hidden(h))
            out = output(h)
            if self.output_softmax:
                out = F.softmax(out, dim=1)
            predictions.append(out)
        return predictions
