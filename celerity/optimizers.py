from typing import List

import torch
from torch import Tensor
from torch.optim import Optimizer


class HedgeOptimizer(Optimizer):
    def __init__(self, model):
        self.model = model
        super().__init__(model.parameters(), defaults={})

    def step(self, X: List[Tensor]) -> None:
        predictions_per_layer = self.model.forward(X)
        losses_per_layer = self.model.loss_per_layer(predictions_per_layer)

        w = [None] * len(losses_per_layer)
        b = [None] * len(losses_per_layer)

        with torch.no_grad():
            for i in range(len(losses_per_layer)):

                losses_per_layer[i].backward(retain_graph=True)
                self.model.lobe.output_layers[i].weight.data -= (
                    self.model.n * self.model.alpha[i] * self.model.lobe.output_layers[i].weight.grad.data
                )
                self.model.lobe.output_layers[i].bias.data -= (
                    self.model.n * self.model.alpha[i] * self.model.lobe.output_layers[i].bias.grad.data
                )

                for j in range(i + 1):
                    if w[j] is None:
                        w[j] = self.model.alpha[i] * self.model.lobe.hidden_layers[j].weight.grad.data
                        b[j] = self.model.alpha[i] * self.model.lobe.hidden_layers[j].bias.grad.data
                    else:
                        w[j] += self.model.alpha[i] * self.model.lobe.hidden_layers[j].weight.grad.data
                        b[j] += self.model.alpha[i] * self.model.lobe.hidden_layers[j].bias.grad.data

                self.model.zero_grad()

            for i in range(len(losses_per_layer)):
                self.model.lobe.hidden_layers[i].weight.data -= self.model.n * w[i]
                self.model.lobe.hidden_layers[i].bias.data -= self.model.n * b[i]

            for i in range(len(losses_per_layer)):
                self.model.alpha[i] *= torch.pow(self.model.b, losses_per_layer[i])
                self.model.alpha[i] = torch.max(self.model.alpha[i], self.model.s / self.model.n_hidden_layers)

        z_t = torch.sum(self.model.alpha)
        self.model.alpha = torch.nn.Parameter(self.model.alpha / z_t, requires_grad=False).to(self.model.device)
