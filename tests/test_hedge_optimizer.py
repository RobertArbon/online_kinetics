import torch
from functools import partial
from deeptime.decomposition.deep import vampnet_loss
from celerity.models import HedgeVAMPNetModel
from celerity.optimizers import HedgeOptimizer

class MockEstimator:
    def __init__(self, model):
        self.model = model
        self.loss_function = lambda x0, xt: torch.mean((x0 - xt)**2)

    def _compute_layer_losses(self, layer_predictions):
        x_0_outputs, x_tau_outputs = layer_predictions
        losses = []
        for x0, xt in zip(x_0_outputs, x_tau_outputs):
            loss = self.loss_function(x0, xt)
            losses.append(loss)
        return losses

def create_test_model(hedge_eta, hedge_beta, hedge_gamma, device='cpu'):
    model = HedgeVAMPNetModel(
        input_dim=2,
        output_dim=2,
        n_hidden_layers=2,
        hidden_layer_width=3,
        hedge_eta=hedge_eta,
        hedge_beta=hedge_beta,
        hedge_gamma=hedge_gamma,
        device=device
    )
    # Set deterministic weights
    with torch.no_grad():
        for i in range(2):
            model.lobe.hidden_layers[i].weight.data.fill_(1.0)
            model.lobe.hidden_layers[i].bias.data.fill_(0.0)
            model.lobe.output_layers[i].weight.data.fill_(1.0)
            model.lobe.output_layers[i].bias.data.fill_(0.0)
        model.layer_weights.data = torch.tensor([0.4, 0.6])
    return model

def test_hedge_eta_influence():
    # Small eta
    model1 = create_test_model(hedge_eta=0.01, hedge_beta=0.98, hedge_gamma=0.1)
    initial_weight1 = model1.lobe.hidden_layers[0].weight.data.clone()
    estimator1 = MockEstimator(model1)
    optimizer1 = HedgeOptimizer(estimator1)
    batch = [torch.tensor([[1,2],[3,4],[5,6],[7,8],[9,10]], dtype=torch.float),
             torch.tensor([[2,3],[4,5],[6,7],[8,9],[10,11]], dtype=torch.float)]
    optimizer1.step(batch)
    change1 = torch.norm(model1.lobe.hidden_layers[0].weight.data - initial_weight1).item()

    # Large eta
    model2 = create_test_model(hedge_eta=0.1, hedge_beta=0.98, hedge_gamma=0.1)
    initial_weight2 = model2.lobe.hidden_layers[0].weight.data.clone()
    estimator2 = MockEstimator(model2)
    optimizer2 = HedgeOptimizer(estimator2)
    optimizer2.step(batch)
    change2 = torch.norm(model2.lobe.hidden_layers[0].weight.data - initial_weight2).item()

    assert change2 > change1, f"Larger eta should cause larger change: {change2} vs {change1}"

def test_hedge_beta_influence():
    # Smaller beta (more decay)
    model1 = create_test_model(hedge_eta=0.01, hedge_beta=0.9, hedge_gamma=0.1)
    initial_weights1 = model1.layer_weights.data.clone()
    estimator1 = MockEstimator(model1)
    optimizer1 = HedgeOptimizer(estimator1)
    batch = [torch.tensor([[1,2],[3,4],[5,6],[7,8],[9,10]], dtype=torch.float),
             torch.tensor([[2,3],[4,5],[6,7],[8,9],[10,11]], dtype=torch.float)]
    optimizer1.step(batch)
    final_weights1 = model1.layer_weights.data

    # Larger beta (less decay)
    model2 = create_test_model(hedge_eta=0.01, hedge_beta=0.99, hedge_gamma=0.1)
    initial_weights2 = model2.layer_weights.data.clone()
    estimator2 = MockEstimator(model2)
    optimizer2 = HedgeOptimizer(estimator2)
    optimizer2.step(batch)
    final_weights2 = model2.layer_weights.data

    # Since beta smaller leads to more decay for the same loss, the change in weights should be larger
    change1 = torch.norm(final_weights1 - initial_weights1).item()
    change2 = torch.norm(final_weights2 - initial_weights2).item()
    assert change1 > change2, f"Smaller beta should cause larger change in layer weights: {change1} vs {change2}"

def test_hedge_gamma_influence():
    # Smaller gamma
    model1 = create_test_model(hedge_eta=0.01, hedge_beta=0.98, hedge_gamma=0.01)
    estimator1 = MockEstimator(model1)
    optimizer1 = HedgeOptimizer(estimator1)
    batch = [torch.tensor([[1,2],[3,4],[5,6],[7,8],[9,10]], dtype=torch.float),
             torch.tensor([[2,3],[4,5],[6,7],[8,9],[10,11]], dtype=torch.float)]
    optimizer1.step(batch)
    min_weight1 = torch.min(model1.layer_weights).item()

    # Larger gamma
    model2 = create_test_model(hedge_eta=0.01, hedge_beta=0.98, hedge_gamma=0.1)
    estimator2 = MockEstimator(model2)
    optimizer2 = HedgeOptimizer(estimator2)
    optimizer2.step(batch)
    min_weight2 = torch.min(model2.layer_weights).item()

    assert min_weight2 > min_weight1, f"Larger gamma should result in larger min weight: {min_weight2} vs {min_weight1}"
