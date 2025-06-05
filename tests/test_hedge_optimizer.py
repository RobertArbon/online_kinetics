import pytest
import torch
import numpy as np
from functools import partial
from deeptime.decomposition.deep import vampnet_loss
from celerity.models import HedgeVAMPNetEstimator
from celerity.optimizers import HedgeOptimizer

def create_small_estimator(hedge_eta, hedge_beta, hedge_gamma, n_hidden_layers=2):
    loss_function = partial(vampnet_loss, method='VAMP2', mode='regularize', epsilon=1e-6)
    est = HedgeVAMPNetEstimator(
        input_dim=2,
        output_dim=2,
        n_hidden_layers=n_hidden_layers,
        hidden_layer_width=3,
        device="cpu",
        hedge_eta=hedge_eta,
        hedge_beta=hedge_beta,
        hedge_gamma=hedge_gamma,
        loss_function=loss_function,
        n_epochs=1
    )
    return est

def create_dummy_batch():
    x0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    xtau = torch.tensor([[2.0, 3.0], [4.0, 5.0]], dtype=torch.float32)
    return [x0, xtau]

def compute_weight_change(before, after):
    return torch.norm(after - before).item()

@pytest.mark.parametrize("eta1, eta2", [(0.01, 0.05)])
def test_hedge_eta_direction(eta1, eta2):
    assert eta2 > eta1
    batch = create_dummy_batch()
    
    base_est = create_small_estimator(0.01, 0.99, 0.1)  # Use a default eta
    base_state = {k: v.clone() for k, v in base_est.model.state_dict().items()}
    
    # For eta1
    est1 = create_small_estimator(eta1, 0.99, 0.1)
    est1.model.load_state_dict(base_state)
    est1.model.hedge_eta.data = torch.tensor(eta1)
    initial_weight1 = est1.model.lobe.hidden_layers[0].weight.data.clone()
    est1.optimizer.step(batch)
    change1 = compute_weight_change(initial_weight1, est1.model.lobe.hidden_layers[0].weight.data)
    
    # For eta2
    est2 = create_small_estimator(eta2, 0.99, 0.1)
    est2.model.load_state_dict(base_state)
    est2.model.hedge_eta.data = torch.tensor(eta2)
    initial_weight2 = est2.model.lobe.hidden_layers[0].weight.data.clone()
    est2.optimizer.step(batch)
    change2 = compute_weight_change(initial_weight2, est2.model.lobe.hidden_layers[0].weight.data)
    
    assert change2 > change1, f"Change with larger eta should be larger: {change2} > {change1}"

# Similarly for beta
@pytest.mark.parametrize("beta1, beta2", [(0.99, 0.95)])
def test_hedge_beta_direction(beta1, beta2):
    assert beta2 < beta1  # Smaller beta means more penalization
    batch = create_dummy_batch()
    
    base_est = create_small_estimator(0.01, 0.99, 0.1)
    base_state = {k: v.clone() for k, v in base_est.model.state_dict().items()}
    
    # For beta1
    est1 = create_small_estimator(0.01, beta1, 0.1)
    est1.model.load_state_dict(base_state)
    est1.model.hedge_beta.data = torch.tensor(beta1)
    initial_alpha1 = est1.model.layer_weights.clone()
    est1.optimizer.step(batch)
    change_alpha1 = torch.norm(est1.model.layer_weights - initial_alpha1).item()
    
    # For beta2
    est2 = create_small_estimator(0.01, beta2, 0.1)
    est2.model.load_state_dict(base_state)
    est2.model.hedge_beta.data = torch.tensor(beta2)
    initial_alpha2 = est2.model.layer_weights.clone()
    est2.optimizer.step(batch)
    change_alpha2 = torch.norm(est2.model.layer_weights - initial_alpha2).item()
    
    # Since smaller beta should cause more change in alphas for the same losses
    assert change_alpha2 > change_alpha1

# For gamma, check that alphas don't go below min
@pytest.mark.parametrize("gamma", [0.1, 0.2])
def test_hedge_gamma_min(gamma):
    batch = create_dummy_batch()
    est = create_small_estimator(0.01, 0.5, gamma)  # Use small beta to force decrease
    for _ in range(10):  # Run multiple steps to force alphas down
        est.optimizer.step(batch)
    min_alpha = torch.min(est.model.layer_weights).item()
    expected_min = gamma / est.model.n_hidden_layers
    assert min_alpha >= expected_min - 1e-6
