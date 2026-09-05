import torch
from pinn.pdes import Poisson1D

def test_exact_solution_default_boundaries():
    p=Poisson1D(); x=torch.tensor([[-1.0],[1.0]],requires_grad=True); u=p.exact(x)
    assert torch.allclose(u, torch.zeros_like(u))

def test_residual_shape():
    from pinn.model import MLP
    p=Poisson1D(); m=MLP(1,1,[8,8]); x=torch.rand(5,1)
    assert p.residual(m,x).shape == (5,1)
