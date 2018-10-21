import pytest
import numpy as np
import torch
import cvx_ha
from scipy.optimize import minimize


@pytest.fixture('module')
def A():
    torch.manual_seed(42)
    A = torch.randn(40, 10)
    return A


@pytest.fixture
def x0():
    x0 = torch.ones(10, requires_grad=True)
    x0.data.div_(10)
    return x0


def f(x, A):
    return torch.relu(1 - A @ x).mean()


@pytest.fixture
def x_star(A, x0):
    A = A.numpy()
    x0 = x0.data.numpy()

    def np_f(x):
        return np.clip(1 - A @ x, 0, np.inf).mean()

    solution = minimize(np_f, x0=x0,
                        constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x)-1}],
                        bounds=[(0, 1)]*10)
    return torch.tensor(solution.x).float()


@pytest.fixture
def y_star(x_star, A):
    return f(x_star, A).item()


def test_sgd(A, x0, x_star, y_star):
    def closure():
        y = f(x0, A)
        y.backward()
        return y.item()

    opt = cvx_ha.ha1.ProjectedSGD([x0], lr=1/(10 * 1600)**.5, projection=cvx_ha.ha1.simplex_projection)
    for i in range(1600):
        opt.zero_grad()
        y = opt.step(closure)
    assert torch.allclose(x_star, x0, atol=1e-3)
    assert np.allclose(y, y_star, atol=1e-3)


def test_dual_averaging(A, x0, x_star, y_star):
    def closure():
        y = f(x0, A)
        y.backward()
        return y.item()

    opt = cvx_ha.ha1.DualAveraging([x0], lr=1/1600**.5, dual_projection=cvx_ha.ha1.simplex_dual_projection)
    for i in range(1600):
        opt.zero_grad()
        y = opt.step(closure)
    assert torch.allclose(x_star, x0, atol=1e-3)
    assert np.allclose(y, y_star, atol=1e-3)


def test_mirror_descent_1(A, x0, x_star, y_star):
    def closure():
        y = f(x0, A)
        y.backward()
        return y.item()
    Dh = cvx_ha.ha1.mirror_descent.BregmanProjectionUpdate(
        h=lambda x: np.nan_to_num(np.log(x, where=x != 0) * x).sum(),
        dh=lambda x: 1. + np.log(x+1e-12),
        opt_kwargs=dict(
            constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}],
            bounds=[(0, 1)] * len(x0)
        ),
        x0=np.ones(len(x0), dtype='float32') / len(x0),
        torch_compatible=False,

    )
    opt = cvx_ha.ha1.MirrorDescent([x0], lr=1/1600**.3, brehman_projection=Dh)
    for i in range(1600):
        opt.zero_grad()
        y = opt.step(closure)
    assert torch.allclose(x_star, x0, atol=1e-3)
    assert np.allclose(y, y_star, atol=1e-3)


def test_mirror_descent_2(A, x0, x_star, y_star):
    def closure():
        y = f(x0, A)
        y.backward()
        return y.item()
    opt = cvx_ha.ha1.MirrorDescent([x0], lr=1/1600**.3,
                                   brehman_projection=cvx_ha.ha1.mirror_descent.brehman_exp_simplex_projection)
    for i in range(1600):
        opt.zero_grad()
        y = opt.step(closure)
    assert torch.allclose(x_star, x0, atol=1e-3)
    assert np.allclose(y, y_star, atol=1e-3)
