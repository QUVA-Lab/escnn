
import torch
import math
import numpy as np

from torch.optim import Adam, SGD


def _normalize(X: torch.Tensor) -> torch.Tensor:
    X /= torch.norm(X, dim=1, keepdim=True)


def _distance_sphere(X: torch.Tensor, Y: torch.Tensor = None, type: str = 'euclidean') -> torch.Tensor:

    if Y is None:
        Y = X

    with torch.no_grad():
        assert torch.isclose(torch.norm(X, dim=1), torch.tensor(1., dtype=X.dtype)).all()
        assert torch.isclose(torch.norm(Y, dim=1), torch.tensor(1., dtype=Y.dtype)).all()

    if type == 'cos':

        cos = X @ Y.T

        dist = torch.zeros_like(cos)

        ATOL = 1e-15
        grad_mask_0 = torch.isclose(cos, torch.tensor(1., dtype=cos.dtype), atol=ATOL)
        grad_mask_1 = torch.isclose(cos, torch.tensor(-1., dtype=cos.dtype), atol=ATOL)
        grad_mask = grad_mask_0 | grad_mask_1

        dist[~grad_mask] = torch.acos(cos[~grad_mask]) / math.pi
        dist[grad_mask_0] = 0.
        dist[grad_mask_1] = 1.

        return dist

    elif type == 'euclidean':
        dist = torch.norm(X.unsqueeze(1) - Y, dim=2, keepdim=False)
        assert dist.shape == (X.shape[0], Y.shape[0])
        return dist
    else:
        raise ValueError()


def _potential_sphere(X: torch.Tensor, distance: str = 'euclidean') -> torch.Tensor:

    dist = _distance_sphere(X, type=distance)

    mask = torch.triu(torch.ones(X.shape[0], X.shape[0], dtype=torch.bool), diagonal=1)

    dist = dist[mask]

    potential = 1. / dist
    return potential.sum()


def _thomson_sphere(N: int, lr: float = 1e-1, optimizer: str = 'Adam', rng: np.random.RandomState = None, verbose: int = 0) -> torch.Tensor:

    if rng is None:
        rng = np.random

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    X = torch.tensor(
        rng.randn(N, 3).astype(np.float32),
        requires_grad=True, device=device, dtype=torch.float32
    )

    if optimizer == 'Adam':
        optim = Adam(
            params=(X,),
            lr=lr,
            weight_decay=0.
        )
    elif optimizer == 'SGD':
        optim = SGD(
            params=(X,),
            lr=lr,
            momentum=.2,
            weight_decay=0.
        )
    else:
        raise ValueError()

    last_U = torch.tensor(N**2, dtype=X.dtype, device=device)
    best_U = last_U
    best_X = None

    for i in range(5000):

        with torch.no_grad():
            _normalize(X)

        optim.zero_grad()

        U = _potential_sphere(X, 'euclidean')

        if verbose > 1:
            print(f'{i}: {U.item()}')

        if torch.isclose(U, last_U, atol=1e-16, rtol=0.):
            break

        last_U = U.detach()

        if last_U < best_U:
            best_U = last_U
            best_X = X.detach().cpu()

        U.backward()
        optim.step()

    X = best_X
    _normalize(X)

    if verbose > 0:
        # Uc = _potential_sphere(X, 'cos')
        Ue = _potential_sphere(X, 'euclidean')

        # print(f'Geodesic Distance: {Uc.item()}')
        print(f'Euclidean Distance: {Ue.item()}')

    return X.detach().cpu()


def _distance_so3(X: torch.Tensor, Y: torch.Tensor = None, type: str = 'euclidean') -> torch.Tensor:

    if Y is None:
        Y = X

    d1 = _distance_sphere(X, Y, type=type)
    d2 = _distance_sphere(X, -Y, type=type)

    dist = torch.min(d1, d2)

    return dist


def _potential_so3(X: torch.Tensor, distance: str = 'euclidean') -> torch.Tensor:

    dist = _distance_so3(X, type=distance)

    mask = torch.triu(torch.ones(X.shape[0], X.shape[0], dtype=torch.bool), diagonal=1)

    dist = dist[mask]

    potential = 1. / dist
    return potential.sum()


def _thomson_so3(N: int, lr: float = 1e-1, optimizer: str = 'Adam', rng: np.random.RandomState = None, verbose: int = 0) -> torch.Tensor:

    if rng is None:
        rng = np.random

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    X = torch.tensor(
        rng.randn(N, 4).astype(np.float32),
        requires_grad=True, device=device, dtype=torch.float32
    )

    if optimizer == 'Adam':
        optim = Adam(
            params=(X,),
            lr=lr,
            weight_decay=0.
        )
    elif optimizer == 'SGD':
        optim = SGD(
            params=(X,),
            lr=lr,
            momentum=.2,
            weight_decay=0.
        )
    else:
        raise ValueError()

    last_U = torch.tensor(N**2, dtype=X.dtype, device=device)
    best_U = last_U
    best_X = None

    for i in range(5000):

        with torch.no_grad():
            _normalize(X)

        optim.zero_grad()

        U = _potential_so3(X, 'euclidean')

        if verbose > 1:
            print(f'{i}: {U.item()}')

        if torch.isclose(U, last_U, atol=1e-16, rtol=0.):
            break

        last_U = U.detach()

        if last_U < best_U:
            best_U = last_U
            best_X = X.detach().cpu()

        U.backward()
        optim.step()

    X = best_X
    _normalize(X)

    if verbose > 0:

        # Uc = _potential_so3(X, 'cos')
        Ue = _potential_so3(X, 'euclidean')

        # print(f'Geodesic Distance: {Uc.item()}')
        print(f'Euclidean Distance: {Ue.item()}')

    return X.detach().cpu()


from scipy.spatial.transform import Rotation


def _thomson_cube_sphere(N: int, lr: float = 1e-1, optimizer: str = 'Adam', rng: np.random.RandomState = None, verbose: int = 0) -> torch.Tensor:

    if rng is None:
        rng = np.random

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    X = torch.tensor(
        rng.randn(N, 3).astype(np.float32),
        requires_grad=True, device=device, dtype=torch.float32
    )
    cube = torch.tensor(
        Rotation.create_group('O').as_matrix(),
        device=X.device,
        dtype=X.dtype
    )
    assert cube.shape == (24, 3, 3)

    def _cubify(_X: torch.Tensor) -> torch.Tensor:
        return torch.einsum('gij,sj->gsi', cube.to(_X.device), _X).reshape(-1, 3)

    if optimizer == 'Adam':
        optim = Adam(
            params=(X,),
            lr=lr,
            weight_decay=0.
        )
    elif optimizer == 'SGD':
        optim = SGD(
            params=(X,),
            lr=lr,
            momentum=.2,
            weight_decay=0.
        )
    else:
        raise ValueError()

    last_U = torch.tensor((24*N)**2, dtype=X.dtype, device=device)
    best_U = last_U
    best_X = None

    for i in range(5000):

        with torch.no_grad():
            _normalize(X)

        optim.zero_grad()

        U = _potential_sphere(_cubify(X), 'euclidean')

        if verbose > 1:
            print(f'{i}: {U.item()}')

        if torch.isclose(U, last_U, atol=1e-16, rtol=0.):
            break

        last_U = U.detach()

        if last_U < best_U:
            best_U = last_U
            best_X = X.detach().cpu()

        U.backward()
        optim.step()

    X = best_X
    _normalize(X)

    if verbose > 0:
        # Uc = _potential_sphere(_cubify(X), 'cos')
        Ue = _potential_sphere(_cubify(X), 'euclidean')

        # print(f'Geodesic Distance: {Uc.item()}')
        print(f'Euclidean Distance: {Ue.item()}')

    return _cubify(X).detach().cpu()


def _thomson_cube_so3(N: int, lr: float = 1e-1, optimizer: str = 'Adam', rng: np.random.RandomState = None, verbose: int = 0) -> torch.Tensor:

    if rng is None:
        rng = np.random

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    X = torch.tensor(
        rng.randn(N, 4).astype(np.float32),
        requires_grad=True, device=device, dtype=torch.float32
    )

    cube = torch.tensor(
        Rotation.create_group('O').as_quat(),
        device=X.device,
        dtype=X.dtype
    )
    assert cube.shape == (24, 4)

    quaternion_basis = torch.stack([
        torch.tensor([
            [0, 0, 0, 1],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [-1, 0, 0, 0],
        ], dtype=X.dtype),
        torch.tensor([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
        ], dtype=X.dtype),
        torch.tensor([
            [0, -1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, -1, 0],
        ], dtype=X.dtype),
        torch.eye(4, dtype=X.dtype),
    ], dim=0).to(device=X.device)

    def _cubify(_X: torch.Tensor) -> torch.Tensor:
        return torch.einsum('bij,gb,sj->gsi', quaternion_basis.to(_X.device), cube.to(_X.device), _X).reshape(-1, 4)

    if optimizer == 'Adam':
        optim = Adam(
            params=(X,),
            lr=lr,
            weight_decay=0.
        )
    elif optimizer == 'SGD':
        optim = SGD(
            params=(X,),
            lr=lr,
            momentum=.2,
            weight_decay=0.
        )
    else:
        raise ValueError()

    last_U = torch.tensor((24*N)**2, dtype=X.dtype, device=device)
    best_U = last_U
    best_X = None

    for i in range(5000):

        with torch.no_grad():
            _normalize(X)

        optim.zero_grad()

        U = _potential_so3(_cubify(X), 'euclidean')

        if verbose > 1:
            print(f'{i}: {U.item()}')

        if torch.isclose(U, last_U, atol=1e-16, rtol=0.):
            break

        last_U = U.detach()

        if last_U < best_U:
            best_U = last_U
            best_X = X.detach().cpu()

        U.backward()
        optim.step()

    X = best_X
    _normalize(X)

    if verbose > 0:

        # Uc = _potential_so3(_cubify(X), 'cos')
        Ue = _potential_so3(_cubify(X), 'euclidean')

        # print(f'Geodesic Distance: {Uc.item()}')
        print(f'Euclidean Distance: {Ue.item()}')

    return _cubify(X).detach().cpu()

########################################################################################################################

from joblib import Memory
# import os
# cache_sphere = Memory(os.path.join(os.path.dirname(__file__), '_jl_thomson_sphere'), verbose=2)
# cache_so3 = Memory(os.path.join(os.path.dirname(__file__), '_jl_thomson_so3'), verbose=2)
# cache_cube_sphere = Memory(os.path.join(os.path.dirname(__file__), '_jl_thomson_cube_sphere'), verbose=2)
# cache_cube_so3 = Memory(os.path.join(os.path.dirname(__file__), '_jl_thomson_cube_so3'), verbose=2)

from escnn.group import __cache_path__
cache = Memory(__cache_path__, verbose=0)


# @cache_sphere.cache
@cache.cache
def thomson_sphere(N: int) -> np.ndarray:
    attempts = 20
    verbose = 0

    # to ensure a determinist behaviour of this method
    rng = np.random.RandomState(42)

    best_U = N ** 2
    best_X = None
    for i in range(attempts):
        X = _thomson_sphere(N, lr=1e-1, optimizer='Adam', rng=rng, verbose=verbose - 1)
        U = _potential_sphere(X).item()

        if U < best_U:
            best_U = U
            best_X = X

    if verbose > 0:
        print(f'Best Potential: {best_U}')

    return best_X.numpy()


# @cache_so3.cache
@cache.cache
def thomson_so3(N: int) -> np.ndarray:

    attempts = 10
    verbose = 0

    # to ensure a determinist behaviour of this method
    rng = np.random.RandomState(42)

    best_U = N**2
    best_X = None
    for i in range(attempts):
        X = _thomson_so3(N, lr=5e-2, optimizer='Adam', rng=rng, verbose=verbose - 1)
        U = _potential_so3(X).item()

        if U < best_U:
            best_U = U
            best_X = X

    if verbose > 0:
        print(f'Best Potential: {best_U}')

    return best_X.numpy()


# @cache_cube_sphere.cache
@cache.cache
def thomson_cube_sphere(N: int) -> np.ndarray:
    attempts = 20
    verbose = 0

    # to ensure a determinist behaviour of this method
    rng = np.random.RandomState(42)

    best_U = (24*N) ** 2
    best_X = None
    for i in range(attempts):
        X = _thomson_cube_sphere(N, lr=1e-1, optimizer='Adam', rng=rng, verbose=verbose - 1)
        U = _potential_sphere(X).item()

        if U < best_U:
            best_U = U
            best_X = X

    if verbose > 0:
        print(f'Best Potential: {best_U}')

    return best_X.numpy()


# @cache_cube_so3.cache
@cache.cache
def thomson_cube_so3(N: int) -> np.ndarray:

    attempts = 10
    verbose = 0

    # to ensure a determinist behaviour of this method
    rng = np.random.RandomState(42)

    best_U = (24*N)**2
    best_X = None
    for i in range(attempts):
        X = _thomson_cube_so3(N, lr=5e-2, optimizer='Adam', rng=rng, verbose=verbose - 1)
        U = _potential_so3(X).item()

        if U < best_U:
            best_U = U
            best_X = X

    if verbose > 0:
        print(f'Best Potential: {best_U}')

    return best_X.numpy()


if __name__ == '__main__':

    # X = thomson_sphere(120)
    # X = thomson_so3(130)
    # X = _thomson_cube_so3(3, verbose=3)
    X = _thomson_cube_sphere(1, verbose=3)
    # X = _thomson_so3(72, verbose=3)
