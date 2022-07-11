import unittest
from unittest import TestCase

from escnn.nn import *
from escnn.group import *
from escnn.gspaces import *

import numpy as np

import torch

from typing import List, Iterable, Callable
from collections import defaultdict


def compute_energies(F, rho: Representation):
    E = (F ** 2).mean(axis=1)
    energies = defaultdict(float)
    
    p = 0
    for irr in rho.irreps:
        irr = rho.group.irrep(*irr)
        
        energies[irr.attributes['frequency']] += E[p:p+irr.size].sum()
        
        p += irr.size
    return dict(energies)


def _so3_sensing_sampling(m):
    G = SO3(1)
    for p in range(m):
        cos_theta = (2 * p - m + 1) / (m - 1)
        theta = np.arccos(cos_theta)
        phi = 2 * np.pi * np.random.rand()
        psi = 2 * np.pi * np.random.rand()
        
        x = G.element(
            np.asarray([phi, theta, psi]),
            'ZYZ'
        )
        
        yield x


def _so3_random_sampling(n):
    G = SO3(1)
    for _ in range(n):
        theta = np.pi * np.random.rand()
        phi = 2 * np.pi * np.random.rand()
        psi = 2 * np.pi * np.random.rand()

        x = G.element(
            np.asarray([phi, theta, psi]),
            'ZYZ'
        )

        yield x


def _uniform_sampling(n, G: Group):
    for _ in range(n):
        yield G.sample()


def _regular_sampling(n, G: Group):
    for x in G.testing_elements(n):
        yield x


def _so3_platonic_sampling(solid):
    G = SO3(1)
    for x in G.grid(solid):
        yield x


def _so3_hopf_sampling(N):
    G = SO3(1)
    for x in G.grid('hopf', N=N):
        yield x


def _so3_thomson_cube_sampling(n):
    G = SO3(1)
    thomson = G.grid('thomson', n)
    cube = G.grid('octa')

    for c in cube:
        for t in thomson:
            yield c @ t


def activation(activation: str):
    if activation == 'identity':
        activation = lambda x: x
    elif activation == 'relu':
        activation = lambda x: np.maximum(x, 0.)
    elif activation == 'sin':
        activation = lambda x: np.sin(x)
    elif activation == 'elu':
        activation = lambda x: torch.nn.functional.elu(torch.tensor(x)).detach().numpy() #+ 1.
    else:
        raise ValueError
    return activation


def preconditioning_so3(preconditioning: str):
    
    if preconditioning == 'sin':
        return lambda sample: np.sqrt(np.sin(sample.to('ZYZ')[1]))
    elif preconditioning == 'tan':
        return lambda sample: np.abs(np.cbrt(np.tan(sample.to('ZYZ')[1])))
    elif preconditioning == 'uniform':
        return lambda sample: 1.
    else:
        raise ValueError()


def kernel_so3(L: int, kernel: str, **kwargs):
    d = 0
    for l in range(L+1):
        d += (2*l+1)**2
    
    V = np.zeros((d, 1))
    
    if kernel == 'dirac':
        p = 0
        for l in range(L + 1):
            d = (2 * l + 1)
            V[p:p+d**2, 0] = np.eye(d).reshape(-1) * d
            p += d ** 2
    elif kernel == 'P':
        assert 'r' in kwargs
        r = kwargs['r']
        
        p = 0
        for l in range(L + 1):
            d = 2 * l + 1
            f = (1 - r) ** 2 * r ** l / (1 + r) / d
            V[p:p + d ** 2, 0] = np.eye(d).reshape(-1) * f * d
            p += d ** 2
    else:
        raise ValueError
        
    V /= np.linalg.norm(V.reshape(-1))
    return V


def check_reconstruction(
        rho: Representation,
        kernel: np.ndarray,
        sampler: Iterable[GroupElement],
        preconditioning: Callable[[GroupElement], float],
        activation: Callable[[float], float],
        regularization: float = 0.,
        samples: int = 200,
):

    G = rho.group
    assert kernel.shape == (rho.size, 1)

    def reconstruct(x, w, y, l=0.):
        
        A = w * x
        # A = x
        A_inv = np.linalg.inv(A.T @ A + l * np.eye(A.shape[1])) @ A.T
        # A_inv = A.T
        # A_inv = A_inv * w.T

        return A_inv @ y

    def build_sensing_matrix(samples: Iterable[GroupElement], preconditioning):
    
        X = []
        W = []
        for x in samples:
            X.append(kernel.T @ rho(x))
            W.append(preconditioning(x))
        N = len(X)
    
        X = np.concatenate(X, axis=0) / np.sqrt(N)
        assert X.shape == (N, rho.size)
    
        W = np.array(W).reshape(N, 1)
    
        return W, X

    features = np.random.randn(rho.size, samples)
    features /= np.linalg.norm(features, axis=0, keepdims=True)
    features *= 3.

    energies = sorted(list(compute_energies(features, rho).items()))
    energies = np.asarray([e for l, e in energies])
    # print("Initial Energies: ", energies)

    def evaluate_samples(samples, preconditioning, activation, l=0.):
    
        W, X = build_sensing_matrix(samples, preconditioning)
    
        N = W.shape[0]
    
        Y = activation(X @ features)
        try:
            F = reconstruct(X, W, Y, l=l)
        except np.linalg.LinAlgError:
            print('error reconstruction')
            return None, None

        # energies = sorted(list(compute_energies(F, rho).items()))
        # energies = np.asarray([e for l, e in energies])
        # print("Energies: ", energies)
        
        errors = []
    
        # for g in G.testing_elements(3):
        for _ in range(400):
            g = G.sample()
            Yg = activation(X @ rho(g) @ features)
        
            try:
                Fg = reconstruct(X, W, Yg, l=l)
            except np.linalg.LinAlgError:
                errors.append(float('inf'))
                print('skip element')
                continue
        
            gF = rho(g) @ F
        
            error = np.linalg.norm(gF - Fg, axis=-1)
            error /= np.linalg.norm(gF, axis=-1)
            errors.append(error)
        errors = np.concatenate(errors)
        assert len(errors.shape) == 1
        return N, errors

    np.set_printoptions(precision=3, suppress=True, linewidth=10000)
    N, error = evaluate_samples(sampler, preconditioning, activation, l=regularization)

    if N is not None:
        return N, error
    else:
        return None, None


def test_so2_elu():

    G = SO2(3)

    L = 5

    irreps = directsum(
        [G.irrep(l) for l in range(L + 1)]
    )

    kernel = np.zeros((irreps.size, 1))
    kernel[0] = 1.
    kernel[1::2] = 1.

    for n in range(irreps.size, 2 * irreps.size + 1, max(1, irreps.size // 10)):
        check_reconstruction(
            irreps,
            kernel,
            # sampler=_uniform_sampling(n, G),
            sampler=_regular_sampling(n, G),
            preconditioning=preconditioning_so3('uniform'),
            activation=activation('elu'),
            regularization=0.
        )


def test_so2_relu():
    G = SO2(3)
    
    L = 3
    
    irreps = directsum(
        [G.irrep(l) for l in range(L + 1)]
    )
    
    kernel = np.zeros((irreps.size, 1))
    kernel[0] = 1.
    kernel[1::2] = 1.
    
    for n in range(irreps.size, 2 * irreps.size + 1, max(1, irreps.size // 10)):
        check_reconstruction(
            irreps,
            kernel,
            # sampler=_uniform_sampling(n, G),
            sampler=_regular_sampling(n, G),
            preconditioning=preconditioning_so3('uniform'),
            activation=activation('relu'),
            regularization=0.
        )


def test_so2_identity():
    G = SO2(3)
    
    L = 3
    
    irreps = directsum(
        [G.irrep(l) for l in range(L + 1)]
    )
    
    kernel = np.zeros((irreps.size, 1))
    kernel[0] = 1.
    kernel[1::2] = 1.
    
    for n in range(irreps.size, 2 * irreps.size + 1, max(1, irreps.size // 10)):
        check_reconstruction(
            irreps,
            kernel,
            # sampler=_uniform_sampling(n, G),
            sampler=_regular_sampling(n, G),
            preconditioning=preconditioning_so3('uniform'),
            activation=activation('identity'),
            regularization=0.
        )


def test_so3_elu():
    G = SO3(3)
    
    L = 3
    
    irreps = G.bl_regular_representation(L)
    
    # kernel = kernel_so3(L, 'dirac')
    kernel = kernel_so3(L, 'P', r=0.7)
    
    for n in range(irreps.size, 2 * irreps.size + 1, max(1, irreps.size // 10)):
        # for solid in ['tetra', 'cube', 'ico']:
        
        check_reconstruction(
            irreps,
            kernel,
            sampler=_uniform_sampling(n, G),
            # sampler=_so3_platonic_sampling(solid),
            # sampler=_regular_sampling(n, G),
            # sampler=_so3_sensing_sampling(n),
            preconditioning=preconditioning_so3('uniform'),
            activation=activation('elu'),
            regularization=0.  # 1e-8
        )


def test_so3_relu():
    G = SO3(3)
    
    L = 2
    
    irreps = G.bl_regular_representation(L)
    
    kernel = kernel_so3(L, 'P', r=0.9)
    # kernel = kernel_so3(L, 'dirac')
    
    print(irreps.size)
    for n in range(irreps.size, 2 * irreps.size + 15, max(1, irreps.size // 10)):
        # for solid in ['tetra', 'cube', 'ico']:
        # for n in range(6, 13):
        check_reconstruction(
            irreps,
            kernel,
            sampler=_uniform_sampling(n, G),
            # sampler=_so3_platonic_sampling(solid),
            # sampler=_regular_sampling(n, G),
            # sampler=_so3_hopf_sampling(n),
            # sampler=_so3_sensing_sampling(n),
            preconditioning=preconditioning_so3('uniform'),
            activation=activation('relu'),
            regularization=1e-9
        )


def test_so3_identity():
    G = SO3(3)
    
    L = 3
    
    irreps = G.bl_regular_representation(L)
    
    kernel = kernel_so3(L, 'P', r=0.7)
    
    for n in range(irreps.size, 2 * irreps.size + 1, max(1, irreps.size // 10)):
        check_reconstruction(
            irreps,
            kernel,
            sampler=_uniform_sampling(n, G),
            # sampler=_regular_sampling(n, G),
            preconditioning=preconditioning_so3('uniform'),
            activation=activation('identity'),
            regularization=0.
        )

#%%
import matplotlib.pyplot as plt

def plot_errs(ax, errs, label):
    l = ax.plot(errs[:, 0], errs[:, 1], label=label)
    color = l[0].get_color()
    ax.fill_between(errs[:, 0], errs[:, 1] - errs[:, 2], errs[:, 1] + errs[:, 2], label=None, color=color, alpha=0.2)


#%%

def compute_error(kernel, L):

    G = SO3(3)

    irreps = G.bl_regular_representation(L)

    errs = []
    # for n in range(irreps.size, int(1.5 * irreps.size) + 1, max(1, irreps.size // 20)):
    for solid in ['tetra', 'cube', 'ico']:

        e = []
        for _ in range(20):
            N, errors = check_reconstruction(
                irreps,
                kernel,
                samples=10,
                # sampler=_uniform_sampling(n, G),
                sampler=_so3_platonic_sampling(solid),
                # sampler=_regular_sampling(n, G),
                # sampler=_so3_sensing_sampling(n),
                preconditioning=preconditioning_so3('uniform'),
                activation=activation('elu'),
                regularization=1e-8
            )
            e.append(errors)
        e = np.stack(e)
        m = e.mean()
        s = e.std()
        print(f"{N}:\t {m:.5f} +- {s:.5f}")
        errs.append((N, m, s))

    errs = np.asarray(errs)
    return errs



L=2
kernel = kernel_so3(L, 'dirac')
errs_delta = compute_error(kernel, L)

kernel = kernel_so3(L, 'P', r=0.7)
errs_P = compute_error(kernel, L)


fig, ax = plt.subplots()
plot_errs(ax, errs_delta, r'$\delta$ + SIN')
plot_errs(ax, errs_P, r'$P$ + SIN')

plt.legend()
plt.ylim(0., errs_delta[:, 1].max()*1.1)

