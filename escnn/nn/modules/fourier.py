import numpy as np
import torch

from escnn.nn import FourierFieldType, GeometricTensor, GridTensor
from escnn.group import Group, GroupElement
from torch.nn import Module

from typing import Sequence, Optional

__all__ = ['FourierTransform', 'InverseFourierTransform']

# Docs:
# - Low level class, meant for building other modules
# - If use directly, possible to break equivariance

class InverseFourierTransform(Module):

    def __init__(
            self,
            in_type: FourierFieldType,
            out_grid: Sequence[GroupElement],
            *,
            normalize: bool = True,
    ):
        super().__init__()

        assert isinstance(in_type, FourierFieldType)

        self.in_type = in_type
        self.out_grid = list(out_grid)

        A = _build_ift(in_type, out_grid, normalize)
        A = torch.tensor(A, dtype=torch.get_default_dtype())

        self.register_buffer('A', A)

    def forward(self, input: GeometricTensor) -> GridTensor:
        assert input.type == self.in_type
        
        x_hat = input.tensor.view(
                input.shape[0],
                self.in_type.channels,
                self.in_type.rho.size,
                *input.shape[2:],
        )

        x = torch.einsum('bcf...,gf->bcg...', x_hat, self.A)

        return GridTensor(x, self.out_grid, input.coords)

class FourierTransform(Module):

    def __init__(
            self,
            in_grid,
            out_type,
            *,
            extra_irreps: Optional[list] = None,
            normalize: bool = True,
    ):
        super().__init__()

        assert isinstance(out_type, FourierFieldType)

        self.in_grid = in_grid
        self.out_type = out_type

        if extra_irreps is None:
            ift_type = out_type
        else:
            extra_irreps = [
                    x
                    for x in extra_irreps
                    if x not in out_type.bl_irreps
            ]
            ift_type = FourierFieldType(
                    out_type.gspace,
                    out_type.channels,
                    out_type.bl_irreps + extra_irreps,
                    subgroup_id=out_type.subgroup_id,
            )

        A = _build_ift(ift_type, in_grid, normalize)

        eps = 1e-8
        n = ift_type.rho.size
        Ainv = np.linalg.inv(A.T @ A + eps * np.eye(n)) @ A.T

        if extra_irreps is not None:
            Ainv = Ainv[:out_type.rho.size, :]

        Ainv = torch.tensor(Ainv, dtype=torch.get_default_dtype())
        
        self.register_buffer('Ainv', Ainv)

    def forward(self, input: GridTensor) -> GeometricTensor:
        assert input.grid == self.in_grid

        y = input.tensor

        y_hat = torch.einsum('bcg...,fg->bcf...', y, self.Ainv)

        y_hat = y_hat.reshape(y.shape[0], self.out_type.size, *y.shape[3:])

        return GeometricTensor(y_hat, self.out_type, input.coords)


def _build_ift(in_type: FourierFieldType, out_grid, normalize: bool):
    """
    Create a matrix that will apply an inverse Fourier transform to a feature 
    vector of the given *in_type*.
    """
    assert isinstance(in_type, FourierFieldType)

    G = in_type.fibergroup

    if in_type.subgroup_id is None:
        kernel = _build_regular_kernel(G, in_type.bl_irreps)
    else:
        kernel = _build_quotient_kernel(G, in_type.subgroup_id, in_type.bl_irreps)

    assert kernel.shape[0] == in_type.rho.size

    if normalize:
        kernel = kernel / np.linalg.norm(kernel)

    kernel = kernel.reshape(-1, 1)
    
    return np.concatenate(
        [
            in_type.rho(g) @ kernel
            for g in out_grid
        ], axis=1
    ).T

def _build_regular_kernel(G: Group, irreps: list[tuple]):
    kernel = []
    
    for irr in irreps:
        irr = G.irrep(*irr)
        
        c = int(irr.size//irr.sum_of_squares_constituents)
        k = irr(G.identity)[:, :c] * np.sqrt(irr.size)
        kernel.append(k.T.reshape(-1))
    
    kernel = np.concatenate(kernel)
    return kernel

def _build_quotient_kernel(G: Group, subgroup_id: tuple, irreps: list[tuple]):
    kernel = []
    
    X: HomSpace = G.homspace(subgroup_id)
    
    for irr in irreps:
        k = X._dirac_kernel_ft(irr, X.H.trivial_representation.id)
        kernel.append(k.T.reshape(-1))
    
    kernel = np.concatenate(kernel)
    return kernel
