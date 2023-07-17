import numpy as np
import torch

from escnn.group import *

from typing import List, Union, Tuple, Callable, Dict, Iterable

from escnn.kernels import *
from escnn.nn import *


__all__ = [
    'NormalizedR3Conv',
]


class GaussianRadialProfileNormalized(KernelBasis):
    
    def __init__(self, radii: List[float], sigma: Union[List[float], float]):
        r"""

        Basis for kernels defined over a radius in :math:`\R^+_0`.

        Each basis element is defined as a Gaussian function.
        Different basis elements are centered at different radii (``rings``) and can possibly be associated with
        different widths (``sigma``).

        More precisely, the following basis is implemented:

        .. math::

            \mathcal{B} = \left\{ b_i (r) :=  \exp \left( \frac{ \left( r - r_i \right)^2}{2 \sigma_i^2} \right) \right\}_i

        In order to build a complete basis of kernels, you should combine this basis with a basis which defines the
        angular profile, see for example :class:`~escnn.kernels.SphericalShellsBasis` or
        :class:`~escnn.kernels.CircularShellsBasis`.

        Args:
            radii (list): centers of each basis element. They should be different and spread to cover all
                domain of interest
            sigma (list or float): widths of each element. Can potentially be different.


        """
        
        if isinstance(sigma, float):
            sigma = [sigma] * len(radii)
        
        assert len(radii) == len(sigma)
        assert isinstance(radii, list)
        
        for r in radii:
            assert r >= 0.
        
        for s in sigma:
            assert s > 0.
        
        super(GaussianRadialProfileNormalized, self).__init__(len(radii), (1, 1))
        
        self.register_buffer('radii', torch.tensor(radii, dtype=torch.float32).reshape(1, -1, 1, 1))
        self.register_buffer('sigma', torch.tensor(sigma, dtype=torch.float32).reshape(1, -1, 1, 1))
    
    def sample(self, radii: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        r"""

        Sample the continuous basis elements on the discrete set of radii in ``radii``.
        Optionally, store the resulting multidimentional array in ``out``.

        ``radii`` must be an array of shape `(N, 1)`, where `N` is the number of points.

        Args:
            radii (~torch.Tensor): radii where to evaluate the basis elements
            out (~torch.Tensor, optional): pre-existing array to use to store the output

        Returns:
            the sampled basis

        """
        assert len(radii.shape) == 2
        assert radii.shape[1] == 1
        S = radii.shape[0]
        
        if out is None:
            out = torch.empty((S, self.dim, self.shape[0], self.shape[1]), device=radii.device, dtype=radii.dtype)
        
        assert out.shape == (S, self.dim, self.shape[0], self.shape[1])
        
        radii = radii.reshape(-1, 1, 1, 1)

        assert not torch.isnan(radii).any()

        d = (self.radii - radii) ** 2

        if radii.requires_grad:
            unnormalized_out = torch.exp(-0.5 * d / self.sigma ** 2, out=out)
            out[:] = unnormalized_out / unnormalized_out.sum(1, keepdims=True)
        else:
            out = torch.exp(-0.5 * d / self.sigma ** 2, out=out)
            out /= out.sum(1, keepdims=True)

        return out
    
    def __getitem__(self, r):
        assert r < self.dim
        return {"radius": self.radii[0, r, 0, 0].item(), "sigma": self.sigma[0, r, 0, 0].item(), "idx": r}
    
    def __eq__(self, other):
        if isinstance(other, GaussianRadialProfile):
            return (
                        torch.allclose(self.radii, other.radii.to(self.radii.device))
                    and torch.allclose(self.sigma, other.sigma.to(self.sigma.device))
            )
        else:
            return False
    
    def __hash__(self):
        return hash(self.radii.cpu().numpy().tobytes()) + hash(self.sigma.cpu().numpy().tobytes())


def normalized_kernels_SO3_act_R3(in_repr: Representation, out_repr: Representation,
                       radii: List[float],
                       # sigma: Union[List[float], float],
                       maximum_frequency: int = None,
                       adjoint: np.ndarray = None,
                       filter: Callable[[Dict], bool] = None
                       ) -> KernelBasis:
    r"""

    Builds a basis for convolutional kernels equivariant to continuous rotations, modeled by the
    group :math:`SO(3)`.
    ``in_repr`` and ``out_repr`` need to be :class:`~escnn.group.Representation` s of :class:`~escnn.group.SO3`.

    Because the equivariance constraints allow any choice of radial profile, we use a
    :class:`~escnn.kernels.GaussianRadialProfile`.
    ``radii`` specifies the radial distances at which the rings are centered while ``sigma`` contains the width of each
    of these rings (see :class:`~escnn.kernels.GaussianRadialProfile`).

    Args:
        in_repr (Representation): the representation specifying the transformation of the input feature field
        out_repr (Representation): the representation specifying the transformation of the output feature field
        radii (list): radii of the rings defining the basis for the radial profile
        # sigma (list or float): widths of the rings defining the basis for the radial profile
        adjoint (~numpy.ndarray, optional): 3x3 orthogonal matrix defining a change of basis on the base space

    """
    assert in_repr.group == out_repr.group

    group = in_repr.group

    assert isinstance(group, SO3)

    sigma = 1. / np.sqrt(2. * np.log(2)) / len(radii)

    radial_profile = GaussianRadialProfileNormalized(radii, sigma)

    if maximum_frequency is None:
        max_in_freq = max(freq for freq, in in_repr.irreps)
        max_out_freq = max(freq for freq, in out_repr.irreps)
        maximum_frequency = max_in_freq + max_out_freq

    basis = SteerableKernelBasis(
        SphericalShellsBasis(maximum_frequency, radial_profile, filter=filter),
        in_repr, out_repr,
        RestrictedWignerEckartBasis,
        sg_id='so3'
    )

    if adjoint is not None and not np.allclose(adjoint, np.eye(2)):
        assert adjoint.shape == (3, 3)
        basis = AdjointBasis(basis, adjoint)

    return basis


_fields_intertwiners_basis_memory_normaliedSO3 = dict()


class NormalizedR3Conv(R3Conv):

    def _build_kernel_basis(self, in_repr: Representation, out_repr: Representation) -> KernelBasis:

        # sigma = self._sigma
        rings = self._rings
        maximum_frequency=self._maximum_frequency
        # build the key
        key = dict()
        key["rings"] = tuple(rings)
        key["maximum_requency"] = maximum_frequency
        key = tuple(sorted(key.items()))

        if (in_repr.name, out_repr.name) not in _fields_intertwiners_basis_memory_normaliedSO3[key]:
            basis = normalized_kernels_SO3_act_R3(in_repr, out_repr, rings, maximum_frequency=maximum_frequency)
            # store the basis in the dictionary
            _fields_intertwiners_basis_memory_normaliedSO3[key][(in_repr.name, out_repr.name)] = basis
        # return the dictionary with all the basis built for this filter size
        return _fields_intertwiners_basis_memory_normaliedSO3[key][(in_repr.name, out_repr.name)]

