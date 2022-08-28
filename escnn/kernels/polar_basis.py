import numpy as np
import torch

from escnn.group import *

from abc import ABC, abstractmethod
from typing import List, Union, Tuple, Callable, Dict, Iterable

from .basis import KernelBasis, EmptyBasisException
from .steerable_filters_basis import SteerableFiltersBasis


__all__ = [
    'GaussianRadialProfile',
    'SphericalShellsBasis'
]

def cart2pol(points: torch.Tensor):
    # computes the polar coordinates
    
    cumsum = torch.sqrt(torch.cumsum(points[::-1, :] ** 2, dim=0))
    # cumsum = cumsum[:0:-1, :]
    cumsum = cumsum.flip(0)[:-1, :]

    radii = cumsum[0, :]
    
    angles = torch.acos(points[:-1, :] / cumsum)
    
    mask = points[-1, :] < 0
    angles[-1, mask] = 2 * np.pi - angles[-1, mask]
    
    # the angle at the origin is not well defined
    angles[:, radii.reshape(-1) < 1e-9] = np.nan
    
    return radii, angles


def pol2cart(radii, angles):
    assert len(radii.shape) == 2
    assert len(angles.shape) == 2
    assert radii.shape[0] == 1
    assert angles.shape[0] > 0
    assert radii.shape[1] == angles.shape[1]
    
    points = torch.empty(angles.shape[0] + 1, angles.shape[1], device=angles.device, dtype=angles.dtype)
    
    mask = (radii > 1e-9).reshape(-1)
    points[:, ~mask] = 0.
    
    non_origin_count = mask.sum()
    cos = torch.empty((angles.shape[0] + 1, non_origin_count), device=angles.device, dtype=angles.dtype)
    sin = torch.empty((angles.shape[0] + 1, non_origin_count), device=angles.device, dtype=angles.dtype)
    
    cos[:-1, :] = torch.cos(angles[:, mask])
    cos[-1, :] = 1.
    
    sin[1:, :] = torch.sin(angles[:, mask])
    sin[0, :] = 1.
    sin = torch.cumprod(sin, dim=0)
    
    points[:, mask] = cos * sin * radii[:, mask]
    
    return points


class GaussianRadialProfile(KernelBasis):
    
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
        angular profile through :class:`~escnn.kernels.SphericalShellsBasis`.


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
        
        super(GaussianRadialProfile, self).__init__(len(radii), (1, 1))
        
        self.register_buffer('radii', torch.tensor(radii, dtype=torch.float32).reshape(1, 1, -1, 1))
        self.register_buffer('sigma', torch.tensor(sigma, dtype=torch.float32).reshape(1, 1, -1, 1))
    
    def sample(self, radii: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        r"""

        Sample the continuous basis elements on the discrete set of radii in ``radii``.
        Optionally, store the resulting multidimentional array in ``out``.

        ``radii`` must be an array of shape `(1, N)`, where `N` is the number of points.

        Args:
            radii (~numpy.ndarray): radii where to evaluate the basis elements
            out (~numpy.ndarray, optional): pre-existing array to use to store the output

        Returns:
            the sampled basis

        """
        assert len(radii.shape) == 2
        assert radii.shape[0] == 1
        
        if out is None:
            out = torch.empty((self.shape[0], self.shape[1], self.dim, radii.shape[1]), device=radii.device, dtype=radii.dtype)
        
        assert out.shape == (self.shape[0], self.shape[1], self.dim, radii.shape[1])
        
        radii = radii.reshape(1, 1, 1, -1)
        
        d = (self.radii - radii) ** 2
        
        out = torch.exp(-0.5 * d / self.sigma ** 2, out=out)
        
        return out
    
    def __getitem__(self, r):
        assert r < self.dim
        return {"radius": self.radii[0, 0, r, 0], "sigma": self.sigma[0, 0, r, 0], "idx": r}
    
    def __eq__(self, other):
        if isinstance(other, GaussianRadialProfile):
            return torch.allclose(self.radii, other.radii) and torch.allclose(self.sigma, other.sigma)
        else:
            return False
    
    def __hash__(self):
        return hash(self.radii.cpu().numpy().tobytes()) + hash(self.sigma.cpu().numpy().tobytes())



from lie_learn.representations.SO3.spherical_harmonics import rsh


def spherical_harmonics(points: torch.Tensor, L: int):

    assert len(points.shape) == 2
    assert points.shape[0] == 3

    device = points.device
    dtype = points.dtype

    S = points.shape[1]

    radii = torch.norm(points, dim=0).detach().cpu().numpy()
    x, y, z = points.detach().cpu().numpy()

    angles = np.empty((2, S))
    angles[0, :] = np.arccos(np.clip(z / radii, -1., 1.))
    angles[1, :] = np.arctan2(y, x)

    Y = np.empty(((L+1)**2, S))
    for l in range(L+1):
        for m in range(-l, l + 1):
            Y[l**2 + m + l, :] = rsh(l, m, np.pi - angles[0, :], angles[1, :])

        # the central column of the Wigner D Matrices is proportional to the corresponding Spherical Harmonic
        # we need to correct by this proportion factor
        Y[l**2:(l+1)**2, ...] *= np.sqrt(4 * np.pi / (2 * l + 1))
        if l % 2 == 1:
            Y[l**2:(l+1)**2, ...] *= -1

    return torch.tensor(Y, device=device, dtype=dtype)


def circular_harmonics(points: torch.Tensor, L: int, phase: float = 0.):

    assert len(points.shape) == 2
    assert points.shape[0] == 2

    device = points.device
    dtype = points.dtype

    S = points.shape[1]

    # radii = torch.norm(points, dim=0).detach().cpu().numpy()
    x, y = points

    angles = torch.atan2(y, x).view(1, 1, S) - phase

    freqs = torch.arange(1, L+1, device=device, dtype=dtype).view(1, L, 1)

    freqs_times_angles = freqs * angles

    del freqs, angles

    Y = torch.empty((2 * L + 1, S), dtype=dtype, device=device)

    Y[0, ...] = 1.
    Y[1::2, ...] = torch.cos(freqs_times_angles)
    Y[2::2, ...] = torch.sin(freqs_times_angles)

    return Y


class SphericalShellsBasis(SteerableFiltersBasis):

    def __init__(self,
                 L: int,
                 radial: GaussianRadialProfile,
                 filter: Callable[[Dict], bool] = None
                 ):
        r"""

        Build the tensor product basis of a radial profile basis and a spherical harmonics basis for kernels over the
        Euclidean space :math:`\R^3`.
        
        The kernel space is spanned by an independent basis for each shell.
        The kernel space over shells with positive radius is spanned by spherical harmonics of frequency up to `L`
        (an independent copy of each for each cell).
        The kernel over the shells with zero radius (the origin) is only spanned by the frequency `0` harmonic.
        
        Given the bases :math:`O = \{o_i\}_i` for the origin, :math:`A = \{a_j\}_j` for the spherical shells and
        :math:`D = \{d_r\}_r` for the radial component (indexed by :math:`r \geq 0`, the radius different rings),
        this basis is defined as

        .. math::
            C = \left\{c_{i,j}(\bold{p}) := d_r(||\bold{p}||) a_j(\hat{\bold{p}}) \right\}_{r>0, j} \cup \{d_0(||\bold{p}||) o_i\}_i

        where :math:`(||\bold{p}||, \hat{\bold{p}})` are the polar coordinates of the point
        :math:`\bold{p} \in \R^n`.
        
        Note that the basis on the origin is represented as a simple `torch.Tensor` of 3 dimensions, where the last one
        indexes the basis elements as :math:`i` above.
        
        The radial component is parametrized using :class:`~escnn.kernels.GaussianRadialProfile`.
        
        Args:
            L (int): the maximum spherical frequency
            radial (GaussianRadialProfile): the basis for the radial profile
            filter (callable, optional): function used to filter out some basis elements. It takes as input a dict
                describing a basis element and should return a boolean value indicating whether to keep (`True`) or
                discard (`False`) the element. By default (`None`), all basis elements are kept.

        Attributes:
            ~.radial (GaussianRadialProfile): the radial basis
            ~.L (int): the maximum spherical frequency

        """

        self.L = L

        assert isinstance(radial, GaussianRadialProfile)

        self._angular_dim = (L+1)**2

        # number of invariant subspaces
        self._num_inv_spaces = 0

        G = o3_group(L)

        if filter is not None:
            _filter = torch.zeros(self._angular_dim * len(radial), dtype=torch.bool)

            js = []
            _idx_map = []
            _steerable_idx_map = []
            i = 0
            steerable_i = 0
            for j in range(self.L+1):

                j_id = (j % 2, j)  # the id of the O(3) irrep
                attr2 = {
                    'irrep:' + k: v
                    for k, v in G.irrep(*j_id).attributes.items()
                }
                dim = 2 * j + 1

                multiplicity = 0

                for attr1 in radial:
                    attr = dict()
                    attr.update(attr1)
                    attr.update(attr2)

                    if filter(attr):
                        multiplicity += 1
                        _filter[i:i+dim] = 1
                        _idx_map += list(range(i, i+dim))
                        _steerable_idx_map.append(steerable_i)

                    i += dim
                    steerable_i += 1

                js.append(
                    (
                        (j%2, j), # the O(3) irrep ID
                        multiplicity
                    )
                )
                self._num_inv_spaces += multiplicity
                    
            self._idx_map = np.array(_idx_map)
            self._steerable_idx_map = np.array(_steerable_idx_map)
        else:
            _filter = None
            self._idx_map = None
            js = [
                (
                    (j % 2, j),  # the O(3) irrep ID
                    len(radial)
                )
                for j in range(L+1)
            ]

        super(SphericalShellsBasis, self).__init__(G, G.standard_representation(), js)

        self.radial = radial

        if _filter is not None:
            self.register_buffer('_filter', _filter)
        else:
            self._filter = None

    def sample(self, points: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        r"""

        Sample the continuous basis elements on a discrete set of ``points`` in the space :math:`\R^n`.
        Optionally, store the resulting multidimensional array in ``out``.

        ``points`` must be an array of shape `(n, N)` containing `N` points in the space.
        Note that the points are specified in cartesian coordinates :math:`(x, y, z, ...)`.

        Args:
            points (~numpy.ndarray): points in the n-dimensional Euclidean space where to evaluate the basis elements
            out (~numpy.ndarray, optional): pre-existing array to use to store the output

        Returns:
            the sampled basis

        """
        assert len(points.shape) == 2
        assert points.shape[0] == self.dimensionality

        S = points.shape[1]

        radii = torch.sqrt((points ** 2).sum(dim=0, keepdim=True))
        
        non_origin_mask = (radii > 1e-9).reshape(-1)
        sphere = points[:, non_origin_mask] / radii[:, non_origin_mask]

        if out is None:
            out = torch.empty(1, 1, self.dim, S, device=points.device, dtype=points.dtype)
        
        assert out.shape == (1, 1, self.dim, S)
        
        # sample the radial basis
        radial = self.radial.sample(radii)
        assert radial.shape == (1, 1, len(self.radial), S)
        radial = radial[0, 0]

        # sample the angular basis
        spherical = torch.empty(self._angular_dim, S, device=points.device, dtype=points.dtype)
        spherical[:] = np.nan

        # where r>0, we sample all frequencies
        spherical[:, non_origin_mask] = spherical_harmonics(sphere, self.L)
        
        # only frequency 0 is sampled at the origin. Other frequencies are set to 0
        spherical[:1, ~non_origin_mask] = 1.
        spherical[1:, ~non_origin_mask] = 0.

        assert not torch.isnan(radial).any()
        # assert not torch.isnan(spherical[..., non_origin_mask]).any()
        # assert not torch.isnan(spherical[..., ~non_origin_mask]).any()
        assert not torch.isnan(spherical).any()

        tensor_product = torch.einsum("ap,bp->abp", radial, spherical)

        n_radii = len(self.radial)

        if self._filter is None:
            tmp_out = out
        else:
            tmp_out = torch.empty(1, 1, self._angular_dim*n_radii, S, device=points.device, dtype=points.dtype)

        for j in range(self.L+1):
            first, last = j**2, (j+1)**2
            tmp_out[0, 0, first * n_radii:last * n_radii, :].view(n_radii, 2*j+1, S)[:] = tensor_product[:, first:last, :]

        if self._filter is not None:
            out[:] = tmp_out[..., self._filter, :]

        return out

    def steerable_attrs_iter(self):
        # This attributes don't describe a single basis element but a group of basis elements which span an invariant
        # subspace. This is needed to generate the attributes of the SteerableKernelBasis

        idx = 0
        i = 0

        # since this methods return iterables of attributes built on the fly, load all attributes first and then
        # iterate on these lists
        radial_attrs = list(self.radial)

        for j in range(self.L + 1):
            attr1 = {
                'irrep:' + k: v
                for k, v in self.group.irrep(j%2, j).attributes.items()
            }

            for radial_idx, attr2 in enumerate(radial_attrs):
                if self._filter is None or (self._filter[i:i+2*j+1] == 1).all():
                    assert attr2["idx"] == radial_idx

                    attr = dict()
                    attr.update(attr1)
                    attr.update(attr2)
                    attr["idx"] = idx
                    attr["radial_idx"] = radial_idx
                    attr["j"] = (j % 2, j)  # the id of the O(3) irrep
                    attr["shape"] = (1, 1)

                    yield attr
                    idx += 1
                i += 2*j+1

    def steerable_attrs(self, idx):
        # This attributes don't describe a single basis element but a group of basis elements which span an invariant
        # subspace. This is needed to generate the attributes of the SteerableKernelBasis

        assert idx < self._num_inv_spaces

        if self._steerable_idx_map is None:
            _idx = idx
        else:
            _idx = self._steerable_idx_map[idx]

        j, radial_idx = divmod(_idx, len(self.radial))

        attr2 = self.radial[radial_idx]

        assert attr2["idx"] == radial_idx

        attr = dict()
        attr.update(attr2)

        j = (j%2, j) # the id of the O(3) irrep
        attr = {
            'irrep:'+k: v
            for k, v in self.group.irrep(*j).attributes.items()
        }
        attr["j"] = j

        attr["idx"] = idx
        attr["radial_idx"] = radial_idx
        attr["shape"] = (1, 1)

        return attr

    def steerable_attrs_j_iter(self, j: Tuple) -> Iterable:
        # This attributes don't describe a single basis element but a group of basis elements which span an invariant
        # subspace. This is needed to generate the attributes of the SteerableKernelBasis

        j_id = j
        f, j = j_id

        if f != j%2:
            return

        idx = sum(self.multiplicity(_j) for _j in range(j))
        i = 0

        # since this methods return iterables of attributes built on the fly, load all attributes first and then
        # iterate on these lists
        radial_attrs = list(self.radial)

        attr1 = {
            'irrep:' + k: v
            for k, v in self.group.irrep(*j_id).attributes.items()
        }

        for radial_idx, attr2 in enumerate(radial_attrs):
            if self._filter is None or (self._filter[i:i+2*j+1] == 1).all():
                assert attr2["idx"] == radial_idx

                attr = dict()
                attr.update(attr1)
                attr.update(attr2)
                attr["idx"] = idx
                attr["radial_idx"] = radial_idx
                attr["j"] = j_id
                attr["shape"] = (1, 1)

                yield attr
                idx += 1
            i += 2*j+1

    def steerable_attrs_j(self, j: Tuple, idx) -> Dict:
        # This attributes don't describe a single basis element but a group of basis elements which span an invariant
        # subspace. This is needed to generate the attributes of the SteerableKernelBasis

        j_id = j
        f, j = j_id

        if f != j % 2:
            return

        assert idx < self.multiplicity(j)

        idx += sum(self.multiplicity(_j) for _j in range(j))

        if self._steerable_idx_map is None:
            _idx = idx
        else:
            _idx = self._steerable_idx_map[idx]

        _j, radial_idx = divmod(_idx, len(self.radial))
        assert _j == j, (j, _j)

        attr1 = {
            'irrep:' + k: v
            for k, v in self.group.irrep(*j_id).attributes.items()
        }

        attr2 = self.radial[radial_idx]

        assert attr2["idx"] == radial_idx

        attr = dict()
        attr.update(attr1)
        attr.update(attr2)

        attr["idx"] = idx
        attr["radial_idx"] = radial_idx
        attr["j"] = j_id
        attr["shape"] = (1, 1)

        return attr

    def __getitem__(self, idx):
        assert idx < self.dim
        
        if self._idx_map is None:
            _idx = idx
        else:
            _idx = self._idx_map[idx]

        j = int(np.floor(np.sqrt(_idx // len(self.radial))))

        assert j**2 * len(self.radial) <= _idx < (j+1)**2 * len(self.radial), (_idx, j, self.L, len(self.radial))

        j_idx = _idx - j**2 * len(self.radial)
        
        radial_idx, m = divmod(j_idx, 2*j+1)

        j_id = (j%2, j) # the id of the O(3) irrep
        attr1 = {
            'irrep:' + k: v
            for k, v in self.group.irrep(*j_id).attributes.items()
        }

        attr2 = self.radial[radial_idx]

        assert attr2["idx"] == radial_idx

        attr = dict()
        attr.update(attr1)
        attr.update(attr2)

        attr["idx"] = idx
        attr["radial_idx"] = radial_idx
        attr["j"] = j_id
        attr["m"] = m
        attr["shape"] = (1, 1)

        return attr

    def __iter__(self):
        idx = 0
        i = 0

        # since this methods return iterables of attributes built on the fly, load all attributes first and then
        # iterate on these lists
        radial_attrs = list(self.radial)

        for j in range(self.L+1):

            j_id = (j % 2, j)  # the id of the O(3) irrep
            attr1 = {
                'irrep:' + k: v
                for k, v in self.group.irrep(*j_id).attributes.items()
            }
            for radial_idx, attr2 in enumerate(radial_attrs):
                for m in range(2*j+1):
                    if self._filter is None or self._filter[i] == 1:
                        assert attr2["idx"] == radial_idx

                        attr = dict()
                        attr.update(attr1)
                        attr.update(attr2)
                        attr["idx"] = idx
                        attr["radial_idx"] = radial_idx
                        attr["j"] = j_id
                        attr["m"] = m
                        attr["shape"] = (1, 1)

                        yield attr
                        idx += 1
                    i += 1
    
    def __eq__(self, other):
        if isinstance(other, SphericalShellsBasis):
            return (
                    self.radial == other.radial and
                    self.L == other.L and
                    self._filter == other._filter
            )
        else:
            return False
    
    def __hash__(self):
        return self.L + hash(self.radial) + hash(self._filter)


class CircularShellsBasis(SteerableFiltersBasis):

    def __init__(self,
                 L: int,
                 radial: GaussianRadialProfile,
                 filter: Callable[[Dict], bool] = None,
                 axis: float = np.pi/2,
                 ):
        r"""

        Build the tensor product basis of a radial profile basis and a circular harmonics basis for kernels over the
        Euclidean space :math:`\R^2`.

        The kernel space is spanned by an independent basis for each shell.
        The kernel space over shells with positive radius is spanned by circular harmonics of frequency up to `L`
        (an independent copy of each for each cell).
        The kernel over the shells with zero radius (the origin) is only spanned by the frequency `0` harmonic.

        Given the bases :math:`O = \{o_i\}_i` for the origin, :math:`A = \{a_j\}_j` for the circular shells and
        :math:`D = \{d_r\}_r` for the radial component (indexed by :math:`r \geq 0`, the radius different rings),
        this basis is defined as

        .. math::
            C = \left\{c_{i,j}(\bold{p}) := d_r(||\bold{p}||) a_j(\hat{\bold{p}}) \right\}_{r>0, j} \cup \{d_0(||\bold{p}||) o_i\}_i

        where :math:`(||\bold{p}||, \hat{\bold{p}})` are the polar coordinates of the point
        :math:`\bold{p} \in \R^n`.

        Note that the basis on the origin is represented as a simple `torch.Tensor` of 3 dimensions, where the last one
        indexes the basis elements as :math:`i` above.

        The radial component is parametrized using :class:`~escnn.kernels.GaussianRadialProfile`.

        Args:
            L (int): the maximum circular frequency
            radial (GaussianRadialProfile): the basis for the radial profile
            filter (callable, optional): function used to filter out some basis elements. It takes as input a dict
                describing a basis element and should return a boolean value indicating whether to keep (`True`) or
                discard (`False`) the element. By default (`None`), all basis elements are kept.

        Attributes:
            ~.radial (GaussianRadialProfile): the radial basis
            ~.L (int): the maximum circular frequency

        """

        self.L = L

        assert isinstance(radial, GaussianRadialProfile)

        self._angular_dim = 2*L+1

        # number of invariant subspaces
        self._num_inv_spaces = 0

        G = o2_group(L)

        if filter is not None:
            _filter = torch.zeros(self._angular_dim * len(radial), dtype=torch.bool)

            js = []
            _idx_map = []
            _steerable_idx_map = []
            i = 0
            steerable_i = 0
            for j in range(self.L + 1):

                attr2 = {
                    'irrep:' + k: v
                    for k, v in G.irrep(int(j>0), j).attributes.items()
                }
                attr2['j'] = (int(j>0), j)  # the id of the O(2) irrep

                dim = 2 if j > 0 else 1

                multiplicity = 0

                for attr1 in radial:
                    attr = dict()
                    attr.update(attr1)
                    attr.update(attr2)

                    if filter(attr):
                        multiplicity += 1
                        _filter[i:i + dim] = 1
                        _idx_map += list(range(i, i + dim))
                        _steerable_idx_map.append(steerable_i)

                    i += dim
                    steerable_i += 1

                js.append(
                    (
                        (int(j>0), j),  # the O(2) irrep ID
                        multiplicity
                    )
                )
                self._num_inv_spaces += multiplicity

            self._idx_map = np.array(_idx_map)
            self._steerable_idx_map = np.array(_steerable_idx_map)
        else:
            _filter = None
            self._idx_map = None
            js = [
                (
                    (int(j>0), j),  # the O(2) irrep ID
                    len(radial)
                )
                for j in range(L + 1)
            ]

        self.axis = axis

        action = G.standard_representation()
        action = change_basis(action, action(G.element((0, axis), 'radians')), name=f'StandardAction|axis=[{axis}]')
        super(CircularShellsBasis, self).__init__(G, action, js)

        self.radial = radial

        if _filter is None:
            self._filter = None
        else:
            self.register_buffer('_filter', _filter)

    def sample(self, points: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
        r"""

        Sample the continuous basis elements on a discrete set of ``points`` in the space :math:`\R^n`.
        Optionally, store the resulting multidimensional array in ``out``.

        ``points`` must be an array of shape `(n, N)` containing `N` points in the space.
        Note that the points are specified in cartesian coordinates :math:`(x, y, z, ...)`.

        Args:
            points (~numpy.ndarray): points in the n-dimensional Euclidean space where to evaluate the basis elements
            out (~numpy.ndarray, optional): pre-existing array to use to store the output

        Returns:
            the sampled basis

        """
        assert len(points.shape) == 2
        assert points.shape[0] == self.dimensionality

        S = points.shape[1]

        radii = torch.sqrt((points ** 2).sum(dim=0, keepdim=True))

        non_origin_mask = (radii > 1e-9).reshape(-1)
        sphere = points[:, non_origin_mask] / radii[:, non_origin_mask]

        if out is None:
            out = torch.empty(1, 1, self.dim, S, device=points.device, dtype=points.dtype)

        assert out.shape == (1, 1, self.dim, S)

        # sample the radial basis
        radial = self.radial.sample(radii)
        assert radial.shape[:2] == (1, 1)
        radial = radial[0, 0]

        # sample the angular basis
        circular = torch.empty(self._angular_dim, S, device=points.device, dtype=points.dtype)
        circular[:] = np.nan

        # where r>0, we sample all frequencies
        circular[:, non_origin_mask] = circular_harmonics(sphere, self.L, phase=self.axis)

        # only frequency 0 is sampled at the origin. Other frequencies are set to 0
        circular[:1, ~non_origin_mask] = 1.
        circular[1:, ~non_origin_mask] = 0.

        assert not torch.isnan(radial).any()
        # assert not torch.isnan(circular[..., non_origin_mask]).any()
        # assert not torch.isnan(circular[..., ~non_origin_mask]).any()
        assert not torch.isnan(circular).any()

        tensor_product = torch.einsum("ap,bp->abp", radial, circular)

        n_radii = len(self.radial)

        if self._filter is None:
            tmp_out = out
        else:
            tmp_out = torch.empty(1, 1, self._angular_dim*n_radii, S, device=points.device, dtype=points.dtype)

        for j in range(self.L+1):
            dim = 2 if j > 0 else 1
            last = 2*j+1
            first = last - dim
            tmp_out[0, 0, first * n_radii:last * n_radii, :].view(n_radii, dim, S)[:] = tensor_product[:, first:last, :]

        if self._filter is not None:
            out[:] = tmp_out[..., self._filter, :]

        return out

    def steerable_attrs_iter(self):
        # This attributes don't describe a single basis element but a group of basis elements which span an invariant
        # subspace. This is needed to generate the attributes of the SteerableKernelBasis

        idx = 0
        i = 0

        # since this methods return iterables of attributes built on the fly, load all attributes first and then
        # iterate on these lists
        radial_attrs = list(self.radial)

        for j in range(self.L + 1):
            dim = 2 if j > 0 else 1
            j_id = (int(j>0), j)  # the id of the O(2) irrep

            attr1 = {
                'irrep:' + k: v
                for k, v in self.group.irrep(*j_id).attributes.items()
            }

            for radial_idx, attr2 in enumerate(radial_attrs):
                if self._filter is None or (self._filter[i:i + dim] == 1).all():
                    assert attr2["idx"] == radial_idx

                    attr = dict()
                    attr.update(attr1)
                    attr.update(attr2)
                    attr["idx"] = idx
                    attr["radial_idx"] = radial_idx
                    attr["j"] = j_id
                    attr["shape"] = (1, 1)

                    yield attr
                    idx += 1
                i += dim

    def steerable_attrs(self, idx):
        # This attributes don't describe a single basis element but a group of basis elements which span an invariant
        # subspace. This is needed to generate the attributes of the SteerableKernelBasis

        assert idx < self._num_inv_spaces

        if self._steerable_idx_map is None:
            _idx = idx
        else:
            _idx = self._steerable_idx_map[idx]

        j, radial_idx = divmod(_idx, len(self.radial))

        j_id = (int(j > 0), j)  # the id of the O(2) irrep
        attr1 = {
            'irrep:' + k: v
            for k, v in self.group.irrep(*j_id).attributes.items()
        }

        attr2 = self.radial[radial_idx]

        assert attr2["idx"] == radial_idx

        attr = dict()
        attr.update(attr1)
        attr.update(attr2)

        attr["idx"] = idx
        attr["radial_idx"] = radial_idx
        attr["j"] = j_id
        attr["shape"] = (1, 1)

        return attr

    def steerable_attrs_j_iter(self, j: Tuple) -> Iterable:
        # This attributes don't describe a single basis element but a group of basis elements which span an invariant
        # subspace. This is needed to generate the attributes of the SteerableKernelBasis

        j_id = j
        f, j = j

        if f != int(j>0):
            return

        idx = sum(self.multiplicity(_j) for _j in range(j))
        dim = 2 if j > 0 else 1
        i = 0

        attr1 = {
            'irrep:' + k: v
            for k, v in self.group.irrep(*j_id).attributes.items()
        }
        # since this methods return iterables of attributes built on the fly, load all attributes first and then
        # iterate on these lists
        radial_attrs = list(self.radial)

        for radial_idx, attr2 in enumerate(radial_attrs):
            if self._filter is None or (self._filter[i:i + dim] == 1).all():
                assert attr2["idx"] == radial_idx

                attr = dict()
                attr.update(attr1)
                attr.update(attr2)
                attr["idx"] = idx
                attr["radial_idx"] = radial_idx
                attr["j"] = j_id
                attr["shape"] = (1, 1)

                yield attr
                idx += 1
            i += dim

    def steerable_attrs_j(self, j: Tuple, idx) -> Dict:
        # This attributes don't describe a single basis element but a group of basis elements which span an invariant
        # subspace. This is needed to generate the attributes of the SteerableKernelBasis

        j_id = j
        f, j = j_id

        if f != int(j>0):
            return

        assert idx < self.multiplicity(j)

        idx += sum(self.multiplicity(_j) for _j in range(j))

        if self._steerable_idx_map is None:
            _idx = idx
        else:
            _idx = self._steerable_idx_map[idx]

        _j, radial_idx = divmod(_idx, len(self.radial))
        assert _j == j, (j, _j)

        attr1 = {
            'irrep:' + k: v
            for k, v in self.group.irrep(*j_id).attributes.items()
        }

        attr2 = self.radial[radial_idx]

        assert attr2["idx"] == radial_idx

        attr = dict()
        attr.update(attr1)
        attr.update(attr2)

        attr["idx"] = idx
        attr["radial_idx"] = radial_idx
        attr["j"] = j_id
        attr["shape"] = (1, 1)

        return attr

    def __getitem__(self, idx):
        assert idx < self.dim

        if self._idx_map is None:
            _idx = idx
        else:
            _idx = self._idx_map[idx]

        j = (_idx // len(self.radial) + 1) //2

        assert (2*j+1) * len(self.radial) <= _idx < (2*j + 3) * len(self.radial), (_idx, j, self.L, len(self.radial))

        j_idx = _idx - (2*j +1) * len(self.radial)

        radial_idx, m = divmod(j_idx, 2 if j>0 else 1)

        j_id = (int(j>0), j)  # the id of the O(3) irrep
        attr1 = {
            'irrep:' + k: v
            for k, v in self.group.irrep(*j_id).attributes.items()
        }

        attr2 = self.radial[radial_idx]

        assert attr2["idx"] == radial_idx

        attr = dict()
        attr.update(attr1)
        attr.update(attr2)

        attr["idx"] = idx
        attr["radial_idx"] = radial_idx
        attr["j"] = j_id
        attr["m"] = m
        attr["shape"] = (1, 1)

        return attr

    def __iter__(self):
        idx = 0
        i = 0

        # since this methods return iterables of attributes built on the fly, load all attributes first and then
        # iterate on these lists
        radial_attrs = list(self.radial)

        for j in range(self.L + 1):
            dim = 2 if j>0 else 1
            j_id = (int(j>0), j)
            attr1 = {
                'irrep:' + k: v
                for k, v in self.group.irrep(*j_id).attributes.items()
            }
            for radial_idx, attr2 in enumerate(radial_attrs):
                for m in range(dim):
                    if self._filter is None or self._filter[i] == 1:
                        assert attr2["idx"] == radial_idx

                        attr = dict()
                        attr.update(attr1)
                        attr.update(attr2)
                        attr["idx"] = idx
                        attr["radial_idx"] = radial_idx
                        attr["j"] = j_id
                        attr["m"] = m
                        attr["shape"] = (1, 1)

                        yield attr
                        idx += 1
                    i += 1

    def __eq__(self, other):
        if isinstance(other, CircularShellsBasis):
            return (
                    self.radial == other.radial and
                    self.L == other.L and
                    self._filter == other._filter
            )
        else:
            return False

    def __hash__(self):
        return self.L + hash(self.radial) + hash(self._filter)


if __name__ == "__main__":

    for _ in range(100):
        for n in range(2, 6):
            x = torch.randn(n, 4)

            radii, angles = cart2pol(x)
            y = pol2cart(radii, angles)

            print(x)
            print(radii)
            print(angles)
            print(y)
            assert torch.allclose(x, y)
