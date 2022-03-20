import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union, Tuple, Callable, Dict

from .basis import KernelBasis, EmptyBasisException
from .steerable_basis import SteerableKernelBasis
from .spaces import PointRn

__all__ = [
    'GaussianRadialProfile',
    'SphericalShellsBasis'
]

def cart2pol(points):
    # computes the polar coordinates
    
    cumsum = np.sqrt(np.cumsum(points[::-1, :] ** 2, axis=0)[:0:-1, :])
    
    radii = cumsum[0, :]
    
    angles = np.arccos(points[:-1, :] / cumsum)
    
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
    
    points = np.empty((angles.shape[0] + 1, angles.shape[1]))
    
    mask = (radii > 1e-9).reshape(-1)
    points[:, ~mask] = 0.
    
    non_origin_count = mask.sum()
    cos = np.empty((angles.shape[0] + 1, non_origin_count))
    sin = np.empty((angles.shape[0] + 1, non_origin_count))
    
    cos[:-1, :] = np.cos(angles[:, mask])
    cos[-1, :] = 1.
    
    sin[1:, :] = np.sin(angles[:, mask])
    sin[0, :] = 1.
    sin = np.cumprod(sin, axis=0)
    
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
        
        self.radii = np.array(radii).reshape(1, 1, -1, 1)
        self.sigma = np.array(sigma).reshape(1, 1, -1, 1)
    
    def sample(self, radii: np.ndarray, out: np.ndarray = None) -> np.ndarray:
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
            out = np.empty((self.shape[0], self.shape[1], self.dim, radii.shape[1]))
        
        assert out.shape == (self.shape[0], self.shape[1], self.dim, radii.shape[1])
        
        radii = radii.reshape(1, 1, 1, -1)
        
        d = (self.radii - radii) ** 2
        
        out = np.exp(-0.5 * d / self.sigma ** 2, out=out)
        
        return out
    
    def __getitem__(self, r):
        assert r < self.dim
        return {"radius": self.radii[0, 0, r, 0], "sigma": self.sigma[0, 0, r, 0], "idx": r}
    
    def __eq__(self, other):
        if isinstance(other, GaussianRadialProfile):
            return np.allclose(self.radii, other.radii) and np.allclose(self.sigma, other.sigma)
        else:
            return False
    
    def __hash__(self):
        return hash(self.radii.tobytes()) + hash(self.sigma.tobytes())


class SphericalShellsBasis(KernelBasis):
    
    def __init__(self,
                 n: int,
                 angular: SteerableKernelBasis,
                 radial: GaussianRadialProfile,
                 filter: Callable[[Dict], bool] = None
                 ):
        r"""

        Build the tensor product basis of a radial profile basis and an spherical profile basis for kernels over the
        Euclidean space :math:`\R^n`.
        
        The kernel space is spanned by an independent basis for each shell.
        The kernel space over shells with positive radius is spanned the basis defined by the `angular` basis
        (an independent copy of each for each cell).
        The kernel over the shells with zero radius (the origin) is spanned by the `origin` basis.
        
        Given the bases :math:`O = \{o_i\}_i` for the origin, :math:`A = \{a_j\}_j` for the spherical shells and
        :math:`D = \{d_r\}_r` for the radial component (indexed by :math:`r \geq 0`, the radius different rings),
        this basis is defined as

        .. math::
            C = \left\{c_{i,j}(\bold{p}) := d_r(||\bold{p}||) a_j(\hat{\bold{p}}) \right\}_{r>0, j} \cup \{d_0(||\bold{p}||) o_i\}_i

        where :math:`(||\bold{p}||, \hat{\bold{p}})` are the polar coordinates of the point
        :math:`\bold{p} \in \R^n`.
        
        Note that the basis on the origin is represented as a simple `np.ndarray` of 3 dimensions, where the last one
        indexes the basis elements as :math:`i` above.
        
        The radial component is parametrized using :class:`~escnn.kernels.GaussianRadialProfile`.
        

        Args:
            n (int): dimension of the Euclidean base space
            angular (SteerableKernelBasis): the angular basis
            radial (GaussianRadialProfile): the basis for the radial profile
            filter (callable, optional): function used to filter out some basis elements. It takes as input a dict
                describing a basis element and should return a boolean value indicating whether to keep (`True`) or
                discard (`False`) the element. By default (`None`), all basis elements are kept.

        Attributes:
            ~.radial (GaussianRadialProfile): the radial basis
            ~.angular (SteerableKernelBasis): the angular basis
            ~.origin (SteerableKernelBasis): the basis for the origin

        """

        self.n = n
        self.radial = radial
        self.angular = angular
        
        # TODO - create singleton classes "spaces"
        # include Spheres S^d, points, lines, etc...
        # SpaceIsom should be a pair (HomSpace, Space)
        # the _is_point() method should belong to the space
        # here we should assert that the SpaceIsom is associated with the sphere S^{n-1}
        
        sphere = angular.X
        try:
            origin = SteerableKernelBasis(
                PointRn(self.n, sphere.G),
                angular.in_repr,
                angular.out_repr,
                angular._irrep_basis,
                # the origin contains only the frequency 0 harmonic
                harmonics=[sphere.zero_harmonic],
                **angular._irrep__basis_kwargs
            )
        except EmptyBasisException:
            origin = None

        if origin is not None:
            assert angular.shape == origin.shape[:2]
            assert angular.in_repr == origin.in_repr
            assert angular.out_repr == origin.out_repr
            assert angular.group == origin.group

            assert len(origin.js) == 1
            assert origin.js[0] == sphere.zero_harmonic
            assert origin.js[0] in angular.js
            assert (
                    origin.dim_harmonic(sphere.zero_harmonic)
                    ==
                    angular.dim_harmonic(sphere.zero_harmonic)
            )
            assert origin.dim_harmonic(sphere.zero_harmonic) == origin.dim
            
        self.origin = origin

        if filter is not None:
            self._filter = np.zeros(len(self.angular) * len(self.radial))
            _idx_map = []
            i = 0
            for attr1 in self.radial:
                for attr2 in self.angular:
                    attr = dict()
                    attr.update(attr1)
                    attr.update(attr2)
                    
                    if filter(attr):
                        self._filter[i] = 1
                        self._idx_map.append(i)
                    i += 1
                    
            dim = self._filter.sum()
            self._idx_map = np.array(_idx_map)
        else:
            self._filter = None
            self._idx_map = None
            dim = len(self.angular) * len(self.radial)
        
        super(SphericalShellsBasis, self).__init__(dim, (radial.shape[0] * angular.shape[0], radial.shape[1] * angular.shape[1]))
    
    def sample(self, points: np.ndarray, out: np.ndarray = None) -> np.ndarray:
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
        assert points.shape[0] == self.n
        
        # computes the polar coordinates
        # radii, angles = cart2pol(points)
        
        radii = np.sqrt((points ** 2).sum(axis=0, keepdims=True))
        
        non_origin_mask = (radii > 1e-99).reshape(-1)
        sphere = points[:, non_origin_mask] / radii[:, non_origin_mask]
        origin = points[:, ~non_origin_mask]

        if out is None:
            out = np.empty((self.shape[0], self.shape[1], self.dim, points.shape[1]))
        
        assert out.shape == (self.shape[0], self.shape[1], self.dim, points.shape[1])
        
        # sample the radial basis
        o1 = self.radial.sample(radii)

        # sample the angular basis
        o2 = np.empty((self.shape[0], self.shape[1], self.angular.dim, points.shape[1]))
        o2.fill(np.nan)

        # where r>0, we sample all frequencies
        o2[..., non_origin_mask] = self.angular.sample(sphere)
        
        # only frequency 0 is sampled at the origin. Other frequencies are set to 0
        if self.origin is not None:
            o2[..., :self.origin.dim, ~non_origin_mask] = self.origin.sample(origin)
            o2[..., self.origin.dim:, ~non_origin_mask] = 0.
        else:
            o2[..., ~non_origin_mask] = 0.
            
        assert not np.isnan(o1).any()
        assert not np.isnan(o2[..., non_origin_mask]).any()
        assert not np.isnan(o2[..., ~non_origin_mask]).any()
        assert not np.isnan(o2).any()

        m, n, a, p = o1.shape
        q, r, b, p = o2.shape
       
        if self._filter is None:
            np.einsum("mnap,qrbp->mqnrabp", o1, o2, out=out.reshape((m, q, n, r, a, b, p)))
            return out.reshape((q * m, n * r, self.dim, p))
        else:
            out[:] = np.einsum("mnap,qrb->mqnrabp", o1, o2).reshape((m * q, n * r, a * b, p))[..., self._filter, :]
            return out
    
    def __getitem__(self, idx):
        assert idx < self.dim
        
        if self._idx_map is None:
            _idx = idx
        else:
            _idx = self._idx_map[idx]
        
        idx1, idx2 = divmod(_idx, self.angular.dim)
        attr1 = self.radial[idx1]
        attr2 = self.angular[idx2]
        
        assert attr1["idx"] == idx1
        assert attr2["idx"] == idx2

        attr = dict()
        attr.update(attr1)
        attr.update(attr2)

        attr["idx"] = idx
        attr["idx1"] = attr1["idx"]
        attr["idx2"] = attr2["idx"]

        return attr

    def __iter__(self):
        idx = 0
        i = 0

        # since this methods return iterables of attributes built on the fly, load all attributes first and then
        # iterate on these lists
        radial_attrs = list(self.radial)
        angular_attrs = list(self.angular)

        for idx1, attr1 in enumerate(radial_attrs):
            for idx2, attr2 in enumerate(angular_attrs):
                if self._filter is None or self._filter[i] == 1:
                    
                    assert attr1["idx"] == idx1
                    assert attr2["idx"] == idx2
                    
                    attr = dict()
                    attr.update(attr1)
                    attr.update(attr2)
                    attr["idx1"] = attr1["idx"]
                    attr["idx2"] = attr2["idx"]
                    attr["idx"] = idx

                    yield attr
                    idx += 1
                i += 1
    
    def __eq__(self, other):
        if isinstance(other, SphericalShellsBasis):
            return (
                    self.n == other.n and
                    self.radial == other.radial and
                    self.angular == other.angular and
                    self.origin == other.origin and
                    self._filter == other._filter
            )
        else:
            return False
    
    def __hash__(self):
        return self.n + hash(self.radial) + hash(self.angular) + hash(self.origin) + hash(self._filter)


if __name__ == "__main__":
    
    for _ in range(100):
        for n in range(2, 6):
            x = np.random.randn(n, 4)
            
            radii, angles = cart2pol(x)
            y = pol2cart(radii, angles)
            
            print(x)
            print(radii)
            print(angles)
            print(y)
            assert np.allclose(x, y)
