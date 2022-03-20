import numpy as np

from escnn.group import *

from typing import Sequence, Tuple

from abc import ABC, abstractmethod

__all__ = [
    'SpaceIsomorphism',
    'CircleO2',
    'CircleSO2',
    'SphereO3',
    'SphereSO3',
    'PointRn',
    #
    'Icosidodecahedron',
    'Dodecahedron',
    'Icosahedron',
]


class SpaceIsomorphism(ABC):
    
    def __init__(self, dim: int, X: HomSpace, action: Representation):
        r"""
            Abstract class that defines an embedding of the homogeneous space :math:`X \cong G/H` into the
            Euclidean space :math:`\R^\text{dim}`.
            
            ``action`` should be a :class:`~escnn.group.Representation` of the group :math:`G` defining its linear
            action on the Euclidean space :math:`\R^\text{dim}`.
            
            The embedding map implemented by this class should be *equivariant* with respect to the natural action of
            :math:`G` on :math:`X` and the action of :math:`G` on :math:`\R^\text{dim}` via the representation
            ``action``.
            
            An instance of this class can be used for example to define steerable kernels over a homogeneous space and,
            then, use them on orbits of the group of interest in :math:`\R^d`; see for example
            :class:`~escnn.kernels.WignerEckartBasis` or :class:`~escnn.kernels.RestrictedWignerEckartBasis`.

        """
        
        assert isinstance(dim, int) and dim > 0
        
        self._dim = dim
        
        # HomSpace: homogeneous space of the group G
        self.X = X
        
        # Group: the group acting on the homogeneous space
        self.G = X.G
        # Group: the stabilizer group of the homogeneous space
        self.H = X.H
        
        assert action.group == self.G
        assert action.size == self.dim
        
        # Representation: the representation of the group ``G`` defining its action on the Euclidean space
        self.action = action
        
        self._H_trivial_id = self.H.trivial_representation.id
        
    @property
    def dim(self):
        r"""
            Integer defining the dimensionality of the Euclidean space where the homogeneous space is embedded.
            
        """
        return self._dim
    
    @property
    def zero_harmonic(self) -> Tuple:
        r"""
            Id of the irrep of ``G`` corresponding to the trivial representation and, therefore, to the harmonic of
            constant functions over the homogeneous space
        """
        return self.G.trivial_representation.id
        
    def dimension_basis(self, j: Tuple):
        r"""
        The dimensionality of the subspace of functions on the homogeneous space which transform according to the irrep
        of ``G`` identified by the id ``j``.
        This corresponds to the number of "frequency"-``j`` harmonics.
        
        The method returns a tuple containing the size of the irrep `j` and the number of subspaces transforming
        according to this irrep.
        The product of these numbers is the number of "frequency"-``j`` harmonics.
        
        .. seealso::
            :class:`~escnn.group.IrreducibleRepresentation` and
            :class:`~escnn.group.HomSpace`
        

        """
        return self.X.dimension_basis(j, self._H_trivial_id)[:-1]

    @abstractmethod
    def _is_point(self, points: np.ndarray) -> np.ndarray:
        pass
    
    @abstractmethod
    def _section(self, points: np.ndarray) -> Sequence[GroupElement]:
        pass

    @property
    @abstractmethod
    def _origin(self) -> np.ndarray:
        r'''
            The point in the Euclidean space where the origin of the homogeneous space is mapped to.
        '''
        pass

    def _projection(self, g: GroupElement) -> np.ndarray:
        # action(g) @ x_0
        # where x_0 is the projection of the identity element `e`
        return self.action(g) @ self._origin

    def is_point(self, points: np.ndarray) -> np.ndarray:
        r'''
            Check if each point in ``points`` belongs to the embedding of the homogeneous space into
            the Euclidean one.
        '''
        assert len(points.shape) == 2
        assert points.shape[0] == self.dim
        
        mask = self._is_point(points)
        
        assert mask.shape == (points.shape[1], )
        
        return mask

    def section(self, points: np.ndarray) -> Sequence[GroupElement]:
        r'''
            This method should implement the inverse of the embedding map composed with a section of the homogeneous
            space.
            In other words, the method should first map a point in the Euclidean space to a point
            :math:`x \in X\cong G/H` in the homogeneous space.
            One such point is a *coset*, i.e. :math:`x = gH` for some :math:`g \in G`.
            The method should, then, pick and return a representative element :math:`g' \in x=gH` of the coset.
        '''
        assert len(points.shape) == 2
        assert points.shape[0] == self.dim
        return self._section(points)

    def projection(self, g: GroupElement) -> np.ndarray:
        r'''
            This method should implement the projection map from :math:`G` to :math:`X\cong G/H` composed
            with the embedding map the homogeneous space in the Euclidean one.
            In other words, the method should first project a group element :math:`g \in G` to the coset
            :math:`x=gH \in X\cong G/H`.
            Then, the method embeds this point :math:`x \in X` in the Euclidean space.
        '''
        assert g.group == self.X.G
        
        point = self._projection(g)

        point = point.squeeze()
        assert len(point.shape) == 1
        assert point.shape[0] == self.dim
        return point.reshape(-1, 1)
        
    def _automatic_basis(self, points: np.ndarray, j) -> np.ndarray:
        r'''
            Automaticlly generates a basis for the homogeneous space by using the
            :meth:`~escnn.kernels.SpaceIsomorphism.section` method and the basis generated in
            :class:`~escnn.group.HomSpace` from the irreps coefficients of :math:`G`.
        '''
        
        assert len(points.shape) == 2
        assert points.shape[0] == self.dim
        
        S = points.shape[1]
        
        basis = []
        for p in range(S):
            g = self.section(points[:, p:p+1])[0]
            basis.append(
                self.X.basis(g, j, self._H_trivial_id)[..., 0]
            )
        
        return np.stack(basis, axis=2)
    
    def basis(self, points: np.ndarray, j) -> np.ndarray:
        r'''
            Sample all harmonic basis functions of frequency ``j`` on the input ``points``.
            
            By default, the basis is automatically generated by using the
            :meth:`~escnn.kernels.SpaceIsomorphism.section` method and the basis generated in
            :class:`~escnn.group.HomSpace` from the irreps coefficients of :math:`G`.
            
            This method is very general but often relatively inefficient with respect to ad hoc implementations for
            specific homogeneous spaces (e.g. for spherical harmonics).
            We recommend overriding this method in subclasses if such implementations are available.
            
        '''
        return self._automatic_basis(points, j)

    def __hash__(self):
        return self.dim + hash(self.X) + hash(self.action)
    
    def __eq__(self, other):
        if not isinstance(other, SpaceIsomorphism):
            return False
        else:
            # TODO: this checks that the two instances are equivalent, but they could have different implementation of
            # the basis... Do we want to still consider these instances the same?
            return self.dim == other.dim and self.X == other.X and self.action == other.action
    
    def _test_section_consistency(self):
        
        assert self._origin.shape == (self.dim, 1)
        
        for h in self.H.testing_elements():
            hx0 = self.action(self.X._inclusion(h)) @ self._origin
            assert np.allclose(self._origin, hx0)
        
        for _ in range(50):
            g = self.G.sample()
            x = self.projection(g)
            
            _g = self.section(x)[0]
            _x = self.projection(_g)
            
            assert np.allclose(x, _x)

    def _test_custom_basis_consistency(self):
        for psi in self.G.irreps():
            j = psi.id
            
            for _ in range(50):
                g = self.G.sample()
            
                x = self.projection(g)
                
                basis_default = self._automatic_basis(x, j)
                basis_custom = self.basis(x, j)
                
                shape = self.X.dimension_basis(j, self._H_trivial_id)[:-1]
                assert basis_custom.shape[:-1] == shape
                assert basis_default.shape[:-1] == shape
                
                # c = (basis_default * basis_custom).sum(axis=0) / (basis_default ** 2).sum(axis=0)
                # # assert np.allclose(c, 1.)
                # print('-----------------------')
                # print(j)
                # print(c[..., 0])
                # print('-----------------------')
                assert np.allclose(basis_default, basis_custom) #, rtol=5e-2, atol=1e-2)

    def _test_equivariance(self):
        for psi in self.G.irreps():
            j = psi.id
        
            for _ in range(10):
                x = self.projection(self.G.sample())
                
                Yx = self.basis(x, j)[..., 0]
                
                # for _ in range(10):
                #     g = self.G.sample()
                for g in self.G.testing_elements():
                    
                    gx = self.action(g) @ x
                    Ygx = self.basis(gx, j)[..., 0]
                    
                    gYx = psi(g) @ Yx
                    
                    assert np.allclose(gYx, Ygx)


class CircleSO2(SpaceIsomorphism):
    
    def __init__(self):
        super(CircleSO2, self).__init__(2, so2_group().homspace(1), so2_group().irrep(1))
    
    @property
    def _origin(self) -> np.ndarray:
        # The projection x_0 of the identity element `e` is the vector [1, 0]^T
        return np.array([1, 0]).reshape(-1, 1)
    
    def _is_point(self, points: np.ndarray) -> np.ndarray:
        # return (points**2).sum(axis=0) > 1e-8
        return np.isclose((points**2).sum(axis=0), 1.)
        
    def _section(self, points: np.ndarray) -> Sequence[GroupElement]:
        assert self._is_point(points).all()
        
        theta = np.arctan2(points[1, ...], points[0, ...])
        return [self.G.element(t) for t in theta]
    
    def _projection(self, g: GroupElement) -> np.ndarray:
        # return g.to('C')
        theta = g.to('radians')
        
        return np.asarray([np.cos(theta), np.sin(theta)])

    def basis(self, points: np.ndarray, j) -> np.ndarray:
        assert len(points.shape) == 2
        assert points.shape[0] == self.dim
        S = points.shape[1]
        
        j = self.G.get_irrep_id(j)[0]
        
        assert isinstance(j, int)

        assert self.is_point(points).all()
        
        if j == 0:
            return np.ones((1, 1, S))
        
        theta = np.arctan2(points[1, ...], points[0, ...])

        cos = np.cos(j * theta)
        sin = np.sin(j * theta)
        
        return np.stack([cos, sin], axis=0).reshape(2, 1, S)


class CircleO2(SpaceIsomorphism):
    
    def __init__(self, axis: float = np.pi / 2.):
        self.axis = axis
        
        o2 = o2_group()
        
        action = o2.irrep(1, 1)
        action = change_basis(action, action(o2.element((0, axis), 'radians')), name=f'StandardAction|axis=[{axis}]')
        
        super(CircleO2, self).__init__(2, o2.homspace((0., 1)), action)
        
        # super(CircleO2, self).__init__(2, o2.homspace((2*axis, 1)), action)

    @property
    def _origin(self) -> np.ndarray:
        # The projection x_0 of the identity element `e` is the vector [1, 0]^T
        # rotated (counter-clockwise) by `self.axis`
        return np.array([np.cos(self.axis), np.sin(self.axis)]).reshape(-1, 1)

    def _is_point(self, points: np.ndarray) -> np.ndarray:
        # return (points**2).sum(axis=0) > 1e-8
        return np.isclose((points**2).sum(axis=0), 1.)

    def _section(self, points: np.ndarray) -> Sequence[GroupElement]:
        assert self._is_point(points).all()

        theta = np.arctan2(points[1, ...], points[0, ...])
        theta -= self.axis
        return [self.G.element((0, t)) for t in theta]
    
    # def _projection(self, g: GroupElement) -> np.ndarray:
    #     f, theta = g.to('radians')
    #     if f:
    #         theta = theta / 2
    #     else:
    #         theta += self.axis
    #     return np.asarray([np.cos(theta), np.sin(theta)])
    
    def basis(self, points: np.ndarray, j) -> np.ndarray:
        assert len(points.shape) == 2
        assert points.shape[0] == self.dim
        S = points.shape[1]

        assert self.is_point(points).all()
        j = self.G.get_irrep_id(j)
        
        if j == (1, 0):
            return np.zeros((1, 0, S))
        
        if j == (0, 0):
            return np.ones((1, 1, S))

        j = j[1]

        theta = np.arctan2(points[1, ...], points[0, ...])
        
        theta -= self.axis
        
        cos = np.cos(j * theta)
        sin = np.sin(j * theta)
        
        return np.stack([cos, sin], axis=0).reshape(2, 1, S)


from lie_learn.representations.SO3.spherical_harmonics import rsh


class SphereSO3(SpaceIsomorphism):
    
    def __init__(self):
        super(SphereSO3, self).__init__(3, so3_group().homspace((False, -1)), so3_group().standard_representation())

    @property
    def _origin(self) -> np.ndarray:
        # The projection x_0 of the identity element `e` is the vector [0, 0, ..., 0, 1]^T
        # this choice of x_0 is convenient when we use SO(3)
        # so SO(2) subgroup is identitied with rotations around the Z axis
        return np.array([0, 0, 1]).reshape(-1, 1)

    def _is_point(self, points: np.ndarray) -> np.ndarray:
        # return (points**2).sum(axis=0) > 1e-8
        return np.isclose((points**2).sum(axis=0), 1.)

    def _section(self, points: np.ndarray) -> Sequence[GroupElement]:
        # radii = np.sqrt((points ** 2).sum(0))
        # assert (radii > 1e-9).all()
        assert self._is_point(points).all()
        radii = 1.

        S = points.shape[1]

        x, y, z = points

        theta = np.arccos(np.clip(z / radii, -1., 1.))
        phi = np.arctan2(y, x)

        # rot = np.asarray([0., theta, phi])
        return [self.G.element((0, theta[s], phi[s]), 'zyz') for s in range(S)]

    def basis(self, points: np.ndarray, l) -> np.ndarray:
        assert len(points.shape) == 2
        assert points.shape[0] == self.dim
        S = points.shape[1]
        
        l = self.G.get_irrep_id(l)[0]
        
        assert isinstance(l, int)

        # radii = np.sqrt((points ** 2).sum(0))
        # assert (radii > 1e-9).all()
        assert self._is_point(points).all()
        radii = 1.

        x, y, z = points
        angles = np.empty((2, S))
        angles[0, :] = np.arccos(np.clip(z / radii, -1., 1.))
        angles[1, :] = np.arctan2(y, x)

        yl = np.empty((2 * l + 1, 1, S))
        for m in range(-l, l + 1):
            yl[m + l, 0, :] = rsh(l, m, np.pi - angles[0, :], angles[1, :])
        
        # the central column of the Wigner D Matrices is proportional to the corresponding Spherical Harmonic
        # we need to correct by this proportion factor
        yl *= np.sqrt(4*np.pi / (2*l+1))
        if l % 2 == 1:
            yl *= -1
        
        return yl


class SphereO3(SpaceIsomorphism):
    
    def __init__(self):
        super(SphereO3, self).__init__(3, o3_group().homspace(('cone', -1)), o3_group().standard_representation())

    @property
    def _origin(self) -> np.ndarray:
        # The projection x_0 of the identity element `e` is the vector [0, 0, ..., 0, 1]^T
        # this choice of x_0 is convenient when we use SO(3)
        # so SO(2) subgroup is identitied with rotations around the Z axis
        return np.array([0, 0, 1]).reshape(-1, 1)

    def _is_point(self, points: np.ndarray) -> np.ndarray:
        # return (points**2).sum(axis=0) > 1e-8
        return np.isclose((points**2).sum(axis=0), 1.)

    def _section(self, points: np.ndarray) -> Sequence[GroupElement]:
        # radii = np.sqrt((points ** 2).sum(0))
        # assert (radii > 1e-9).all()
        assert self._is_point(points).all()
        radii = 1.
        
        S = points.shape[1]
        
        x, y, z = points
        
        theta = np.arccos(np.clip(z / radii, -1., 1.))
        phi = np.arctan2(y, x)
        
        # rot = np.asarray([0., theta, phi])
        return [self.G.element((0, (0, theta[s], phi[s])), 'zyz') for s in range(S)]
    
    def basis(self, points: np.ndarray, l) -> np.ndarray:
        assert len(points.shape) == 2
        assert points.shape[0] == self.dim
        S = points.shape[1]
        
        j, l = self.G.get_irrep_id(l)
        
        assert isinstance(l, int)
        assert isinstance(j, int)

        # radii = np.sqrt((points ** 2).sum(0))
        # assert (radii > 1e-9).all()
        assert self._is_point(points).all()
        radii = 1.

        if l % 2 != j:
            return np.empty((2*l+1, 0, S))
        
        x, y, z = points
        angles = np.empty((2, S))
        angles[0, :] = np.arccos(np.clip(z / radii, -1., 1.))
        angles[1, :] = np.arctan2(y, x)
        
        yl = np.empty((2 * l + 1, 1, S))
        for m in range(-l, l + 1):
            yl[m + l, 0, :] = rsh(l, m, np.pi - angles[0, :], angles[1, :])
    
        # the central column of the Wigner D Matrices is proportional to the corresponding Spherical Harmonic
        # we need to correct by this proportion factor
        yl *= np.sqrt(4 * np.pi / (2 * l + 1))
        if l % 2 == 1:
            yl *= -1

        return yl


class PointRn(SpaceIsomorphism):
    
    def __init__(self, n, G: Group):
        r"""
            A space containing a single point (the origin) in :math:`R^n`.
            It is isomorphic to :math:`G/G` for any compact group :math:`G`.
            
        """
        super(PointRn, self).__init__(n, G.homspace(G.subgroup_self_id), directsum([G.trivial_representation]*n))
        
    @property
    def _origin(self) -> np.ndarray:
        # The projection x_0 of the identity element `e` is the vector [0, 0, ..., 0, 0]^T
        return np.zeros((self.dim, 1))

    def _is_point(self, points: np.ndarray) -> np.ndarray:
        # return (points**2).sum(axis=0) > 1e-8
        return points.shape[0] == self._dim and np.isclose((points ** 2).sum(axis=0), 0.)
    
    def _section(self, points: np.ndarray) -> Sequence[GroupElement]:
        assert self._is_point(points).all()
        
        S = points.shape[1]
        return [self.G.identity for s in range(S)]
    
    def basis(self, points: np.ndarray, l) -> np.ndarray:
        assert len(points.shape) == 2
        assert points.shape[0] == self.dim
        S = points.shape[1]
        
        l = self.G.irrep(*self.G.get_irrep_id(l))
        if l == self.G.trivial_representation:
            yl = np.ones((1, 1, S))
            return yl
        else:
            return np.empty((l.size, 0, S))


# Polyhedrons

class FiniteSpaceWithIcosahedralSymmetry(SpaceIsomorphism, ABC):

    def __init__(self, sg_id: Tuple):

        assert sg_id in [(False, 2), (False, 3), (False, 5)]

        G = ico_group()
        super(FiniteSpaceWithIcosahedralSymmetry, self).__init__(3, G.homspace(sg_id), G.standard_representation)

        # retrieve the rotation axis of one of the elements of the stabilizer group to find the point in R^3 which
        # represents the coset of the identity element of G
        axis = self.X._inclusion(self.H.elements[1]).to('Q')[:3]
        self.__origin = axis.reshape(-1, 1) / np.linalg.norm(axis)

        self._points = np.concatenate([
            self.action(g) @ self._origin for g in self.G.elements
        ], axis=1)

        _, idx = np.unique(self._points.round(decimals=5), axis=1, return_index=True)
        self._points = self._points[:, idx]
        self._sections = [self.G.elements[i] for i in idx]

        assert self._points.shape[1] == self.G.order() / self.H.order(), (self._points.shape, self.G.order(), self.H.order())

    @property
    def _origin(self) -> np.ndarray:
        return self.__origin

    def _is_point(self, points: np.ndarray) -> np.ndarray:
        return np.isclose(np.expand_dims(points, 2), np.expand_dims(self._points, 1)).all(axis=0).any(1)

    def _section(self, points: np.ndarray) -> Sequence[GroupElement]:

        pairs = np.isclose(np.expand_dims(points, 2), np.expand_dims(self._points, 1)).all(axis=0)
        assert pairs.any(1).all()

        assert np.allclose(pairs.astype(np.int).sum(1), 1)

        idxs = pairs.argmax(axis=1)

        sections = [self._sections[i] for i in idxs]
        assert len(sections) == points.shape[1]
        return sections


class Icosidodecahedron(FiniteSpaceWithIcosahedralSymmetry):
    def __init__(self):
        super(Icosidodecahedron, self).__init__((False, 2))


class Icosahedron(FiniteSpaceWithIcosahedralSymmetry):
    def __init__(self):
        super(Icosahedron, self).__init__((False, 3))


class Dodecahedron(FiniteSpaceWithIcosahedralSymmetry):
    def __init__(self):
        super(Dodecahedron, self).__init__((False, 5))
