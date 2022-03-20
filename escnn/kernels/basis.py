
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Union, Tuple


class EmptyBasisException(Exception):
    def __init__(self):
        r"""
        Exception raised when a :class:`~escnn.kernels.KernelBasis` with no elements is built.
        
        """
        message = "The KernelBasis you tried to instantiate is empty (dim = 0). You should catch this exception."
        super(EmptyBasisException, self).__init__(message)
        

class KernelBasis(ABC):
    
    def __init__(self, dim: int, shape: Tuple[int, int]):
        r"""
        
        Abstract class for implementing the basis of a kernel space.
        A kernel space is the space of functions in the form:
        
        .. math::
            \mathcal{K} := \{ \kappa: X \to \mathbb{R}^{c_\text{out} \times c_\text{in}} \}
        
        where :math:`X` is the base space on which the kernel is defined.
        For instance, for planar images :math:`X = \R^2`.
        
        Args:
            dim (int): the dimensionality of the basis :math:`|\mathcal{K}|` (number of elements)
            shape (tuple): a tuple containing :math:`c_\text{out}` and :math:`c_\text{in}`
            
        Attributes:
            ~.dim (int): the dimensionality of the basis :math:`|\mathcal{K}|` (number of elements)
            ~.shape (tuple): a tuple containing :math:`c_\text{out}` and :math:`c_\text{in}`
            
        """
        assert isinstance(dim, int)
        assert isinstance(shape, tuple) and len(shape) == 2
        
        assert dim >= 0
        
        if dim == 0:
            raise EmptyBasisException()
        
        self.dim = dim
        self.shape = shape

    def __len__(self):
        return self.dim
    
    def __iter__(self):
        for i in range(self.dim):
            yield self[i]

    @abstractmethod
    def sample(self, points: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        r"""
        Sample the continuous basis elements on discrete points in ``points``.
        Optionally, store the resulting multidimentional array in ``out``.

        ``points`` must be an array of shape `(D, N)`, where `D` is the dimensionality of the (parametrization of the)
        base space while `N` is the number of points.

        Args:
            points (~numpy.ndarray): points where to evaluate the basis elements
            out (~numpy.ndarray, optional): pre-existing array to use to store the output

        Returns:
            the sampled basis

        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> dict:
        pass
    
    @abstractmethod
    def __hash__(self):
        pass
    
    @abstractmethod
    def __eq__(self, other):
        pass


class AdjointBasis(KernelBasis):
    
    def __init__(self, basis: KernelBasis, adjoint: np.ndarray):
        r"""
        
        .. todo::
            only accept orthonormal matrices or generally any invertible one?
        
        Args:
            basis (KernelBasis): a kernel basis
            adjoint (~numpy.ndarray): the matrix defining the change of basis on the base space :math:`X`

        """

        n = adjoint.shape[0]
        assert adjoint.shape == (n, n)
        
        # adjoint_inv = np.linalg.inv(adjoint)
        assert np.allclose(adjoint @ adjoint.T, np.eye(n)), 'Error! The adjunction matrix must be orthonormal'
        assert np.allclose(adjoint.T @ adjoint, np.eye(n)), 'Error! The adjunction matrix must be orthonormal'

        super(AdjointBasis, self).__init__(basis.dim, basis.shape)
        
        self.basis = basis
        self.adj = adjoint
    
    def sample(self, points: np.ndarray, out: np.ndarray = None) -> np.ndarray:
        r"""

        Sample the continuous basis elements on the discrete set of points.
        Optionally, store the resulting multidimentional array in ``out``.

        ``radii`` must be an array of shape `(n, N)`, where `N` is the number of points and `n` their
        dimensionality.

        Args:
            points (~numpy.ndarray): points where to evaluate the basis elements
            out (~numpy.ndarray, optional): pre-existing array to use to store the output

        Returns:
            the sampled basis

        """
        assert len(points.shape) == 2
        assert points.shape[0] == self.adj.shape[0]
        
        transformed_points = self.adj @ points
        return self.basis.sample(transformed_points, out)
    
    def __getitem__(self, r):
        return self.basis[r]
    
    def __eq__(self, other):
        if isinstance(other, AdjointBasis):
            return self.basis == other.basis and np.allclose(self.adj, other.adj)
        # elif self.basis == other:
        #     return np.allclose(self.adj, np.eye(self.adj.shape[0))
        else:
            return False
    
    def __hash__(self):
        return hash(self.adj) + 1000 * hash(self.basis)


class UnionBasis(KernelBasis):

    def __init__(self, bases_list: List[KernelBasis]):

        if len(bases_list) == 0:
            raise EmptyBasisException

        shape = bases_list[0].shape
        dim = 0
        for i in range(len(bases_list)):
            assert bases_list[i].shape == shape
            dim += bases_list[i].dim

        if dim == 0:
            raise EmptyBasisException

        self._bases = bases_list
        super(UnionBasis, self).__init__(dim, shape)

    def sample(self, points: np.ndarray, out: np.ndarray = None) -> np.ndarray:

        if out is None:
            out = np.empty((self.shape[0], self.shape[1], self.dim, points.shape[1]))

        p = 0
        for i in range(len(self._bases)):
            basis = self._bases[i]

            basis.sample(points, out=out[:, :, p:p+basis.dim, :])

            p += basis.dim

        return out

    def __getitem__(self, idx: int) -> dict:
        assert idx < self.dim
        p = 0
        for i in range(len(self._bases)):
            basis = self._bases[i]

            if idx < p + basis.dim:
                break

            p += basis.dim

        attr = self._bases[i][idx - p]

        attr["shape"] = self.shape
        attr['basis_id'] = i
        attr['basis_idx'] = attr['idx']
        attr['idx'] = idx

        return attr

    def __hash__(self):
        return sum(hash(self._bases[i])*(i+1)**2 for i in range(len(self._bases)))

    def __eq__(self, other):
        if not isinstance(other, UnionBasis):
            return False
        elif self.dim != other.dim or self.shape != other.shape or len(self._bases) != len(other._bases):
            return False
        else:
            for i in range(len(self._bases)):
                if self._bases[i] != other._bases[i]:
                    return False
            return True

    def __iter__(self):
        idx = 0
        for i, basis in enumerate(self._bases):
            for attr in basis:
                attr["shape"] = self.shape
                attr['basis_id'] = i
                attr['basis_idx'] = attr['idx']
                attr['idx'] = idx

                assert idx < self.dim

                yield attr

                idx += 1



