import numpy as np

from .steerable_basis import IrrepBasis
from .spaces import SpaceIsomorphism

from escnn.group import *

from typing import Union, Tuple, Dict, Iterable, List
from collections import defaultdict
from itertools import chain

__all__ = [
    "WignerEckartBasis",
    "RestrictedWignerEckartBasis"
]


class WignerEckartBasis(IrrepBasis):
    
    def __init__(self,
                 X: SpaceIsomorphism,
                 in_irrep: Union[str, IrreducibleRepresentation, int],
                 out_irrep: Union[str, IrreducibleRepresentation, int],
                 harmonics: List[Tuple] = None,
                 ):
        r"""
        
        Solves the kernel constraint for a pair of input and output :math:`G`-irreps over an orbit :math:`X` of
        :math:`G` by using the Wigner-Eckart theorem described in
        `A Wigner-Eckart Theorem for Group Equivariant Convolution Kernels <https://arxiv.org/abs/2010.10952>`_ .
        
        Note that the orbit :math:`X` is isomorphic to an homogeneous space :math:`G/H`, for a particular stabilizer
        subgroup :math:`H < G`.
        Hence, the input `X` is an instance of :class:`~escnn.kernels.SpaceIsomorphism` which precisely implements
        such isomorphism.
        
        A basis for functions over :math:`X\cong G/H` is automatically generated via :class:`escnn.group.HomSpace.basis`,
        and can be band-limited by selecting only the subspaces transforming according to the :math:`G`-irreps in the
        input list `harmonics`.
        
        Args:
            X (SpaceIsomorphism): the orbit of :math:`G`
            in_repr (IrreducibleRepresentation): the input irrep
            out_repr (IrreducibleRepresentation): the output irrep
            harmonics (list, optional): the list of :math:`G` irreps to consider for band-limiting
            
        """
        
        group = X.G

        in_irrep = group.irrep(*group.get_irrep_id(in_irrep))
        out_irrep = group.irrep(*group.get_irrep_id(out_irrep))
        
        assert in_irrep.group == X.G
        assert out_irrep.group == X.G
        assert in_irrep.group == out_irrep.group
        
        self.X = X
        
        self.m = in_irrep.id
        self.n = out_irrep.id
        
        if harmonics is not None:
            _js = []
            for j in harmonics:
                if self.X.dimension_basis(j)[1] == 0:
                    # if the G irrep j has multiplicity 0 in L^2(X), there is no harmonic associated with it
                    # so we can skip it
                    continue
                    
                jJl = group._clebsh_gordan_coeff(self.n, self.m, j).shape[-2]
                if jJl > 0 and self.X.dimension_basis(j)[1] > 0:
                    _js.append((j, jJl))
        else:
            _js = group._tensor_product_irreps(self.m, self.n)
    
            _js = [
                (j, jJl)
                for j, jJl in _js
                if self.X.dimension_basis(j)[1] > 0
            ]

        self._coeff = [
            np.einsum(
                'mnsM,kNM->mnksN',
                group._clebsh_gordan_coeff(self.n, self.m, j), group.irrep(*j).endomorphism_basis()
            ) for j, jJl in _js
        ]
        
        for b, (j, jJl) in enumerate(_js):
            coeff = self._coeff[b]
            assert jJl == coeff.shape[-2]

        dim = 0
        self._dim_harmonics = {}
        self._jJl = {}
        for j, jJl in _js:
            self._dim_harmonics[j] = self.X.dimension_basis(j)[1] * jJl * group.irrep(*j).sum_of_squares_constituents
            self._jJl[j] = jJl
            dim += self._dim_harmonics[j]
        
        js = [j for j, _ in _js]
        
        super(WignerEckartBasis, self).__init__(X, in_irrep, out_irrep, js, dim)

    def sample_harmonics(self, points: Dict[Tuple, np.ndarray], out: Dict[Tuple, np.ndarray] = None) -> Dict[Tuple, np.ndarray]:
        if out is None:
            out = {
                j : np.empty((self.shape[0], self.shape[1], self.dim_harmonic(j), points[j].shape[-1]))
                for j in self.js
            }
    
        for j in self.js:
            if j in out:
                assert out[j].shape == (self.shape[0], self.shape[1], self.dim_harmonic(j), points[j].shape[-1])

        for b, j in enumerate(self.js):
            coeff = self._coeff[b]
            
            jJl = coeff.shape[-2]
        
            Ys = points[j]
            np.einsum(
                'Nnksm,miS->NnksiS',
                coeff, Ys,
                out=out[j].reshape((
                    self.out_irrep.size, self.in_irrep.size, self.X.G.irrep(*j).sum_of_squares_constituents, jJl,
                    Ys.shape[1], Ys.shape[2]
                ))
            )
        
        return out

    def dim_harmonic(self, j: Tuple) -> int:
        if j in self._dim_harmonics:
            return self._dim_harmonics[j]
        else:
            return 0

    def attrs_j_iter(self, j: Tuple) -> Iterable:
        
        if self.dim_harmonic(j) == 0:
            return
        
        idx = self._start_index[j]
        
        j_attr = {
            'irrep:'+k: v
            for k, v in self.group.irrep(*j).attributes.items()
        }
        
        dim = self.X.dimension_basis(j)[1]
        for k in range(self.X.G.irrep(*j).sum_of_squares_constituents):
            for s in range(self._jJl[j]):
                for i in range(dim):
                    attr = j_attr.copy()
                    attr["idx"] = idx
                    attr["j"] = j
                    attr["i"] = i
                    attr["s"] = s
                    attr["k"] = k
                    idx += 1
                
                    yield attr

    def attrs_j(self, j: Tuple, idx) -> Dict:
        assert 0 <= idx < self.dim_harmonic(j)
        
        full_idx = self._start_index[j] + idx

        dim = self.X.dimension_basis(j)[1]

        attr = {
            'irrep:'+k: v
            for k, v in self.group.irrep(*j).attributes.items()
        }
        
        attr["idx"] = full_idx
        attr["j"] = j
        attr["i"] = idx % dim
        attr["s"] = (idx // dim) % self._jJl[j]
        attr["k"] = idx // (dim * self._jJl[j])
        
        return attr

    def __getitem__(self, idx):
        assert 0 <= idx < self.dim
    
        i = idx
        for j in self.js:
            dim = self.dim_harmonic(j)
            if i < dim:
                break
            else:
                i -= dim
        
        return self.attrs_j(j, i)
        
    def __iter__(self):
        return chain(self.attrs_j_iter(j) for j in self.js)

    def __eq__(self, other):
        if not isinstance(other, WignerEckartBasis):
            return False
        elif self.X != other.X or self.in_irrep != other.in_irrep or self.out_irrep != other.out_irrep:
            # TODO check isomorphism too!
            return False
        elif len(self.js) != len(other.js):
            return False
        else:
            for b, (j, i) in enumerate(zip(self.js, other.js)):
                if j!=i or not np.allclose(self._coeff[b], other._coeff[b]):
                    return False
            return True

    def __hash__(self):
        return hash(self.X) + hash(self.in_irrep) + hash(self.out_irrep) + hash(tuple(self.js))

    _cached_instances = {}
    
    @classmethod
    def _generator(cls,
                   X: SpaceIsomorphism,
                   psi_in: Union[IrreducibleRepresentation, Tuple],
                   psi_out: Union[IrreducibleRepresentation, Tuple],
                   harmonics: List[Tuple] = None,
                   **kwargs) -> 'IrrepBasis':
    
        assert len(kwargs) == 0
        
        psi_in = X.G.irrep(*X.G.get_irrep_id(psi_in))
        psi_out = X.G.irrep(*X.G.get_irrep_id(psi_out))
        
        if harmonics is not None:
            _harmonics = tuple(sorted(harmonics))
        else:
            _harmonics = None
        key = (X, psi_in.id, psi_out.id, _harmonics)
        
        if key not in cls._cached_instances:
            cls._cached_instances[key] = WignerEckartBasis(X, in_irrep=psi_in, out_irrep=psi_out, harmonics=harmonics)
        return cls._cached_instances[key]


class RestrictedWignerEckartBasis(IrrepBasis):
    
    def __init__(self,
                 X: SpaceIsomorphism,
                 sg_id: Tuple,
                 in_irrep: Union[str, IrreducibleRepresentation, int],
                 out_irrep: Union[str, IrreducibleRepresentation, int],
                 harmonics: Iterable[Tuple],
                 ):
        r"""

        Solves the kernel constraint for a pair of input and output :math:`G`-irreps over an orbit :math:`X` of a larger
        group :math:`G' > G` by using the generalized Wigner-Eckart theorem described in
        `A Program to Build E(N)-Equivariant Steerable CNNs  <https://openreview.net/forum?id=WE4qe9xlnQw>`_
        (Theorem 2.2).

        Note that the orbit :math:`X` is isomorphic to an homogeneous space :math:`G'/H`, for a particular stabilizer
        subgroup :math:`H < G'`.
        Hence, the input `X` is an instance of :class:`~escnn.kernels.SpaceIsomorphism` which precisely implements
        such isomorphism.

        A basis :math:`\mathcal{B} = \{Y_{j'i'}\}` for functions over :math:`X\cong G'/H` is automatically generated via
        :class:`escnn.group.HomSpace.basis`, and can be band-limited by selecting only the subspaces transforming
        according to the :math:`G'`-irreps in the input list `harmonics`.
        
        The equivariance group :math:`G < G'` is identified by the input id ``sg_id``.

        Args:
            X (SpaceIsomorphism): the orbit of :math:`G'`
            sg_id (tuple): id of :math:`G` as a subgroup of :math:`G'`.
            in_repr (IrreducibleRepresentation): the input `G`-irrep
            out_repr (IrreducibleRepresentation): the output `G`-irrep
            harmonics (list, optional): the list of :math:`G'` irreps to consider for band-limiting

        """

        # the larger group G'
        _G = X.G

        # the base space is a homogeneous space for the larger group G'
        self.X = X

        # the smaller equivariance group G
        G, inclusion, restriction = _G.subgroup(sg_id)
        
        self.group = G
        self.sg_id = sg_id

        in_irrep = G.irrep(*G.get_irrep_id(in_irrep))
        out_irrep = G.irrep(*G.get_irrep_id(out_irrep))

        assert in_irrep.group == G
        assert out_irrep.group == G
        assert in_irrep.group == out_irrep.group
        
        self.m = in_irrep.id
        self.n = out_irrep.id

        # irreps of G in the decomposition of the tensor product of in_irrep and out_irrep
        _js_G = [
            j for j, _ in
            G._tensor_product_irreps(self.m, self.n)
        ]
        
        _js = set()
        _js_restriction = defaultdict(list)
        
        assert harmonics is not None, \
            f'A finite list of harmonics to consider is required when using ' \
            f'`RestrictedWignerEckartbasis` to avoid building infinite dimensional bases'
        
        # for each harmonic j' of X considered
        for _j in harmonics:
            if self.X.dimension_basis(_j)[1] == 0:
                # if the G' irrep j' has multiplicity 0 in L^2(X), there is no harmonic associated with it
                # so we can skip it
                continue
                
            # restrict the corresponding G' irrep j' to G
            _j_G = _G.irrep(*_j).restrict(sg_id)
            
            # for each G irrep j in the tensor product decomposition of in_irrep and out_irrep
            for j in _js_G:
                # irrep-decomposition coefficients of j in j'
                id_coeff = []
                p = 0
                # for each G irrep i in the restriction of j' to G
                for i in _j_G.irreps:
                    size = G.irrep(*i).size
                    # if the restricted irrep contains one of the irreps in the tensor product
                    if i == j:
                        id_coeff.append(
                            _j_G.change_of_basis_inv[p:p+size, :]
                        )
                    
                    p += size
                
                # if the G irrep j appears in the restriction of the harmonic j',
                # store its irrep-decomposition coefficients
                if len(id_coeff) > 0:
                    id_coeff = np.stack(id_coeff, axis=-1)
                    _js.add(_j)
                    _js_restriction[_j].append((j, id_coeff))

        _js = sorted(list(_js))
        
        self._coeff = {}
        self._js_restriction = {}
        for _j in _js:
            Y_size = _G.irrep(*_j).size
            coeff = [
                np.einsum(
                    'nmsM,kNM,NYt->nmkstY',
                    G._clebsh_gordan_coeff(self.n, self.m, j), G.irrep(*j).endomorphism_basis(), id_coeff
                ).reshape((out_irrep.size, in_irrep.size, -1, Y_size))
                for j, id_coeff in _js_restriction[_j]
            ]
            
            self._coeff[_j] = np.concatenate(coeff, axis=2)
            self._js_restriction[_j] = [(j, id_coeff.shape[2]) for j, id_coeff in _js_restriction[_j]]

        dim = sum(self.dim_harmonic(_j) for _j in _js)

        super(RestrictedWignerEckartBasis, self).__init__(X, in_irrep, out_irrep, _js, dim)

    def sample_harmonics(self, points: Dict[Tuple, np.ndarray], out: Dict[Tuple, np.ndarray] = None) -> Dict[
        Tuple, np.ndarray]:
        if out is None:
            out = {
                j: np.empty((self.shape[0], self.shape[1], self.dim_harmonic(j), points[j].shape[-1]))
                for j in self.js
            }
    
        for j in self.js:
            if j in out:
                assert out[j].shape == (self.shape[0], self.shape[1], self.dim_harmonic(j), points[j].shape[-1])
    
        for j in self.js:
            coeff = self._coeff[j]
        
            Ys = points[j]
            np.einsum(
                'NnDm,miS->NnDiS',
                coeff, Ys,
                out=out[j].reshape((
                    self.out_irrep.size, self.in_irrep.size, coeff.shape[2], Ys.shape[1], Ys.shape[2]
                ))
            )
    
        return out

    def dim_harmonic(self, j: Tuple) -> int:
        if j in self._coeff:
            return self.X.dimension_basis(j)[1] * self._coeff[j].shape[2]
        else:
            return 0

    def attrs_j_iter(self, j: Tuple) -> Iterable:
        
        if self.dim_harmonic(j) == 0:
            return
        
        idx = self._start_index[j]

        dim = self.X.dimension_basis(j)[1]
        
        j_attr = {
            'irrep:'+k: v
            for k, v in self.X.G.irrep(*j).attributes.items()
        }

        for _j, _jj in self._js_restriction[j]:
            _jJl = self.group._clebsh_gordan_coeff(self.n, self.m, _j).shape[2]
            K = self.group.irrep(*_j).sum_of_squares_constituents
            
            for k in range(K):
                for s in range(_jJl):
                    for t in range(_jj):
                        for i in range(dim):
                            attr = j_attr.copy()
                            attr["idx"] = idx
                            attr["j"] = j
                            attr["_j"] = _j
                            attr["i"] = i
                            attr["t"] = t
                            attr["s"] = s
                            attr["k"] = k

                            assert idx < self.dim

                            idx += 1
                        
                            yield attr

    def attrs_j(self, j: Tuple, idx) -> Dict:
        assert 0 <= idx < self.dim_harmonic(j)
        
        full_idx = self._start_index[j] + idx

        dim = self.X.dimension_basis(j)[1]
    
        for _j, _jj in self._js_restriction[j]:
            _jJl = self.group._clebsh_gordan_coeff(self.n, self.m, _j).shape[2]
            K = self.group.irrep(*_j).sum_of_squares_constituents
            
            d = _jj * _jJl * K * dim
            
            if idx >= d:
                idx -= d
            else:
                break

        attr = {
            'irrep:'+k: v
            for k, v in self.X.G.irrep(*j).attributes.items()
        }
        attr["idx"] = full_idx
        attr["j"] = j
        attr["_j"] = _j
        attr["i"] = idx % dim
        attr["t"] = (idx // dim) % _jj
        attr["s"] = (idx // (dim * _jj)) % _jJl
        attr["k"] = idx // (dim * _jj * _jJl)
        return attr

    def __getitem__(self, idx):
        assert 0 <= idx < self.dim
    
        i = idx
        for j in self.js:
            dim = self.dim_harmonic(j)
            if i < dim:
                break
            else:
                i -= dim
    
        return self.attrs_j(j, i)

    def __iter__(self):
        return chain(self.attrs_j_iter(j) for j in self.js)

    def __eq__(self, other):
        if not isinstance(other, RestrictedWignerEckartBasis):
            return False
        elif self.X != other.X or self.sg_id != other.sg_id or self.in_irrep != other.in_irrep or self.out_irrep != other.out_irrep:
            # TODO check isomorphism too!
            return False
        elif len(self.js) != len(other.js):
            return False
        else:
            for (j, i) in zip(self.js, other.js):
                if j!=i or not np.allclose(self._coeff[j], other._coeff[i]):
                    return False
            return True

    def __hash__(self):
        return hash(self.X) + hash(self.sg_id) + hash(self.in_irrep) + hash(self.out_irrep) + hash(tuple(self.js))

    _cached_instances = {}
    
    @classmethod
    def _generator(cls,
                   X: SpaceIsomorphism,
                   psi_in: Union[IrreducibleRepresentation, Tuple],
                   psi_out: Union[IrreducibleRepresentation, Tuple],
                   harmonics: List[Tuple] = None,
                   **kwargs) -> 'IrrepBasis':
    
        assert len(kwargs) == 1
        assert 'sg_id' in kwargs

        G, _, _ = X.G.subgroup(kwargs['sg_id'])
        psi_in = G.irrep(*G.get_irrep_id(psi_in))
        psi_out = G.irrep(*G.get_irrep_id(psi_out))

        if harmonics is not None:
            _harmonics = tuple(sorted(harmonics))
        else:
            _harmonics = None
            
        key = (X, psi_in.id, psi_out.id, _harmonics, kwargs['sg_id'])

        if key not in cls._cached_instances:
            cls._cached_instances[key] = RestrictedWignerEckartBasis(
                X,
                sg_id=kwargs['sg_id'],
                in_irrep=psi_in,
                out_irrep=psi_out,
                harmonics=harmonics
            )
        
        return cls._cached_instances[key]

