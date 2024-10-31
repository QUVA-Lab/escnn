import numpy as np

from .steerable_basis import IrrepBasis
from .steerable_filters_basis import SteerableFiltersBasis

from escnn.group import *
from escnn.utils import unique_ever_seen

import torch

from typing import Union, Tuple, Dict, Iterable, List
from collections import defaultdict
from itertools import chain

__all__ = [
    "WignerEckartBasis",
    "RestrictedWignerEckartBasis"
]


class WignerEckartBasis(IrrepBasis):
    
    def __init__(self,
                 basis: SteerableFiltersBasis,
                 in_irrep: Union[str, IrreducibleRepresentation, int],
                 out_irrep: Union[str, IrreducibleRepresentation, int],
        ):
        r"""
        
        Solves the kernel constraint for a pair of input and output :math:`G`-irreps by using the Wigner-Eckart theorem
        described in Theorem 2.1 of
        `A Program to Build E(N)-Equivariant Steerable CNNs <https://openreview.net/forum?id=WE4qe9xlnQw>`_
        (see also
        `A Wigner-Eckart Theorem for Group Equivariant Convolution Kernels <https://arxiv.org/abs/2010.10952>`_
        ).
        The method relies on a :math:`G`-Steerable basis of scalar functions over the base space.

        Args:
            basis (SteerableFiltersBasis): a `G`-steerable basis for scalar functions over the base space
            in_repr (IrreducibleRepresentation): the input irrep
            out_repr (IrreducibleRepresentation): the output irrep

        """
        
        group = basis.group

        in_irrep = group.irrep(*group.get_irrep_id(in_irrep))
        out_irrep = group.irrep(*group.get_irrep_id(out_irrep))
        
        assert in_irrep.group == group
        assert out_irrep.group == group
        assert in_irrep.group == out_irrep.group

        self.m = in_irrep.id
        self.n = out_irrep.id
        
        _js = group._tensor_product_irreps(self.m, self.n)

        _js = [
            (j, jJl)
            for j, jJl in _js
            if basis.multiplicity(j) > 0
        ]

        dim = 0
        self._dim_harmonics = {}
        self._jJl = {}
        for j, jJl in _js:
            self._dim_harmonics[j] = basis.multiplicity(j) * jJl * group.irrep(*j).sum_of_squares_constituents
            self._jJl[j] = jJl
            dim += self._dim_harmonics[j]

        super(WignerEckartBasis, self).__init__(basis, in_irrep, out_irrep, dim, harmonics=[_j for _j, _ in _js])

        # SteerableFiltersBasis: a `G`-steerable basis for scalar functions over the base space
        self.basis = basis

        _coeff = [
            torch.einsum(
                # 'mnsi,koi->mnkso',
                'mnsi,koi->ksmno',
                torch.tensor(group._clebsh_gordan_coeff(self.n, self.m, j), dtype=torch.float32),
                torch.tensor(group.irrep(*j).endomorphism_basis(), dtype=torch.float32),
            ) for j in self.js
        ]
        
        for b, j in enumerate(self.js):
            coeff = _coeff[b]
            assert self._jJl[j] == coeff.shape[1]
            self.register_buffer(f'coeff_{b}', coeff)

    def coeff(self, idx: int) -> torch.Tensor:
        return getattr(self, f'coeff_{idx}')

    def sample_harmonics(self, points: Dict[Tuple, torch.Tensor], out: Dict[Tuple, torch.Tensor] = None) -> Dict[Tuple, torch.Tensor]:
        if out is None:
            out = {
                j: torch.zeros(
                    (points[j].shape[0], self.dim_harmonic(j), self.shape[0], self.shape[1]),
                    device=points[j].device, dtype=points[j].dtype
                )
                for j in self.js
            }
    
        for j in self.js:
            if j in out:
                assert out[j].shape == (points[j].shape[0], self.dim_harmonic(j), self.shape[0], self.shape[1]), (
                    out[j].shape, points[j].shape[0], self.dim_harmonic(j), self.shape[0], self.shape[1]
                )

        for b, j in enumerate(self.js):
            if j not in out:
                continue
            coeff = self.coeff(b)
            
            jJl = coeff.shape[1]
        
            Ys = points[j]

            out[j].view((
                Ys.shape[0],
                self.group.irrep(*j).sum_of_squares_constituents, jJl,
                Ys.shape[1],
                self.out_irrep.size, self.in_irrep.size,
            ))[:] = torch.einsum(
                # 'Nnksm,miS->NnksiS',
                'kspnm,qim->qksipn',
                coeff, Ys,
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
        steerable_basis_j_attr = list(self.basis.steerable_attrs_j_iter(j))
        
        for k in range(self.group.irrep(*j).sum_of_squares_constituents):
            for s in range(self._jJl[j]):
                for i, attr_i in enumerate(steerable_basis_j_attr):
                    attr = j_attr.copy()
                    attr.update(**attr_i)
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

        dim = self.basis.multiplicity(j)

        attr = {
            'irrep:'+k: v
            for k, v in self.group.irrep(*j).attributes.items()
        }

        i = idx % dim
        attr_i = self.basis.steerable_attrs_j(j, i)

        attr.update(**attr_i)
        attr["idx"] = full_idx
        attr["j"] = j
        attr["i"] = i
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
        for j in self.js:
            for attr in self.attrs_j_iter(j):
                yield attr
        # return chain(self.attrs_j_iter(j) for j in self.js)

    def __eq__(self, other):
        if not isinstance(other, WignerEckartBasis):
            return False
        elif self.basis != other.basis or self.in_irrep != other.in_irrep or self.out_irrep != other.out_irrep:
            # TODO check isomorphism too!
            return False
        elif len(self.js) != len(other.js):
            return False
        else:
            for b, (j, i) in enumerate(zip(self.js, other.js)):
                if j!=i or not torch.allclose(self.coeff(b), other.coeff(b)):
                    return False
            return True

    def __hash__(self):
        return hash(self.basis) + hash(self.in_irrep) + hash(self.out_irrep) + hash(tuple(self.js))

    _cached_instances = {}
    
    @classmethod
    def _generator(cls,
                   basis: SteerableFiltersBasis,
                   psi_in: Union[IrreducibleRepresentation, Tuple],
                   psi_out: Union[IrreducibleRepresentation, Tuple],
                   **kwargs) -> 'IrrepBasis':
    
        assert len(kwargs) == 0
        
        psi_in = basis.group.irrep(*basis.group.get_irrep_id(psi_in))
        psi_out = basis.group.irrep(*basis.group.get_irrep_id(psi_out))
        
        key = (basis, psi_in.id, psi_out.id)
        
        if key not in cls._cached_instances:
            cls._cached_instances[key] = WignerEckartBasis(basis, in_irrep=psi_in, out_irrep=psi_out)
        return cls._cached_instances[key]


class RestrictedWignerEckartBasis(IrrepBasis):
    
    def __init__(self,
                 basis: SteerableFiltersBasis,
                 sg_id: Tuple,
                 in_irrep: Union[str, IrreducibleRepresentation, int],
                 out_irrep: Union[str, IrreducibleRepresentation, int],
                 ):
        r"""

        Solves the kernel constraint for a pair of input and output :math:`G`-irreps by using the Wigner-Eckart theorem
        described in Theorem 2.1 of
        `A Program to Build E(N)-Equivariant Steerable CNNs  <https://openreview.net/forum?id=WE4qe9xlnQw>`_.
        This method implicitly constructs the required :math:`G`-steerable basis for scalar functions on the base space
        from a :math:`G'`-steerable basis, with :math:`G' > G` a larger group, according to Equation 5 from the same
        paper.

        The equivariance group :math:`G < G'` is identified by the input id ``sg_id``.

        .. warning::
            Note that the group :math:`G'` associated with ``basis`` is generally not the same as the group :math:`G`
            associated with ``in_irrep`` and ``out_irrep`` and which the resulting kernel basis is equivariant to.

        Args:
            basis (SteerableFiltersBasis): :math:`G'`-steerable basis for scalar filters
            sg_id (tuple): id of :math:`G` as a subgroup of :math:`G'`.
            in_repr (IrreducibleRepresentation): the input `G`-irrep
            out_repr (IrreducibleRepresentation): the output `G`-irrep

        """

        # the larger group G'
        _G = basis.group

        G = _G.subgroup(sg_id)[0]
        # Group: the smaller equivariance group G
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
        
        # for each harmonic j' to consider
        for _j in unique_ever_seen(_j for _j, _ in basis.js):
            if basis.multiplicity(_j) == 0:
                continue
                
            # restrict the corresponding G' irrep j' to G
            _j_G = _G.irrep(*_j).restrict(sg_id)
            
            # for each G-irrep j in the tensor product decomposition of in_irrep and out_irrep
            for j in _js_G:
                # irrep-decomposition coefficients of j in j'
                id_coeff = []
                p = 0
                # for each G-irrep i in the restriction of j' to G
                for i in _j_G.irreps:
                    size = G.irrep(*i).size
                    # if the restricted irrep contains one of the irreps in the tensor product
                    if i == j:
                        id_coeff.append(
                            _j_G.change_of_basis_inv[p:p+size, :]
                        )
                    
                    p += size
                
                # if the G irrep j appears in the restriction of the G'-irrep j',
                # store its irrep-decomposition coefficients
                if len(id_coeff) > 0:
                    id_coeff = np.stack(id_coeff, axis=-1)
                    _js.add(_j)
                    _js_restriction[_j].append((j, id_coeff))

        _js = sorted(list(_js))

        self._js_restriction = {}
        self._dim_harmonics = {}
        _coeffs = {}
        dim = 0
        for _j in _js:
            Y_size = _G.irrep(*_j).size
            coeff = [
                torch.einsum(
                    # 'nmsi,kji,jyt->nmksty',
                    'nmsi,kji,jyt->kstnmy',
                    torch.tensor(G._clebsh_gordan_coeff(self.n, self.m, j), dtype=torch.float32),
                    torch.tensor(G.irrep(*j).endomorphism_basis(), dtype=torch.float32),
                    torch.tensor(id_coeff, dtype=torch.float32),
                ).reshape((-1, out_irrep.size, in_irrep.size, Y_size))
                for j, id_coeff in _js_restriction[_j]
            ]
            
            _coeffs[_j] = torch.cat(coeff, dim=0)
            self._js_restriction[_j] = [(j, id_coeff.shape[2]) for j, id_coeff in _js_restriction[_j]]
            self._dim_harmonics[_j] = _coeffs[_j].shape[0]
            dim += self._dim_harmonics[_j] * basis.multiplicity(_j)

        super(RestrictedWignerEckartBasis, self).__init__(basis, in_irrep, out_irrep, dim, harmonics=_js)

        # SteerableFiltersBasis: a `G'`-steerable basis for scalar functions over the base space, for the larger
        # group `G' > G`
        self.basis = basis

        for b, _j in enumerate(self.js):
            self.register_buffer(f'coeff_{b}', _coeffs[_j])

    def coeff(self, idx: int) -> torch.Tensor:
        return getattr(self, f'coeff_{idx}')

    def sample_harmonics(self, points: Dict[Tuple, torch.Tensor], out: Dict[Tuple, torch.Tensor] = None) -> Dict[
        Tuple, torch.Tensor]:
        if out is None:
            out = {
                j: torch.zeros(
                    (points[j].shape[0], self.dim_harmonic(j), self.shape[0], self.shape[1]),
                    device=points[j].device, dtype=points[j].dtype
                )
                for j in self.js
            }
    
        for j in self.js:
            if j in out:
                assert out[j].shape == (points[j].shape[0], self.dim_harmonic(j), self.shape[0], self.shape[1])
    
        for b, j in enumerate(self.js):
            if j not in out:
                continue
            coeff = self.coeff(b)
        
            Ys = points[j]
            out[j].view((
                Ys.shape[0], coeff.shape[0], Ys.shape[1],
                self.out_irrep.size, self.in_irrep.size,
            ))[:] = torch.einsum(
                'dpnm,sim->sdipn',
                coeff, Ys,
            )
    
        return out

    def dim_harmonic(self, j: Tuple) -> int:
        if j in self._dim_harmonics:
            return self.basis.multiplicity(j) * self._dim_harmonics[j]
        else:
            return 0

    def attrs_j_iter(self, j: Tuple) -> Iterable:
        
        if self.dim_harmonic(j) == 0:
            return
        
        idx = self._start_index[j]

        steerable_basis_j_attr = list(self.basis.steerable_attrs_j_iter(j))

        j_attr = {
            'irrep:'+k: v
            for k, v in self.basis.group.irrep(*j).attributes.items()
        }

        count = 0
        for _j, _jj in self._js_restriction[j]:
            _jJl = self.group._clebsh_gordan_coeff(self.n, self.m, _j).shape[2]
            K = self.group.irrep(*_j).sum_of_squares_constituents
            
            for k in range(K):
                for s in range(_jJl):
                    for t in range(_jj):
                        for i, attr_i in enumerate(steerable_basis_j_attr):
                            attr = j_attr.copy()
                            attr.update(**attr_i)
                            attr["idx"] = idx
                            attr["j"] = j
                            attr["_j"] = _j
                            attr["i"] = i
                            attr["t"] = t
                            attr["s"] = s
                            attr["k"] = k

                            assert idx < self.dim
                            assert count < self.dim_harmonic(j), (count, self.dim_harmonic(j))

                            idx += 1
                            count += 1
                        
                            yield attr

        assert count == self.dim_harmonic(j), (count, self.dim_harmonic(j))

    def attrs_j(self, j: Tuple, idx) -> Dict:
        assert 0 <= idx < self.dim_harmonic(j)
        
        full_idx = self._start_index[j] + idx

        dim = self.basis.multiplicity(j)

        for _j, _jj in self._js_restriction[j]:
            _jJl = self.group._clebsh_gordan_coeff(self.n, self.m, _j).shape[2]
            K = self.group.irrep(*_j).sum_of_squares_constituents
            
            d = _jj * _jJl * K * dim
            
            if idx >= d:
                idx -= d
            else:
                break

        i = idx % dim
        attr_i = self.basis.steerable_attrs_j(j, i)

        attr = {
            'irrep:'+k: v
            for k, v in self.basis.group.irrep(*j).attributes.items()
        }
        attr.update(**attr_i)
        attr["idx"] = full_idx
        attr["j"] = j
        attr["_j"] = _j
        attr["i"] = i
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

    def __iter__(self) -> Iterable:
        for j in self.js:
            for attr in self.attrs_j_iter(j):
                yield attr
        # return chain(self.attrs_j_iter(j) for j in self.js)

    def __eq__(self, other):
        if not isinstance(other, RestrictedWignerEckartBasis):
            return False
        elif self.basis != other.basis or self.sg_id != other.sg_id or self.in_irrep != other.in_irrep or self.out_irrep != other.out_irrep:
            # TODO check isomorphism too!
            return False
        elif len(self.js) != len(other.js):
            return False
        else:
            for b, (j, i) in enumerate(zip(self.js, other.js)):
                if j!=i or not torch.allclose(self.coeff(b), other.coeff(b)):
                    return False
            return True

    def __hash__(self):
        return hash(self.basis) + hash(self.sg_id) + hash(self.in_irrep) + hash(self.out_irrep) + hash(tuple(self.js))

    _cached_instances = {}
    
    @classmethod
    def _generator(cls,
                   basis: SteerableFiltersBasis,
                   psi_in: Union[IrreducibleRepresentation, Tuple],
                   psi_out: Union[IrreducibleRepresentation, Tuple],
                   **kwargs) -> 'IrrepBasis':
    
        assert len(kwargs) == 1
        assert 'sg_id' in kwargs

        G, _, _ = basis.group.subgroup(kwargs['sg_id'])
        psi_in = G.irrep(*G.get_irrep_id(psi_in))
        psi_out = G.irrep(*G.get_irrep_id(psi_out))

        key = (
            basis, psi_in.id, psi_out.id,
            kwargs['sg_id']
        )

        if key not in cls._cached_instances:
            cls._cached_instances[key] = RestrictedWignerEckartBasis(
                basis,
                sg_id=kwargs['sg_id'],
                in_irrep=psi_in,
                out_irrep=psi_out,
            )
        
        return cls._cached_instances[key]

