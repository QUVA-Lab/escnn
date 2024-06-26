
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from typing import Tuple, List


import os
from joblib import Memory

import escnn
from escnn.group._numerical import find_intertwiner_basis_sylvester, build_sylvester_constraint
from escnn.group._numerical import InsufficientIrrepsException

from escnn.group import __cache_path__
cache = Memory(__cache_path__, verbose=0)


########################################################################################################################
# Numeric solutions for CG coeffs
########################################################################################################################

class UnderconstrainedCGSystem(Exception):
    
    def __init__(
            self, 
            G: escnn.group.Group,
            J: Tuple,
            l: Tuple,
            j: Tuple,
            S: int,
            message: str = 'The algorithm to compute the CG coefficients failed due to an unsufficient number of samples to constraint the problem',
    ):
        self.G = G 
        self.J = J 
        self.l = l 
        self.j = j 
        self.S = S 
        super(UnderconstrainedCGSystem, self).__init__(message)


MAX_SAMPLES = 20


# cache_cg = Memory(os.path.join(os.path.dirname(__file__), '_jl_clebshgordan'), verbose=2)


# @cache_cg.cache
@cache.cache
def _clebsh_gordan_tensor(J: Tuple, l: Tuple, j: Tuple, group_class: str, **group_keys) -> np.ndarray:
    
    G = escnn.group.groups_dict[group_class]._generator(**group_keys)
    
    psi_J = G.irrep(*J)
    psi_l = G.irrep(*l)
    psi_j = G.irrep(*j)
    
    D = psi_J.size * psi_l.size * psi_j.size
    
    def build_matrices(samples):
        D_Jl = []
        D_j = []
        for g in samples:
            D_J_g = psi_J(g)
            D_l_g = psi_l(g)
            D_j_g = psi_j(g)
        
            D_Jl_g = np.kron(D_J_g, D_l_g)
            
            D_j.append(D_j_g)
            D_Jl.append(D_Jl_g)
        return D_Jl, D_j
    
    try:
        generators = G.generators
        S = len(generators)
    except ValueError:
        generators = []
        # number of samples to use to approximate the solutions
        # usually 3 are sufficient
        S = 3

    _S = S
    
    while True:
        # sometimes it might not converge, so we need to try a few times
        attepts = 5
        while True:
            try:
                samples = generators + [G.sample() for _ in range(S - len(generators))]
                if len(samples) == 0:
                    basis = np.eye(D)
                else:
                    D_Jl, D_j = build_matrices(samples)
                    basis = find_intertwiner_basis_sylvester(D_Jl, D_j)
                
            except np.linalg.LinAlgError:
                if attepts > 0:
                    attepts -= 1
                    continue
                else:
                    raise
            else:
                break
                
        # check that the solutions found are also in the kernel of the constraint matrix built with other random samples
        D_Jl, D_j = build_matrices(generators + [G.sample() for _ in range(20)])
        tensor = build_sylvester_constraint(D_Jl, D_j).todense().reshape(-1, D)
        
        if np.allclose(tensor @ basis, 0.):
            break
        elif S < MAX_SAMPLES:
            # if this not the case, try again using more samples to build the constraint matrix
            S += 1
        else:
            raise UnderconstrainedCGSystem(G, psi_J.id, psi_l.id, psi_j.id, S)
    
    if S > _S:
        print(G.name, psi_J.id, psi_l.id, psi_j.id, S)

    # the dimensionality of this basis corresponds to the multiplicity of `j` in the tensor-product `J x l`
    s = basis.shape[1]
    assert s % psi_j.sum_of_squares_constituents == 0

    jJl = s // psi_j.sum_of_squares_constituents

    CG = basis.reshape((psi_j.size, psi_J.size, psi_l.size, s)).transpose(1, 2, 3, 0)
    # CG indexed as [J, l, s, j]

    if s == 0:
        return CG

    norm = np.sqrt((CG**2).mean(2, keepdims=True).sum(1, keepdims=True).sum(0, keepdims=True))
    CG /= norm
    
    ortho = np.einsum(
        'Jlsj,Jlti,kji->stk',
        CG, CG, psi_j.endomorphism_basis()
    )
    
    ortho = (ortho**2).sum(2) > 1e-9
    assert ortho.astype(np.uint).sum() == s * psi_j.sum_of_squares_constituents, (ortho, s, jJl, psi_j.sum_of_squares_constituents)

    n, dependencies = connected_components(csgraph=csr_matrix(ortho), directed=False, return_labels=True)
    assert n * psi_j.sum_of_squares_constituents == s, (ortho, n, s, psi_j.sum_of_squares_constituents)

    mask = np.zeros((ortho.shape[0]), dtype=bool)
    for i in range(n):
        columns = np.nonzero(dependencies == i)[0]
        assert len(columns) == psi_j.sum_of_squares_constituents
        selected_column = columns[0]
        mask[selected_column] = 1

    assert mask.sum() == n

    CG = CG[..., mask, :]

    assert CG.shape[-2] == jJl

    B = CG.reshape(-1, psi_j.size * jJl)
    assert np.allclose(B.T@B, np.eye(psi_j.size * jJl))
    
    return CG


# cache_dec = Memory(os.path.join(os.path.dirname(__file__), '_jl_tensor_decomposition'), verbose=2)

# @cache_dec.cache
@cache.cache
def _find_tensor_decomposition(J: Tuple, l: Tuple, group_class: str, **group_keys) -> List[Tuple[Tuple, int]]:
    G = escnn.group.groups_dict[group_class]._generator(**group_keys)
    
    psi_J = G.irrep(*J)
    psi_l = G.irrep(*l)
    
    irreps = []
    
    size = 0
    for psi_j in G.irreps():
        CG = G._clebsh_gordan_coeff(psi_J, psi_l, psi_j)
        
        S = CG.shape[-2]
        
        if S > 0:
            irreps.append((psi_j.id, S))
        
        size += psi_j.size * S

    # check that size == psi_J.size * psi_l.size
    
    if size < psi_J.size * psi_l.size:
        from textwrap import dedent
        message = dedent(f"""
            Error! Did not find sufficient irreps to complete the decomposition of the tensor product of '{psi_J.name}' and '{psi_l.name}'.
            It is likely this happened because not sufficiently many irreps in '{G}' have been instantiated.
            Try instantiating more irreps and then repeat this call.
            The sum of the sizes of the irreps found is {size}, but the representation has size {psi_J.size * psi_l.size}.
        """)
        raise InsufficientIrrepsException(G, message)

    assert size <= psi_J.size * psi_l.size, f"""
        Error! Found too many irreps in the the decomposition of the tensor product of '{psi_J.name}' and '{psi_l.name}'.
        This should never happen!
    """

    return irreps


########################################################################################################################
# UNIT TESTS
########################################################################################################################


def check_spherical_harmonics():
    # check compatibility with SH

    from escnn.group import SO3
    from escnn.group.groups.so3group import spherical_harmonic
    
    G = SO3(2)
    
    def cart_to_sp(point):
        x, y, z = point.reshape(3)
        
        theta = np.arccos(np.clip(z, -1., 1.))
        phi = np.arctan2(y, x)
        return np.asarray([theta, phi])
    
    def sp_to_cart(angles):
        x = np.sin(angles[0]) * np.cos(angles[1])
        y = np.sin(angles[0]) * np.sin(angles[1])
        z = np.cos(angles[0])
        return np.asarray([x, y, z]).reshape(-1, 1)
    
    g = np.array([np.pi / 2, 0., 0.])
    print(G.element(g, 'zyz').to('MAT'))
    
    for l in range(9):
        
        print(l)
        n = 3
        for a in range(n):
            phi = 2 * np.pi * a / n
            for b in range(n):
                # theta = np.pi * (2 * b + 1) / (2*n)
                # theta = np.pi * (b + 1) / (n+2)
                # sample around b = 0  and b = np.pi to check singularities
                theta = np.pi * b / n
                
                alpha = np.asarray([theta, phi])
                x = sp_to_cart(alpha)
                
                Yx = spherical_harmonic(l, alpha.reshape(2, 1))
                
                for g in G.testing_elements():
                    # gx = G.irrep(1)(g) @ x
                    gx = G.standard_representation()(g) @ x
                    # gx = G.change_param(g, PARAMETRIZATION, 'MAT') @ x
                    galpha = cart_to_sp(gx)
                    
                    Ygx = spherical_harmonic(l, galpha.reshape(2, 1))
                    
                    gYx = G.irrep(l)(g) @ Yx
                    # print("###################################")
                    # print(g, gx.flatten(), galpha)
                    # print(Yx)
                    # print(Ygx)
                    # print(G.irrep(l)(g) @ Yx)
                    
                    assert np.allclose(gYx, Ygx, atol=1e-6, rtol=1e-6), g


def check_spherical_harmonics2():
    # check compatibility of CG coeffs with SH
    
    from escnn.group import SO3
    from escnn.group.groups.so3group import spherical_harmonic
    
    G = SO3(2)

    def cart_to_sp(point):
        x, y, z = point.reshape(3)
    
        theta = np.arccos(np.clip(z, -1., 1.))
        phi = np.arctan2(y, x)
        return np.asarray([theta, phi])

    def sp_to_cart(angles):
        x = np.sin(angles[0]) * np.cos(angles[1])
        y = np.sin(angles[0]) * np.sin(angles[1])
        z = np.cos(angles[0])
        return np.asarray([x, y, z]).reshape(-1, 1)
    
    for J in range(6):
        for l in range(6):
            for j in range(np.abs(J-l), J+l):
                
                CG = G._clebsh_gordan_coeff(l, J, j).reshape(2*l+1, 2*J+1, 2*j+1)
                
                print(l, J, j)
                
                n = 3
                for a in range(n):
                    phi = 2 * np.pi * a / n
                    for b in range(n):
                        theta = np.pi * b / n
                        
                        alpha = np.asarray([theta, phi])
                        x = sp_to_cart(alpha)
                        
                        Yx = spherical_harmonic(j, alpha.reshape(2, 1))
                        Yx = np.einsum('lJj,jp->lJp', CG, Yx)
                        
                        for g in G.testing_elements():
                            gx = G.standard_representation()(g) @ x
                            galpha = cart_to_sp(gx)
                            
                            Ygx = spherical_harmonic(j, galpha.reshape(2, 1))
                            Ygx = np.einsum('lJj,jp->lJp', CG, Ygx)

                            gYx = np.einsum('Ll,ljp,jJ->LJp', G.irrep(l)(g), Yx, G.irrep(J)(g).T)
                            assert np.allclose(gYx, Ygx, atol=1e-6, rtol=1e-6), g


def test_kron():
    
    for _ in range(5):
        d1 = 5
        d2 = 3
        A = np.random.randn(d1, d1)
        B = np.random.randn(d2, d2)
        
        x = np.random.randn(d1, d2)
        
        y1 = A @ x @ B
        
        y2 = np.kron(A, B.T) @ x.reshape(-1, 1)
        y2 = y2.reshape(d1, d2)
        
        assert np.allclose(y1, y2)


if __name__ == '__main__':
    # check_spherical_harmonics()
    check_spherical_harmonics2()
    # test_kron()

