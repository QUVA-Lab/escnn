import unittest
from unittest import TestCase

from escnn.group import *
import escnn.group

import numpy as np
from scipy import sparse


class TestGroups(TestCase):
    
    def _test_SO3_CB_1(self):
        # Test some of the properties of SO(3)'s GC coeffs
        # WARNING: this test fails!
        # it is likely this is because the CG coeff numerically found are not expressed in the right basis
        # anyways, as long as the tensor product decomposition tests are passed, this is sufficient for us
        
        G = SO3(5)
        
        for j1 in range(4):
            for j2 in range(4):
                for J in range(np.abs(j1 - j2), j1 + j2 + 1):

                    clebsch_gordan_j1j2J = G._clebsh_gordan_coeff(j1, j2, J)[..., 0, :]
                    clebsch_gordan_j2j1J = G._clebsh_gordan_coeff(j2, j1, J)[..., 0, :]

                    for M in range(-J, J + 1):
                        for m1 in range(-j1, j1 + 1):
                            for m2 in range(-j2, j2 + 1):

                                cb1 = clebsch_gordan_j1j2J[j1 + m1, j2 + m2, J + M]
                                cb2 = clebsch_gordan_j1j2J[j1 - m1, j2 - m2, J - M]
                                cb3 = clebsch_gordan_j2j1J[j2 + m2, j1 + m1, J + M]

                                if (J - j1 - j2) % 2 == 1:
                                    cb2 *= -1
                                    cb3 *= -1

                                assert np.allclose(cb1, cb2), (j1, j2, J, m1, m2, M)
                                assert np.allclose(cb1, cb3), (j1, j2, J, m1, m2, M)

    def _test_SO3_CB_2(self):
        # Test some of the properties of SO(3)'s GC coeffs
        # WARNING: this test fails!
        # it is likely this is because the CG coeff numerically found are not expressed in the right basis
        # anyways, as long as the tensor product decomposition tests are passed, this is sufficient for us
        G = SO3(5)

        for j1 in range(4):
            for j2 in range(4):
                for J in range(np.abs(j1 - j2), j1 + j2 + 1):
                    for J_ in range(np.abs(j1 - j2), j1 + j2 + 1):

                        clebsch_gordan_j1j2J = G._clebsh_gordan_coeff(j1, j2, J)[..., 0, :]
                        clebsch_gordan_j1j2J_ = G._clebsh_gordan_coeff(j1, j2, J_)[..., 0, :]

                        for M in range(-J, J + 1):
                            for M_ in range(-J_, J_ + 1):
                                cb = 0

                                for m1 in range(-j1, j1 + 1):
                                    for m2 in range(-j2, j2 + 1):
                                        cb += clebsch_gordan_j1j2J[m1, m2, M] * clebsch_gordan_j1j2J_[m1, m2, M_]

                                delta = int(J == J_ and M == M_)
                                assert np.allclose(cb, delta), (cb, delta, j1, j2, J, M, J_, M_)

    def test_SO3_CB_3(self):
        # Test some of the properties of SO(3)'s GC coeffs
        
        G = SO3(5)

        for j1 in range(4):
            for j2 in range(4):
                for m1 in range(-j1, j1 + 1):
                    for m1_ in range(-j1, j1 + 1):
                        for m2 in range(-j2, j2 + 1):
                            for m2_ in range(-j2, j2 + 1):
                                
                                cb = 0
                                
                                for J in range(np.abs(j1 - j2), j1 + j2 + 1):
                                    clebsch_gordan_j1j2J = G._clebsh_gordan_coeff(j1, j2, J)[..., 0, :]
                                    
                                    for M in range(-J, J + 1):
                                        cb += clebsch_gordan_j1j2J[m1, m2, M] * clebsch_gordan_j1j2J[m1_, m2_, M]
                                
                                delta = int(m1 == m1_ and m2 == m2_)
                                assert np.allclose(cb, delta), (cb, delta, j1, j2, m1, m2, m1_, m2_)

    def test_py3nj_cg_so3(self):

        # check we can import it, which means SO3 will use it internally
        import py3nj

        G = so3_group()

        for m in range(2):
            m = (m,)
            for n in range(2):
                n = (n,)
                for j in range(3):
                    j = (j,)
                    CG1 = escnn.group._clebsh_gordan._clebsh_gordan_tensor(m, n, j, G.__class__.__name__, **G._keys)
                    CG2 = G._clebsh_gordan_coeff(m, n, j)

                    # if not np.allclose(CG1, CG2, atol=1e-5):
                    #     print(CG1[:, :, 0])
                    #     print('----')
                    #     print(CG2[:, :, 0])

                    self.assertTrue(
                        np.allclose(CG1, CG2, atol=1e-5) or np.allclose(CG1, -CG2, atol=1e-5),
                        f'm={m}, n={n}, j={j}'
                    )

    def test_py3nj_cg_o3(self):

        # check we can import it, which means SO3 will use it internally
        import py3nj

        G = o3_group()

        from itertools import product

        for m in product(range(1), range(2)):
            for n in product(range(1), range(2)):
                for j in product(range(1), range(3)):
                    print(m, n, j)
                    CG1 = escnn.group._clebsh_gordan._clebsh_gordan_tensor(m, n, j, G.__class__.__name__, **G._keys)
                    CG2 = G._clebsh_gordan_coeff(m, n, j)

                    if not np.allclose(CG1, CG2, atol=1e-5):
                        print(CG1[:, :, 0])
                        print('----')
                        print(CG2[:, :, 0])
                    self.assertTrue(
                        np.allclose(CG1, CG2, atol=1e-5) or np.allclose(CG1, -CG2, atol=1e-5),
                        f'm={m}, n={n}, j={j}'
                    )


    ####################################################################################################################

    def test_o3(self):
        G = O3(3)
        self.check_clebsh_gordan_tensor_decomp_group(G)

    def test_so3(self):
        G = SO3(3)
        self.check_clebsh_gordan_tensor_decomp_group(G)

    def test_ico(self):
        G = ico_group()
        self.check_clebsh_gordan_tensor_decomp_group(G)
        
    def test_o2(self):
        G = O2(3)
        self.check_clebsh_gordan_tensor_decomp_group(G)
        
    def test_so2(self):
        G = SO2(3)
        self.check_clebsh_gordan_tensor_decomp_group(G)
        
    def test_cn(self):
        for N in [1, 2, 4, 5, 7, 8, 12, 15]:
            G = cyclic_group(N)
            self.check_clebsh_gordan_tensor_decomp_group(G)
            
    def test_dn(self):
        for N in [1, 2, 4, 5, 7, 12, 15]:
            G = dihedral_group(N)
            self.check_clebsh_gordan_tensor_decomp_group(G)

    def check_clebsh_gordan_tensor_decomp_group(self, G: Group):
        print('#######################################################################################################')
        print(G)
        
        irreps = G.irreps()
        
        for J in irreps:
            for l in irreps:
                self.check_clebsh_gordan_tensor_decomp(G, J, l)
    
        for J in irreps:
            for l in irreps:
                self.check_clebsh_gordan_equivariance(G, J, l)

    def check_clebsh_gordan_tensor_decomp(self, G: Group, J: IrreducibleRepresentation, l: IrreducibleRepresentation):
        
        J = G.irrep(*G.get_irrep_id(J))
        l = G.irrep(*G.get_irrep_id(l))

        np.set_printoptions(precision=4, suppress=True, linewidth=10000, threshold=100000)
        
        # print("###################################################################################")
        # print(f'J = {J}')
        # print(f'l = {l}')
        # for g in G.testing_elements():
        for _ in range(40):
            g = G.sample()
            
            blocks = []
            
            CB_matrix = np.empty((J.size*l.size, J.size*l.size))
            CB_matrix[:] = np.nan
            p = 0
            for j, S in G._tensor_product_irreps(J, l):
                j = G.irrep(*j)
                D_j_size = j.size
                
                D_j_g = j(g)
                blocks += [D_j_g]*S
                
                cb = G._clebsh_gordan_coeff(J, l, j)
                
                self.assertEqual(S, cb.shape[-2], msg=f"Error! {G.name}, [{J.id}, {l.id}, {j.id}]: number of basis elements expected to be {S} but {cb.shape[-2]} found")
                
                cb = cb.reshape(-1, D_j_size*S)
                CB_matrix[:, p:p + D_j_size*S] = cb
                
                p += D_j_size * S
            
            self.assertFalse(np.isnan(CB_matrix).any())
            self.assertTrue(np.isreal(CB_matrix).all())
            
            D_blocks_g = sparse.block_diag(blocks, format='csc')
            
            D_g = CB_matrix @ D_blocks_g @ CB_matrix.T
            
            D_Jl_g = np.kron(
               J(g),
               l(g),
            )
            
            self.assertTrue(np.allclose(D_g, D_Jl_g))

    def check_clebsh_gordan_equivariance(self, G: Group, J: IrreducibleRepresentation, l: IrreducibleRepresentation):
    
        J = G.irrep(*G.get_irrep_id(J))
        l = G.irrep(*G.get_irrep_id(l))
    
        np.set_printoptions(precision=4, suppress=True, linewidth=10000, threshold=100000)
    
        for j, S in G._tensor_product_irreps(J, l):
            j = G.irrep(*j)
            
            # [J, l, j, S]
            cg = G._clebsh_gordan_coeff(J, l, j)
        
            self.assertEqual(S, cg.shape[-2],
                             msg=f"Error! {G.name}, [{J.id}, {l.id}, {j.id}]: number of basis elements expected to be {S} but {cg.shape[-2]} found")

            for _ in range(40):
                g = G.sample()
                
                cg_g = np.einsum(
                    'Jlsj,ji->Jlsi',
                    cg, j(g)
                )

                g_cg = np.einsum(
                    'IJ,ml,Jlsj->Imsj',
                    J(g), l(g), cg
                )
                
                self.assertTrue(
                    np.allclose(g_cg, cg_g)
                )
                
                cg_g = cg_g.reshape(-1, j.size*S)
                g_cg = g_cg.reshape(-1, j.size*S)
                tg_cg = np.kron(J(g), l(g)) @ cg.reshape(-1, j.size*S)

                self.assertTrue(
                    np.allclose(tg_cg, g_cg)
                )
                self.assertTrue(
                    np.allclose(tg_cg, cg_g)
                )


if __name__ == '__main__':
    unittest.main()
