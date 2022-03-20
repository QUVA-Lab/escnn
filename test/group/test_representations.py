import unittest
from unittest import TestCase

from escnn.group import *
import numpy as np


class TestRepresentation(TestCase):
    
    # test endomorphism bases are equivariant
    
    def test_endom_irreps_cyclic_even(self):
        self.check_irreps_endom(cyclic_group(2))
        self.check_irreps_endom(cyclic_group(4))
        self.check_irreps_endom(cyclic_group(16))
        
    def test_endom_irreps_cyclic_odd(self):
        self.check_irreps_endom(cyclic_group(1))
        self.check_irreps_endom(cyclic_group(3))
        self.check_irreps_endom(cyclic_group(9))
        self.check_irreps_endom(cyclic_group(13))

    def test_endom_irreps_dihedral_even(self):
        self.check_irreps_endom(dihedral_group(2))
        self.check_irreps_endom(dihedral_group(4))
        self.check_irreps_endom(dihedral_group(16))

    def test_endom_irreps_dihedral_odd(self):
        self.check_irreps_endom(dihedral_group(1))
        self.check_irreps_endom(dihedral_group(3))
        self.check_irreps_endom(dihedral_group(9))

    def test_endom_irreps_so2(self):
        self.check_irreps_endom(so2_group(4))

    def test_endom_irreps_o2(self):
        self.check_irreps_endom(o2_group(4))

    def test_endom_irreps_ico(self):
        self.check_irreps_endom(ico_group())

    def test_endom_irreps_octa(self):
        self.check_irreps_endom(octa_group())

    def test_endom_irreps_so3(self):
        self.check_irreps_endom(so3_group(4))

    def test_endom_irreps_o3(self):
        self.check_irreps_endom(o3_group(4))

    def test_endom_regular_cyclic_even(self):
        self.check_endom(cyclic_group(2).regular_representation)
        self.check_endom(cyclic_group(4).regular_representation)
        self.check_endom(cyclic_group(16).regular_representation)

    def test_endom_regular_cyclic_odd(self):
        self.check_endom(cyclic_group(1).regular_representation)
        self.check_endom(cyclic_group(3).regular_representation)
        self.check_endom(cyclic_group(9).regular_representation)
        self.check_endom(cyclic_group(13).regular_representation)

    def test_endom_regular_dihedral_even(self):
        self.check_endom(dihedral_group(2).regular_representation)
        self.check_endom(dihedral_group(4).regular_representation)
        self.check_endom(dihedral_group(16).regular_representation)

    def test_endom_regular_dihedral_odd(self):
        self.check_endom(dihedral_group(1).regular_representation)
        self.check_endom(dihedral_group(3).regular_representation)
        self.check_endom(dihedral_group(9).regular_representation)

    def test_endom_regular_ico(self):
        self.check_endom(ico_group().regular_representation)

    def test_endom_bl_regular_so3(self):
        self.check_endom(so3_group(4).bl_regular_representation(3))

    def test_endom_bl_regular_o3(self):
        self.check_endom(o3_group(4).bl_regular_representation(3))

    # test representations

    def test_regular_cyclic(self):
        g = cyclic_group(15)
        rr = g.regular_representation
        self.check_representation(rr)
        self.check_character(rr)

    def test_regular_dihedral(self):
        g = dihedral_group(10)
        rr = g.regular_representation
        self.check_representation(rr)
        self.check_character(rr)

    def test_mix_cyclic(self):
        g = cyclic_group(15)
        rr = directsum(list(g.representations.values()))
        
        self.check_representation(rr)
        self.check_character(rr)

    def test_mix_dihedral(self):
        g = dihedral_group(10)
        rr = directsum(list(g.representations.values()))

        self.check_representation(rr)
        self.check_character(rr)

    def test_mix_so2(self):
        g = so2_group(6)
        rr = directsum(list(g.representations.values()))
    
        self.check_representation(rr)
        self.check_character(rr)

    def test_mix_o2(self):
        g = o2_group(6)
        rr = directsum(list(g.representations.values()))

        self.check_representation(rr)
        self.check_character(rr)

    def test_irreps_so3(self):
        g = so3_group(6)
        
        # the standard representation (equivalent to frequency 1 real irrep / wigner D matrix) should correspond
        # to the usual 3x3 rotation matrix
        for e in g.testing_elements():
            
            psi = e.to('MAT')
            rho = g.standard_representation()(e)

            v = np.array([1, 0, 0]).reshape(-1, 1)
            gv = (psi @ v).reshape(-1)
            ea_e = e.to('zyz')
            
            self.assertTrue(np.allclose(psi, rho), msg=f"""
                \n{e}: \n {ea_e} | gv = {gv} \n {psi} \n != \n {rho} \n
            """)
        
        for l in range(1, 6):
            r = g.irrep(l)
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
            print(r.name)
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
            self.check_representation(r)
            self.check_character(r)

    def test_mix_so3(self):
        g = so3_group(6)
        rr = directsum(list(g.representations.values()))
    
        self.check_representation(rr)
        self.check_character(rr)

    def test_irreps_o3(self):
        g = o3_group(6)
    
        # the frequency 1 real irrep (wigner D matrix) should correspond to the usual 3x3 rotation matrix
        for e in g.testing_elements():
            s, psi = e.to('MAT')
            psi = psi * (-1 if s else 1)
            rho = g.standard_representation()(e)
        
            self.assertTrue(np.allclose(psi, rho), msg=f"""
                \n{e}: \n {psi} \n != \n {rho} \n
            """)
    
        for l in range(1, 6):
            for r in range(2):
                r = g.irrep(r, l)
                print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
                print(r.name)
                print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
                self.check_representation(r)
                self.check_character(r)

    def test_mix_o3(self):
        g = o3_group(6)
        rr = directsum(list(g.representations.values()))
    
        self.check_representation(rr)
        self.check_character(rr)

    def test_irreps_fullocta(self):
        g = full_octa_group()

        for l in range(-1, 4):
            for k in range(2):
                r = g.irrep((k,), (l,))
                print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
                print(r.name)
                print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
                self.check_representation(r)
                self.check_character(r)

    def test_irreps_octa(self):
        g = octa_group()

        for l in range(-1, 4):
            r = g.irrep(l)
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
            print(r.name)
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
            self.check_representation(r)
            self.check_character(r)

    def test_quotients_octa(self):
        g = octa_group()

        for r in [
            g.cube_edges_representation,
            g.cube_vertices_representation,
            g.cube_faces_representation
        ]:
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
            print(r.name)
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
            self.check_representation(r)
            self.check_character(r)

    def test_irreps_fullico(self):
        g = full_ico_group()
    
        for l in range(5):
            for k in range(2):
                r = g.irrep((k,), (l,))
                print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
                print(r.name)
                print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
                self.check_representation(r)
                self.check_character(r)

    def test_irreps_ico(self):
        g = ico_group()
    
        for l in range(5):
            r = g.irrep(l)
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
            print(r.name)
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
            self.check_representation(r)
            self.check_character(r)

    def test_quotients_ico(self):
        g = ico_group()

        for r in [
            g.ico_edges_representation,
            g.ico_vertices_representation,
            g.ico_faces_representation,
        ]:
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
            print(r.name)
            print('+++++++++++++++++++++++++++++++++++++++++++++++++++')
            self.check_representation(r)
            self.check_character(r)

    def check_irreps_endom(self, group: Group):
        for irrep in group.irreps():
            self.check_endom(irrep)
            
            self.assertTrue(irrep.sum_of_squares_constituents == irrep.endomorphism_basis().shape[0])

            for k in range(irrep.sum_of_squares_constituents):
                E_k = irrep.endomorphism_basis()[k, ...]
                
                # check orthogonality
                self.assertTrue(np.allclose(E_k@E_k.T, np.eye(irrep.size)))
                self.assertTrue(np.allclose(E_k.T@E_k, np.eye(irrep.size)))
                
                if k == 0:
                    # if k=0, the matrix need to be the identity
                    self.assertTrue(np.allclose(E_k, np.eye(irrep.size)))
                else:
                    # all other ones need to be skew symmetric
                    self.assertTrue(np.allclose(E_k, -E_k.T))
                
                for l in range(irrep.sum_of_squares_constituents):
                    E_l = irrep.endomorphism_basis()[l, ...]
                    
                    d_kl = (E_l * E_k).sum() / irrep.size
                    
                    if k == l:
                        self.assertTrue(np.allclose(d_kl, 1.))
                    else:
                        self.assertTrue(np.allclose(d_kl, 0.))
                    
                        for i in range(irrep.size):
                            self.assertTrue(np.allclose(
                                E_k[:, i].T @ E_l[:, i], 0.
                            ))

            w = (irrep.endomorphism_basis()**2).sum(axis=0) > 1e-9
        
            self.assertTrue(np.allclose(w.sum(axis=1), irrep.sum_of_squares_constituents))

    def check_endom(self, repr: Representation):
        group = repr.group
    
        np.set_printoptions(precision=2, threshold=2 * repr.size ** 2, suppress=True,
                            linewidth=10 * repr.size + 3)
    
        end_basis = repr.endomorphism_basis()
    
        for a in group.testing_elements():
            r = repr(a)
        
            self.assertTrue(np.allclose(r @ end_basis, end_basis @ r),
                            msg=f"{group.name} | {repr.name} | {a}: endomorphism basis not equivariant\n\n")

    def check_representation(self, repr: Representation):
    
        group = repr.group
        
        np.set_printoptions(precision=2, threshold=2 * repr.size ** 2, suppress=True,
                            linewidth=10 * repr.size + 3)
        
        P = directsum([group.irrep(*irr) for irr in repr.irreps], name="irreps")
        
        self.assertTrue(np.allclose(repr.change_of_basis @ repr.change_of_basis.T, np.eye(repr.size)))
        self.assertTrue(np.allclose(repr.change_of_basis.T @ repr.change_of_basis, np.eye(repr.size)))
        
        # for a in group.testing_elements():
        for _ in range(30):
            a = group.sample()
            repr_1 = repr(a)
            repr_2 = repr.change_of_basis @ P(a) @ repr.change_of_basis_inv

            self.assertTrue(np.allclose(repr_1, repr_2),
                            msg=f"{a}:\n{repr_1}\ndifferent from\n {repr_2}\n")
            
            repr_a_inv = repr(~a)
            repr_inv_a = np.linalg.inv(repr_1)
            self.assertTrue(np.allclose(repr_a_inv, repr_inv_a),
                            msg=f"{a}:\n{repr_a_inv}\ndifferent from\n {repr_inv_a}\n")
            
            # for b in group.testing_elements():
            for _ in range(30):
                b = group.sample()
                repr_ab = repr(a) @ repr(b)
                c = a @ b
                repr_c = repr(c)
                
                self.assertTrue(np.allclose(repr_ab, repr_c), msg=f"{a} x {b} = {c}:\n{repr_ab}\ndifferent from\n {repr_c}\n")

    def check_character(self, repr: Representation):
    
        group = repr.group
    
        np.set_printoptions(precision=2, threshold=2 * repr.size ** 2, suppress=True,
                            linewidth=10 * repr.size + 3)
    
        # for a in group.testing_elements():
        for _ in range(30):
            a = group.sample()

            char_a_1 = repr.character(a)
            char_a_2 = np.trace(repr(a))
        
            self.assertAlmostEqual(char_a_1, char_a_2,
                                   msg=f"""{a}: Character of {repr} different from its trace \n {char_a_1} != {char_a_2} \n""")
        

if __name__ == '__main__':
    unittest.main()
