import unittest
from unittest import TestCase

from escnn.group import *
from escnn.group._numerical import decompose_representation_general

import numpy as np


class TestNumericalDecomposition(TestCase):

    ########################
    # Dihedral
    ########################

    def test_restrict_rr_dihedral_even_flips(self):
        dg = DihedralGroup(10)
        repr = dg.regular_representation
        sg_id = (1, 1)
        self.check_decomposition(repr.restrict(sg_id))

    def test_restrict_rr_dihedral_odd_flips(self):
        dg = DihedralGroup(11)
        repr = dg.regular_representation
        sg_id = (1, 1)
        self.check_decomposition(repr.restrict(sg_id))

    def test_restrict_rr_dihedral_even_dihedral_even(self):
        dg = DihedralGroup(12)
        repr = dg.regular_representation
        sg_id = (1, 6)
        self.check_decomposition(repr.restrict(sg_id))

    def test_restrict_rr_dihedral_even_dihedral_odd(self):
        dg = DihedralGroup(12)
        repr = dg.regular_representation
        sg_id = (1, 3)
        self.check_decomposition(repr.restrict(sg_id))

    def test_restrict_rr_dihedral_odd_dihedral_odd(self):
        dg = DihedralGroup(9)
        repr = dg.regular_representation
        sg_id = (1, 3)
        self.check_decomposition(repr.restrict(sg_id))

    def test_restrict_rr_dihedral_even_cyclic_even(self):
        dg = DihedralGroup(8)
        repr = dg.regular_representation
        sg_id = (None, 4)
        self.check_decomposition(repr.restrict(sg_id))

    def test_restrict_rr_dihedral_even_cyclic_odd(self):
        dg = DihedralGroup(12)
        repr = dg.regular_representation
        sg_id = (None, 3)
        self.check_decomposition(repr.restrict(sg_id))

    def test_restrict_rr_dihedral_odd_cyclic_odd(self):
        dg = DihedralGroup(9)
        repr = dg.regular_representation
        sg_id = (None, 3)
        self.check_decomposition(repr.restrict(sg_id))

    def test_restrict_rr_cyclic_even_cyclic_even(self):
        dg = CyclicGroup(16)
        repr = dg.regular_representation
        sg_id = 8
        self.check_decomposition(repr.restrict(sg_id))

    def test_restrict_rr_cyclic_even_cyclic_odd(self):
        dg = CyclicGroup(14)
        repr = dg.regular_representation
        sg_id = 7
        self.check_decomposition(repr.restrict(sg_id))

    def test_restrict_rr_cyclic_odd_cyclic_odd(self):
        dg = CyclicGroup(15)
        repr = dg.regular_representation
        sg_id = 5
        self.check_decomposition(repr.restrict(sg_id))

    ########################
    # O(2)
    ########################

    def test_restrict_o2_flips(self):
        dg = O2(10)
        repr = directsum(dg.irreps())
        sg_id = (1., 1)
        self.check_decomposition(repr.restrict(sg_id))

    def test_restrict_o2_dihedral_even(self):
        dg = O2(10)
        repr = directsum(dg.irreps())
        sg_id = (0., 6)
        self.check_decomposition(repr.restrict(sg_id))

    def test_restrict_o2_dihedral_odd(self):
        dg = O2(10)
        repr = directsum(dg.irreps())
        sg_id = (0., 3)
        self.check_decomposition(repr.restrict(sg_id))

    def test_restrict_o2_so2(self):
        dg = O2(10)
        repr = directsum(dg.irreps())
        sg_id = (None, -1)
        self.check_decomposition(repr.restrict(sg_id))

    def test_restrict_o2_cyclic_even(self):
        dg = O2(10)
        repr = directsum(dg.irreps())
        sg_id = (None, 4)
        self.check_decomposition(repr.restrict(sg_id))

    def test_restrict_o2_cyclic_odd(self):
        dg = O2(10)
        repr = directsum(dg.irreps())
        sg_id = (None, 3)
        self.check_decomposition(repr.restrict(sg_id))

    def test_restrict_o2_o2(self):
        dg = O2(10)
        repr = directsum(dg.irreps())
        sg_id = (1., -1)
        self.check_decomposition(repr.restrict(sg_id))

    ########################
    # SO(2)
    ########################

    def test_restrict_so2_cyclic_even(self):
        dg = SO2(10)
        repr = directsum(dg.irreps())
        sg_id = 8
        self.check_decomposition(repr.restrict(sg_id))

    def test_restrict_so2_cyclic_odd(self):
        dg = SO2(10)
        repr = directsum(dg.irreps())
        sg_id = 7
        self.check_decomposition(repr.restrict(sg_id))

    ########################
    # SO(3)
    ########################

    def test_restrict_so3_ico(self):
        dg = SO3(5)
        for repr in dg.irreps():

            sg_id = 'ico'
            self.check_decomposition(repr.restrict(sg_id))

            for _ in range(5):
                adj = dg.sample()
                sg_id = 'ico', adj
                self.check_decomposition(repr.restrict(sg_id))

    def test_restrict_so3_so2(self):
        dg = SO3(5)
        
        for irrep in dg.irreps():
        
            sg_id = (False, -1)
            self.check_decomposition(irrep.restrict(sg_id))
            
            for _ in range(5):
                adj = dg.sample()
                sg_id = (False, -1, adj)
                self.check_decomposition(irrep.restrict(sg_id))

    def test_restrict_so3_cn(self):
        dg = SO3(5)
        repr = directsum(dg.irreps())
        for _ in range(5):
            adj = dg.sample()
            for n in [1, 2, 3, 5, 7, 16]:
                sg_id = (False, n, adj)
                self.check_decomposition(repr.restrict(sg_id))

    def test_restrict_so3_o2(self):
        dg = SO3(9)
    
        for irrep in dg.irreps():
        
            sg_id = (True, -1)
            self.check_decomposition(irrep.restrict(sg_id))
        
            for _ in range(5):
                adj = dg.sample()
                sg_id = (True, -1, adj)
                self.check_decomposition(irrep.restrict(sg_id))

    def test_restrict_so3_dn(self):
        dg = SO3(5)
        repr = directsum(dg.irreps())
        for _ in range(5):
            adj = dg.sample()
            for n in [1, 2, 3, 5, 7, 16]:
                sg_id = (True, n, adj)
                self.check_decomposition(repr.restrict(sg_id))

    ########################
    # O(3)
    ########################

    def test_restrict_o3_fullico(self):
        dg = O3(6)
    
        for irrep in dg.irreps():
            sg_id = (True, 'ico')
            self.check_decomposition(irrep.restrict(sg_id))
        
            for _ in range(5):
                adj = dg.sample()
                sg_id = (True, 'ico', adj)
                self.check_decomposition(irrep.restrict(sg_id))

    def test_restrict_o3_fullocta(self):
        dg = O3(6)

        for irrep in dg.irreps():
            sg_id = (True, 'octa')
            self.check_decomposition(irrep.restrict(sg_id))

            for _ in range(5):
                adj = dg.sample()
                sg_id = (True, 'octa', adj)
                self.check_decomposition(irrep.restrict(sg_id))

    def test_restrict_o3_so3(self):
        dg = O3(7)
        
        for irrep in dg.irreps():
            
            sg_id = 'so3'
            self.check_decomposition(irrep.restrict(sg_id))
        
            for _ in range(5):
                adj = dg.sample()
                sg_id = ('so3', adj)
                self.check_decomposition(irrep.restrict(sg_id))

    def test_restrict_o3_ico(self):
        dg = O3(7)
        
        for irrep in dg.irreps():
            sg_id = (False, 'ico')
            self.check_decomposition(irrep.restrict(sg_id))
        
            for _ in range(5):
                adj = dg.sample()
                sg_id = (False, 'ico', adj)
                self.check_decomposition(irrep.restrict(sg_id))

    def test_restrict_o3_octa(self):
        dg = O3(7)

        for irrep in dg.irreps():
            sg_id = (False, 'octa')
            self.check_decomposition(irrep.restrict(sg_id))

            for _ in range(5):
                adj = dg.sample()
                sg_id = (False, 'octa', adj)
                self.check_decomposition(irrep.restrict(sg_id))

    def test_restrict_o3_dih_o2(self):
        dg = O3(7)
        
        for irrep in dg.irreps():
            for axis in [0., np.pi / 2, np.pi / 3, np.pi * 1.25, 2 * np.pi]:
                sg_id = (False, 2*axis, -1)
                self.check_decomposition(irrep.restrict(sg_id))
            
                for _ in range(5):
                    adj = dg.sample()
                    assert adj.group == dg
                    sg_id = (False, 2*axis, -1, adj)
                    self.check_decomposition(irrep.restrict(sg_id))

    def test_restrict_o3_so2(self):
        dg = O3(7)
        
        for irrep in dg.irreps():
            sg_id = (False, False, -1)
            self.check_decomposition(irrep.restrict(sg_id))
        
            for _ in range(5):
                adj = dg.sample()
                sg_id = (False, False, -1, adj)
                self.check_decomposition(irrep.restrict(sg_id))

    def test_restrict_o3_dih_dn(self):
        dg = O3(7)
        
        for irrep in dg.irreps():
            for n in [3, 4, 6, 9]:
                for axis in [0., np.pi / 2, np.pi / 3, np.pi * 1.25, 2 * np.pi]:
                    sg_id = (False, 2*axis, n)
                    self.check_decomposition(irrep.restrict(sg_id))
                
                    for _ in range(5):
                        adj = dg.sample()
                        sg_id = (False, 2*axis, n, adj)
                        self.check_decomposition(irrep.restrict(sg_id))
            
    def test_restrict_o3_cn(self):
        dg = O3(7)
        
        for irrep in dg.irreps():
            for n in [2, 3, 4, 6, 9]:
            
                sg_id = (False, False, n)
                self.check_decomposition(irrep.restrict(sg_id))
            
                for _ in range(5):
                    adj = dg.sample()
                    sg_id = (False, False, n, adj)
                    self.check_decomposition(irrep.restrict(sg_id))

    def test_restrict_o3_flip(self):
        dg = O3(7)
        
        for irrep in dg.irreps():
            for axis in [0., np.pi / 2, np.pi / 3, np.pi * 1.25, 2 * np.pi]:
            
                sg_id = (False, 2*axis, 1)
                self.check_decomposition(irrep.restrict(sg_id))
            
                for _ in range(5):
                    adj = dg.sample()
                    sg_id = (False, 2*axis, 1, adj)
                    self.check_decomposition(irrep.restrict(sg_id))

    def test_restrict_o3_cylinder_so2xc2(self):
        dg = O3(7)

        for irrep in dg.irreps():

            # cylinder aligned along Z axis
            # i.e., rotation along Z axis
            # and 3d inversions
            sg_id = (True, False, -1)
            self.check_decomposition(irrep.restrict(sg_id))

            for _ in range(5):
                adj = dg.sample()
                sg_id = (True, False, -1, adj)
                self.check_decomposition(irrep.restrict(sg_id))

    def test_restrict_o3_cylinder_cnxc2(self):
        dg = O3(7)

        for irrep in dg.irreps():
            for n in [2, 3, 4, 6, 9]:

                # cylinder aligned along Z axis
                # i.e., rotation along Z axis
                # and 3d inversions
                sg_id = (True, False, n)
                self.check_decomposition(irrep.restrict(sg_id))

                for _ in range(5):
                    adj = dg.sample()
                    sg_id = (True, False, n, adj)
                    self.check_decomposition(irrep.restrict(sg_id))

    def test_restrict_o3_fullcylinder_o2xc2(self):
        dg = O3(7)

        for irrep in dg.irreps():

            # cylinder aligned along Z axis
            # i.e., rotation along Z axis
            # and 3d inversions
            sg_id = (True, True, -1)
            self.check_decomposition(irrep.restrict(sg_id))

            for _ in range(5):
                adj = dg.sample()
                sg_id = (True, True, -1, adj)
                self.check_decomposition(irrep.restrict(sg_id))

    def test_restrict_o3_fullcylinder_dnxc2(self):
        dg = O3(7)

        for irrep in dg.irreps():
            for n in [2, 3, 4, 6, 9]:

                # cylinder aligned along Z axis
                # i.e., rotation along Z axis
                # and 3d inversions
                sg_id = (True, True, n)
                self.check_decomposition(irrep.restrict(sg_id))

                for _ in range(5):
                    adj = dg.sample()
                    sg_id = (True, True, n, adj)
                    self.check_decomposition(irrep.restrict(sg_id))

    def test_restrict_o3_cone_o2(self):
        dg = O3(7)
        
        for irrep in dg.irreps():
    
            # Cone aligned along Z axis
            # i.e., rotation along Z axis
            # on XY plane, mirror wrt Y axis (i.e. flip along X axis)
            sg_id = ('cone', 0., -1)
            self.check_decomposition(irrep.restrict(sg_id))
        
            # Cone aligned along Z axis
            # i.e., rotation along Z axis
            # on XY plane, mirror wrt X axis (i.e. flip along Y axis)
            sg_id = ('cone', np.pi, -1)
            self.check_decomposition(irrep.restrict(sg_id))
        
            for _ in range(5):
                adj = dg.sample()
                sg_id = ('cone', -1, adj)
                self.check_decomposition(irrep.restrict(sg_id))
            
    def test_restrict_o3_cone_dn(self):
        dg = O3(7)
        
        for irrep in dg.irreps():
    
            for N in [2, 3, 5, 8]:
                # Cone aligned along Z axis
                # i.e., rotation along Z axis
                # on XY plane, mirror wrt Y axis (i.e. flip along X axis)
                sg_id = ('cone', 0., N)
                self.check_decomposition(irrep.restrict(sg_id))
            
                # Cone aligned along Z axis
                # i.e., rotation along Z axis
                # on XY plane, mirror wrt X axis (i.e. flip along Y axis)
                sg_id = ('cone', np.pi, N)
                self.check_decomposition(irrep.restrict(sg_id))
            
                for _ in range(5):
                    adj = dg.sample()
                    sg_id = ('cone', N, adj)
                    self.check_decomposition(irrep.restrict(sg_id))
                
    def test_restrict_o3_mir(self):
        dg = O3(7)
        
        for irrep in dg.irreps():
    
            # Mirroring along the Y axis
            sg_id = ('cone', 0., 1)
            self.check_decomposition(irrep.restrict(sg_id))
        
            # Mirroring along the X axis
            sg_id = ('cone', np.pi, 1)
            self.check_decomposition(irrep.restrict(sg_id))
        
            for _ in range(5):
                adj = dg.sample()
                sg_id = ('cone', 1, adj)
                self.check_decomposition(irrep.restrict(sg_id))
            
    def test_restrict_o3_inv(self):
        dg = O3(7)
        
        for irrep in dg.irreps():
    
            # Inversion wrt the origin
            sg_id = (True, False, 1)
            self.check_decomposition(irrep.restrict(sg_id))

    ########################
    # Some DirectProducts
    ########################

    def test_restrict_cylinder_c2xso2(self):
        dg = cylinder_group(3)

        for irr in dg.irreps():

            # the C2 subgrousp
            sg_id = (2, 1)
            self.check_decomposition(irr.restrict(sg_id))

            for N in [-1, 2, 3, 5, 8]:

                # the CN or SO(2) subgroups
                sg_id = (1, N)
                self.check_decomposition(irr.restrict(sg_id))

    def test_restrict_cylinder_c2xo2(self):
        dg = full_cylinder_group(3)

        for irr in dg.irreps():

            # the C2 subgrousp
            sg_id = (2, (None, 1))
            self.check_decomposition(irr.restrict(sg_id))

            for N in [-1, 2, 3, 5, 8]:

                # the DN or O(2) subgroups
                sg_id = (1, (0., N))
                self.check_decomposition(irr.restrict(sg_id))

                # the CN or SO(2) subgroups
                sg_id = (1, (None, N))
                self.check_decomposition(irr.restrict(sg_id))

    def test_restrict_fullico(self):
        dg = full_ico_group()

        for irr in dg.irreps():

            # the C2 subgrousp
            sg_id = (2, (False, 1))
            self.check_decomposition(irr.restrict(sg_id))

            for n in [1, 2, 3, 5]:
                sg_id = (1, (False, n))
                self.check_decomposition(irr.restrict(sg_id))

                # sg_id = (1, (True, n))
                # self.check_decomposition(irr.restrict(sg_id)

    def test_restrict_fullocta(self):
        dg = full_octa_group()

        for irr in dg.irreps():

            # the C2 subgrousp
            sg_id = (2, (False, 1))
            self.check_decomposition(irr.restrict(sg_id))

            for n in [1, 2, 3, 4]:
                sg_id = (1, (False, n))
                self.check_decomposition(irr.restrict(sg_id))

                # sg_id = (1, (True, n))
                # self.check_decomposition(irr.restrict(sg_id)

    def test_restrict_so2xso2(self):
        L = 2
        dg: DirectProductGroup = double_group(so2_group(2*L))

        for l1 in range(L+1):
            for l2 in range(L+1):
                irr = dg.irrep((l1,), (l2,))

                for sg_id in [dg.subgroup1_id, dg.subgroup2_id, 'diagonal']:
                    self.check_decomposition(irr.restrict(sg_id))

    def test_restrict_so3xso3(self):
        L = 2
        dg: DirectProductGroup = double_group(so3_group(2*L))

        for l1 in range(L+1):
            for l2 in range(L+1):
                irr = dg.irrep((l1,), (l2,))

                for sg_id in [dg.subgroup1_id, dg.subgroup2_id, 'diagonal']:
                    self.check_decomposition(irr.restrict(sg_id))

    def test_restrict_so3xso2(self):
        L = 2
        dg: DirectProductGroup = direct_product(so3_group(L), so2_group(L))

        for l1 in range(L+1):
            for l2 in range(L+1):
                irr = dg.irrep((l1,), (l2,))

                for sg_id in [dg.subgroup1_id, dg.subgroup2_id]:
                    self.check_decomposition(irr.restrict(sg_id))

    def test_restrict_so3xo2(self):
        L = 2
        dg: DirectProductGroup = direct_product(so3_group(L), o2_group(L))

        for l1 in range(L+1):
            for l2 in range(L+1):
                irr = dg.irrep((l1,), (1, l2))

                for sg_id in [dg.subgroup1_id, dg.subgroup2_id]:
                    self.check_decomposition(irr.restrict(sg_id))

    def test_directsum_so2(self):
        L = 2
        G = so2_group(L)

        np.set_printoptions(precision=3, suppress=True, linewidth=10000, threshold=10000)

        self.check_decomposition(G.irrep(0) + G.irrep(0))
        self.check_decomposition(G.irrep(1) + G.irrep(1))
        self.check_decomposition(G.irrep(0) + G.irrep(1))
        self.check_decomposition(G.irrep(1) + G.irrep(2))
        self.check_decomposition(G.irrep(0) + G.irrep(1) + G.irrep(2))
        self.check_decomposition(G.irrep(1) + G.irrep(2) + G.irrep(3))

    def test_directsum_so3(self):
        L = 2
        G = so3_group(L)

        np.set_printoptions(precision=3, suppress=True, linewidth=10000, threshold=10000)

        self.check_decomposition(G.irrep(0) + G.irrep(0))
        self.check_decomposition(G.irrep(1) + G.irrep(1))
        self.check_decomposition(G.irrep(0) + G.irrep(1))
        self.check_decomposition(G.irrep(1) + G.irrep(2))
        self.check_decomposition(G.irrep(0) + G.irrep(1) + G.irrep(2))
        self.check_decomposition(G.irrep(1) + G.irrep(2) + G.irrep(3))

    ####################################################################################################################

    def check_decomposition(self, repr: Representation):

        G = repr.group

        change_of_basis, irreps_multiplicities = decompose_representation_general(repr, G)

        irreps = []
        for irr, m in irreps_multiplicities:
            irreps += [irr]*m
        new_repr = Representation(G, 'test', irreps, change_of_basis)

        for e in G.testing_elements():
            
            repr_a = repr(e)
            repr_b = new_repr(e)

            np.set_printoptions(precision=2, threshold=2 * repr_a.size**2, suppress=True, linewidth=10*repr_a.size + 3)
            if not np.allclose(repr_a, repr_b):
                print(f"{repr.name} | {e}:\n{repr_a}\ndifferent from\n {repr_b}\n")

            self.assertTrue(np.allclose(repr_a, repr_b), msg=f"{G.name} | {repr.name} | {e}:\n{repr_a}\ndifferent from\n {repr_b}\n")

if __name__ == '__main__':
    unittest.main()
