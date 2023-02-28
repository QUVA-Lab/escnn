import unittest
from unittest import TestCase

from escnn.group import *

import numpy as np


class TestRestrictRepresentations(TestCase):

    ########################
    # Dihedral
    ########################

    def test_restrict_dihedral(self):
        dg = DihedralGroup(8)
        sg_id = (0, 4)
        for irrep in dg.irreps():
            self.check_restriction(dg, sg_id, irrep)
    
    def test_restrict_irreps_dihedral_odd_dihedral(self):
        N = 9
        dg = DihedralGroup(N)
        for rot in range(1, N):
            if N % rot == 0:
                for axis in range(int(N // rot)):
                    sg_id = (axis, rot)
                    for irrep in dg.irreps():
                        self.check_restriction(dg, sg_id, irrep)
    
    def test_restrict_irreps_dihedral_odd_cyclic_odd(self):
        dg = DihedralGroup(9)
        sg_id = (None, 3)
        for irrep in dg.irreps():
            self.check_restriction(dg, sg_id, irrep)
    
    def test_restrict_irreps_dihedral_odd_flips(self):
        dg = DihedralGroup(11)
        for axis in range(11):
            sg_id = (axis, 1)
            for irrep in dg.irreps():
                self.check_restriction(dg, sg_id, irrep)
    
    def test_restrict_irreps_dihedral_even_dihedral(self):
        N = 12
        dg = DihedralGroup(N)
        for rot in range(1, N):
            if N % rot == 0:
                for axis in range(int(N//rot)):
                    sg_id = (axis, rot)
                    for irrep in dg.irreps():
                        self.check_restriction(dg, sg_id, irrep)

    def test_restrict_irreps_dihedral_even_cyclic_even(self):
        dg = DihedralGroup(12)
        sg_id = (None, 4)
        for irrep in dg.irreps():
            self.check_restriction(dg, sg_id, irrep)

    def test_restrict_irreps_dihedral_even_cyclic_odd(self):
        dg = DihedralGroup(12)
        sg_id = (None, 3)
        for irrep in dg.irreps():
            self.check_restriction(dg, sg_id, irrep)
            
    def test_restrict_irreps_dihedral_even_flips(self):
        dg = DihedralGroup(12)
        sg_id = (1, 1)
        for irrep in dg.irreps():
            self.check_restriction(dg, sg_id, irrep)

    def test_restrict_irreps_dihedral_even_cyclic(self):
        dg = DihedralGroup(12)
        sg_id = (None, 12)
        for irrep in dg.irreps():
            self.check_restriction(dg, sg_id, irrep)
            
    def test_restrict_irreps_dihedral_odd_cyclic(self):
        dg = DihedralGroup(13)
        sg_id = (None, 13)
        for irrep in dg.irreps():
            self.check_restriction(dg, sg_id, irrep)

    def test_restrict_rr_dihedral_even_flips(self):
        dg = DihedralGroup(10)
        repr = dg.regular_representation
        sg_id = (1, 1)
        self.check_restriction(dg, sg_id, repr)

    def test_restrict_rr_dihedral_odd_flips(self):
        dg = DihedralGroup(11)
        repr = dg.regular_representation
        sg_id = (1, 1)
        self.check_restriction(dg, sg_id, repr)

    def test_restrict_rr_dihedral_even_dihedral_even(self):
        dg = DihedralGroup(12)
        repr = dg.regular_representation
        sg_id = (1, 6)
        self.check_restriction(dg, sg_id, repr)

    def test_restrict_rr_dihedral_even_dihedral_odd(self):
        dg = DihedralGroup(12)
        repr = dg.regular_representation
        sg_id = (1, 3)
        self.check_restriction(dg, sg_id, repr)

    def test_restrict_rr_dihedral_odd_dihedral_odd(self):
        dg = DihedralGroup(9)
        repr = dg.regular_representation
        sg_id = (1, 3)
        self.check_restriction(dg, sg_id, repr)

    def test_restrict_rr_dihedral_even_cyclic_even(self):
        dg = DihedralGroup(8)
        repr = dg.regular_representation
        sg_id = (None, 4)
        self.check_restriction(dg, sg_id, repr)

    def test_restrict_rr_dihedral_even_cyclic_odd(self):
        dg = DihedralGroup(12)
        repr = dg.regular_representation
        sg_id = (None, 3)
        self.check_restriction(dg, sg_id, repr)

    def test_restrict_rr_dihedral_odd_cyclic_odd(self):
        dg = DihedralGroup(9)
        repr = dg.regular_representation
        sg_id = (None, 3)
        self.check_restriction(dg, sg_id, repr)

    def test_restrict_rr_cyclic_even_cyclic_even(self):
        dg = CyclicGroup(16)
        repr = dg.regular_representation
        sg_id = 8
        self.check_restriction(dg, sg_id, repr)

    def test_restrict_rr_cyclic_even_cyclic_odd(self):
        dg = CyclicGroup(14)
        repr = dg.regular_representation
        sg_id = 7
        self.check_restriction(dg, sg_id, repr)

    def test_restrict_rr_cyclic_odd_cyclic_odd(self):
        dg = CyclicGroup(15)
        repr = dg.regular_representation
        sg_id = 5
        self.check_restriction(dg, sg_id, repr)

    ########################
    # Cyclic
    ########################

    def test_restrict_irreps_cyclic_odd_cyclic_odd(self):
        dg = CyclicGroup(9)
        sg_id = 3
        for irrep in dg.irreps():
            self.check_restriction(dg, sg_id, irrep)

    def test_restrict_irreps_cyclic_even_cyclic_even(self):
        dg = CyclicGroup(8)
        sg_id = 2
        for irrep in dg.irreps():
            self.check_restriction(dg, sg_id, irrep)

    def test_restrict_irreps_cyclic_even_cyclic_odd(self):
        dg = CyclicGroup(10)
        sg_id = 5
        for irrep in dg.irreps():
            self.check_restriction(dg, sg_id, irrep)

    ########################
    # O(2)
    ########################

    def test_restrict_irreps_o2_dihedral_odd(self):
        dg = O2(10)
        
        S = 7
        for axis in range(S):
            sg_id = (axis*2*np.pi/(5*S), 5)
            for irrep in dg.bl_irreps(8):
                irrep = dg.irrep(*irrep)
                self.check_restriction(dg, sg_id, irrep)

    def test_restrict_irreps_o2_cyclic_odd(self):
        dg = O2(10)
        sg_id = (None, 3)
        for irrep in dg.bl_irreps(8):
            irrep = dg.irrep(*irrep)
            self.check_restriction(dg, sg_id, irrep)

    def test_restrict_irreps_o2_flips(self):
        dg = O2(10)

        S = 7
        for axis in range(S):
            sg_id = (axis*2*np.pi/S, 1)
            for irrep in dg.bl_irreps(8):
                irrep = dg.irrep(*irrep)
                self.check_restriction(dg, sg_id, irrep)

    def test_restrict_irreps_o2_dihedral_even(self):
        dg = O2(10)

        S = 7
        for axis in range(S):
            sg_id = (axis*2*np.pi/(6*S), 6)
            for irrep in dg.bl_irreps(8):
                irrep = dg.irrep(*irrep)
                self.check_restriction(dg, sg_id, irrep)

    def test_restrict_irreps_o2_cyclic_even(self):
        dg = O2(10)
        sg_id = (None, 4)
        for irrep in dg.bl_irreps(8):
            irrep = dg.irrep(*irrep)
            self.check_restriction(dg, sg_id, irrep)

    def test_restrict_o2_flips(self):
        dg = O2(10)
        repr = directsum([dg.irrep(*irr) for irr in dg.bl_irreps(7)])
        sg_id = (1., 1)
        self.check_restriction(dg, sg_id, repr)

    def test_restrict_o2_dihedral_even(self):
        dg = O2(10)
        repr = directsum([dg.irrep(*irr) for irr in dg.bl_irreps(7)])
        sg_id = (0., 6)
        self.check_restriction(dg, sg_id, repr)

    def test_restrict_o2_dihedral_odd(self):
        dg = O2(10)
        repr = directsum([dg.irrep(*irr) for irr in dg.bl_irreps(7)])
        sg_id = (0., 3)
        self.check_restriction(dg, sg_id, repr)

    def test_restrict_o2_so2(self):
        dg = O2(10)
        repr = directsum([dg.irrep(*irr) for irr in dg.bl_irreps(7)])
        sg_id = (None, -1)
        self.check_restriction(dg, sg_id, repr)

    def test_restrict_o2_cyclic_even(self):
        dg = O2(10)
        repr = directsum([dg.irrep(*irr) for irr in dg.bl_irreps(7)])
        sg_id = (None, 4)
        self.check_restriction(dg, sg_id, repr)

    def test_restrict_o2_cyclic_odd(self):
        dg = O2(10)
        repr = directsum([dg.irrep(*irr) for irr in dg.bl_irreps(7)])
        sg_id = (None, 3)
        self.check_restriction(dg, sg_id, repr)

    def test_restrict_o2_o2(self):
        dg = O2(10)
        repr = directsum([dg.irrep(*irr) for irr in dg.bl_irreps(7)])
        sg_id = (1., -1)
        self.check_restriction(dg, sg_id, repr)

    ########################
    # SO(2)
    ########################

    def test_restrict_irreps_so2_cyclic_odd(self):
        dg = SO2(10)
        sg_id = 3
        for irrep in dg.bl_irreps(6):
            irrep = dg.irrep(*irrep)
            self.check_restriction(dg, sg_id, irrep)

    def test_restrict_irreps_so2_cyclic_even(self):
        dg = SO2(10)
        sg_id = 4
        for irrep in dg.bl_irreps(6):
            irrep = dg.irrep(*irrep)
            self.check_restriction(dg, sg_id, irrep)

    def test_restrict_so2_cyclic_even(self):
        dg = SO2(10)
        repr = directsum([dg.irrep(*irr) for irr in dg.bl_irreps(7)])
        sg_id = 8
        self.check_restriction(dg, sg_id, repr)

    def test_restrict_so2_cyclic_odd(self):
        dg = SO2(10)
        repr = directsum([dg.irrep(*irr) for irr in dg.bl_irreps(7)])
        sg_id = 7
        self.check_restriction(dg, sg_id, repr)

    ########################
    # SO(3)
    ########################

    def test_restrict_so3_ico(self):
        dg = SO3(5)
        for repr in dg.irreps():

            sg_id = 'ico'
            self.check_restriction(dg, sg_id, repr)

            for _ in range(5):
                adj = dg.sample()
                sg_id = 'ico', adj
                self.check_restriction(dg, sg_id, repr)

    def test_restrict_so3_so2(self):
        dg = SO3(5)
        
        for irrep in dg.bl_irreps(6):
            irrep = dg.irrep(*irrep)
        
            sg_id = (False, -1)
            self.check_restriction(dg, sg_id, irrep)
            
            for _ in range(5):
                adj = dg.sample()
                sg_id = (False, -1, adj)
                self.check_restriction(dg, sg_id, irrep)

    def test_restrict_so3_cn(self):
        dg = SO3(5)
        repr = directsum([dg.irrep(*irr) for irr in dg.bl_irreps(5)])
        for _ in range(5):
            adj = dg.sample()
            for n in [1, 2, 3, 5, 7, 16]:
                sg_id = (False, n, adj)
                self.check_restriction(dg, sg_id, repr)

    def test_restrict_so3_o2(self):
        dg = SO3(9)
    
        for irrep in dg.bl_irreps(6):
            irrep = dg.irrep(*irrep)
        
            sg_id = (True, -1)
            self.check_restriction(dg, sg_id, irrep)
        
            for _ in range(5):
                adj = dg.sample()
                sg_id = (True, -1, adj)
                self.check_restriction(dg, sg_id, irrep)

    def test_restrict_so3_dn(self):
        dg = SO3(5)
        repr = directsum([dg.irrep(*irr) for irr in dg.bl_irreps(5)])
        for _ in range(5):
            adj = dg.sample()
            for n in [1, 2, 3, 5, 7, 16]:
                sg_id = (True, n, adj)
                self.check_restriction(dg, sg_id, repr)

    ########################
    # O(3)
    ########################

    def test_restrict_o3_fullico(self):
        dg = O3(6)
    
        for irrep in dg.bl_irreps(6):
            irrep = dg.irrep(*irrep)
            sg_id = (True, 'ico')
            self.check_restriction(dg, sg_id, irrep)
        
            for _ in range(5):
                adj = dg.sample()
                sg_id = (True, 'ico', adj)
                self.check_restriction(dg, sg_id, irrep)

    def test_restrict_o3_fullocta(self):
        dg = O3(6)

        for irrep in dg.bl_irreps(6):
            irrep = dg.irrep(*irrep)
            sg_id = (True, 'octa')
            self.check_restriction(dg, sg_id, irrep)

            for _ in range(5):
                adj = dg.sample()
                sg_id = (True, 'octa', adj)
                self.check_restriction(dg, sg_id, irrep)

    def test_restrict_o3_so3(self):
        dg = O3(7)
        
        for irrep in dg.bl_irreps(6):
            irrep = dg.irrep(*irrep)
            
            sg_id = 'so3'
            self.check_restriction(dg, sg_id, irrep)
        
            for _ in range(5):
                adj = dg.sample()
                sg_id = ('so3', adj)
                self.check_restriction(dg, sg_id, irrep)

    def test_restrict_o3_ico(self):
        dg = O3(7)
        
        for irrep in dg.bl_irreps(6):
            irrep = dg.irrep(*irrep)
            sg_id = (False, 'ico')
            self.check_restriction(dg, sg_id, irrep)
        
            for _ in range(5):
                adj = dg.sample()
                sg_id = (False, 'ico', adj)
                self.check_restriction(dg, sg_id, irrep)

    def test_restrict_o3_octa(self):
        dg = O3(7)

        for irrep in dg.bl_irreps(6):
            irrep = dg.irrep(*irrep)
            sg_id = (False, 'octa')
            self.check_restriction(dg, sg_id, irrep)

            for _ in range(5):
                adj = dg.sample()
                sg_id = (False, 'octa', adj)
                self.check_restriction(dg, sg_id, irrep)

    def test_restrict_o3_dih_o2(self):
        dg = O3(7)
        
        for irrep in dg.bl_irreps(5):
            irrep = dg.irrep(*irrep)
            for axis in [0., np.pi / 2, np.pi / 3, np.pi * 1.25, 2 * np.pi]:
                sg_id = (False, 2*axis, -1)
                self.check_restriction(dg, sg_id, irrep)
            
                for _ in range(3):
                    adj = dg.sample()
                    assert adj.group == dg
                    sg_id = (False, 2*axis, -1, adj)
                    self.check_restriction(dg, sg_id, irrep)

    def test_restrict_o3_so2(self):
        dg = O3(7)
        
        for irrep in dg.bl_irreps(5):
            irrep = dg.irrep(*irrep)
            sg_id = (False, False, -1)
            self.check_restriction(dg, sg_id, irrep)
        
            for _ in range(3):
                adj = dg.sample()
                sg_id = (False, False, -1, adj)
                self.check_restriction(dg, sg_id, irrep)

    def test_restrict_o3_dih_dn(self):
        dg = O3(7)
        
        for irrep in dg.bl_irreps(5):
            irrep = dg.irrep(*irrep)
            for n in [3, 4, 6, 9]:
                for axis in [0., np.pi / 2, np.pi / 3, np.pi * 1.25, 2 * np.pi]:
                    sg_id = (False, 2*axis, n)
                    self.check_restriction(dg, sg_id, irrep)
                
                    for _ in range(3):
                        adj = dg.sample()
                        sg_id = (False, 2*axis, n, adj)
                        self.check_restriction(dg, sg_id, irrep)
            
    def test_restrict_o3_cn(self):
        dg = O3(7)
        
        for irrep in dg.bl_irreps(5):
            irrep = dg.irrep(*irrep)
            for n in [2, 3, 4, 6, 9]:
            
                sg_id = (False, False, n)
                self.check_restriction(dg, sg_id, irrep)
            
                for _ in range(3):
                    adj = dg.sample()
                    sg_id = (False, False, n, adj)
                    self.check_restriction(dg, sg_id, irrep)

    def test_restrict_o3_flip(self):
        dg = O3(7)
        
        for irrep in dg.bl_irreps(5):
            irrep = dg.irrep(*irrep)
            for axis in [0., np.pi / 2, np.pi / 3, np.pi * 1.25, 2 * np.pi]:
            
                sg_id = (False, 2*axis, 1)
                self.check_restriction(dg, sg_id, irrep)
            
                for _ in range(3):
                    adj = dg.sample()
                    sg_id = (False, 2*axis, 1, adj)
                    self.check_restriction(dg, sg_id, irrep)

    def test_restrict_o3_cylinder_so2xc2(self):
        dg = O3(7)

        for irrep in dg.bl_irreps(5):
            irrep = dg.irrep(*irrep)

            # cylinder aligned along Z axis
            # i.e., rotation along Z axis
            # and 3d inversions
            sg_id = (True, False, -1)
            self.check_restriction(dg, sg_id, irrep)

            for _ in range(3):
                adj = dg.sample()
                sg_id = (True, False, -1, adj)
                self.check_restriction(dg, sg_id, irrep)

    def test_restrict_o3_cylinder_cnxc2(self):
        dg = O3(7)

        for irrep in dg.bl_irreps(5):
            irrep = dg.irrep(*irrep)
            for n in [2, 3, 4, 6, 9]:

                # cylinder aligned along Z axis
                # i.e., rotation along Z axis
                # and 3d inversions
                sg_id = (True, False, n)
                self.check_restriction(dg, sg_id, irrep)

                for _ in range(3):
                    adj = dg.sample()
                    sg_id = (True, False, n, adj)
                    self.check_restriction(dg, sg_id, irrep)

    def test_restrict_o3_fullcylinder_o2xc2(self):
        dg = O3(7)

        for irrep in dg.bl_irreps(5):
            irrep = dg.irrep(*irrep)

            # cylinder aligned along Z axis
            # i.e., rotation along Z axis
            # and 3d inversions
            sg_id = (True, True, -1)
            self.check_restriction(dg, sg_id, irrep)

            for _ in range(3):
                adj = dg.sample()
                sg_id = (True, True, -1, adj)
                self.check_restriction(dg, sg_id, irrep)

    def test_restrict_o3_fullcylinder_dnxc2(self):
        dg = O3(7)

        for irrep in dg.bl_irreps(5):
            irrep = dg.irrep(*irrep)
            for n in [2, 3, 4, 6, 9]:

                # cylinder aligned along Z axis
                # i.e., rotation along Z axis
                # and 3d inversions
                sg_id = (True, True, n)
                self.check_restriction(dg, sg_id, irrep)

                for _ in range(3):
                    adj = dg.sample()
                    sg_id = (True, True, n, adj)
                    self.check_restriction(dg, sg_id, irrep)

    def test_restrict_o3_cone_o2(self):
        dg = O3(7)
        
        for irrep in dg.bl_irreps(5):
            irrep = dg.irrep(*irrep)
    
            # Cone aligned along Z axis
            # i.e., rotation along Z axis
            # on XY plane, mirror wrt Y axis (i.e. flip along X axis)
            sg_id = ('cone', 0., -1)
            self.check_restriction(dg, sg_id, irrep)
        
            # Cone aligned along Z axis
            # i.e., rotation along Z axis
            # on XY plane, mirror wrt X axis (i.e. flip along Y axis)
            sg_id = ('cone', np.pi, -1)
            self.check_restriction(dg, sg_id, irrep)
        
            for _ in range(3):
                adj = dg.sample()
                sg_id = ('cone', -1, adj)
                self.check_restriction(dg, sg_id, irrep)
            
    def test_restrict_o3_cone_dn(self):
        dg = O3(7)
        
        for irrep in dg.bl_irreps(5):
            irrep = dg.irrep(*irrep)
    
            for N in [2, 3, 5, 8]:
                # Cone aligned along Z axis
                # i.e., rotation along Z axis
                # on XY plane, mirror wrt Y axis (i.e. flip along X axis)
                sg_id = ('cone', 0., N)
                self.check_restriction(dg, sg_id, irrep)
            
                # Cone aligned along Z axis
                # i.e., rotation along Z axis
                # on XY plane, mirror wrt X axis (i.e. flip along Y axis)
                sg_id = ('cone', np.pi, N)
                self.check_restriction(dg, sg_id, irrep)
            
                for _ in range(3):
                    adj = dg.sample()
                    sg_id = ('cone', N, adj)
                    self.check_restriction(dg, sg_id, irrep)
                
    def test_restrict_o3_mir(self):
        dg = O3(7)
        
        for irrep in dg.bl_irreps(5):
            irrep = dg.irrep(*irrep)
    
            # Mirroring along the Y axis
            sg_id = ('cone', 0., 1)
            self.check_restriction(dg, sg_id, irrep)
        
            # Mirroring along the X axis
            sg_id = ('cone', np.pi, 1)
            self.check_restriction(dg, sg_id, irrep)
        
            for _ in range(3):
                adj = dg.sample()
                sg_id = ('cone', 1, adj)
                self.check_restriction(dg, sg_id, irrep)
            
    def test_restrict_o3_inv(self):
        dg = O3(7)
        
        for irrep in dg.bl_irreps(5):
            irrep = dg.irrep(*irrep)
    
            # Inversion wrt the origin
            sg_id = (True, False, 1)
            self.check_restriction(dg, sg_id, irrep)

    def test_restrict_ico(self):
        dg = ico_group()

        for irr in dg.irreps():
            for n in [1, 2, 3, 5]:
                sg_id = (False, n)
                self.check_restriction(dg, sg_id, irr)

            sg_id = (True, 1)
            self.check_restriction(dg, sg_id, irr)

    ########################
    # Some DirectProducts
    ########################

    def test_restrict_cylinder_c2xso2(self):
        dg = cylinder_group(3)

        for irr in dg.irreps():

            # the C2 subgrousp
            sg_id = (2, 1)
            self.check_restriction(dg, sg_id, irr)

            for N in [-1, 2, 3, 5, 8]:

                # the CN or SO(2) subgroups
                sg_id = (1, N)
                self.check_restriction(dg, sg_id, irr)

    def test_restrict_cylinder_c2xo2(self):
        dg = full_cylinder_group(3)

        for irr in dg.irreps():

            # the C2 subgrousp
            sg_id = (2, (None, 1))
            self.check_restriction(dg, sg_id, irr)

            for N in [-1, 2, 3, 5, 8]:

                # the DN or O(2) subgroups
                sg_id = (1, (0., N))
                self.check_restriction(dg, sg_id, irr)

                # the CN or SO(2) subgroups
                sg_id = (1, (None, N))
                self.check_restriction(dg, sg_id, irr)

    def test_restrict_fullico(self):
        dg = full_ico_group()

        for irr in dg.irreps():

            # the C2 subgroup
            sg_id = (2, (False, 1))
            self.check_restriction(dg, sg_id, irr)

            for n in [1, 2, 3, 5]:
                sg_id = (1, (False, n))
                self.check_restriction(dg, sg_id, irr)

                # sg_id = (1, (True, n))
                # self.check_restriction(dg, sg_id, irr)

    def test_restrict_fullocta(self):
        dg = full_octa_group()

        for irr in dg.irreps():

            # the C2 subgrousp
            sg_id = (2, (False, 1))
            self.check_restriction(dg, sg_id, irr)

            for n in [1, 2, 3, 4]:
                sg_id = (1, (False, n))
                self.check_restriction(dg, sg_id, irr)

                # sg_id = (1, (True, n))
                # self.check_restriction(dg, sg_id, irr)

    def test_restrict_o2xo2(self):
        L = 2
        dg: DirectProductGroup = double_group(o2_group(2*L))

        irr = dg.irrep((1,1), (1,1))
        self.check_restriction(dg, dg.subgroup1_id, irr)

    def test_restrict_so3xso3(self):
        L = 2
        dg: DirectProductGroup = double_group(so3_group(2*L))

        print(dg._keys)

        for l1 in range(L+1):
            for l2 in range(L+1):
                print(l1, l2)

                irr = dg.irrep((l1,), (l2,))

                for sg_id in [dg.subgroup1_id, dg.subgroup2_id, 'diagonal']:
                    print(sg_id)
                    self.check_restriction(dg, sg_id, irr)

    ####################################################################################################################

    def check_restriction(self, group, subgroup_id, repr):
    
        assert repr.group == group
    
        sg, parent_element, child_element = group.subgroup(subgroup_id)
    
        restrict_repr = repr.restrict(subgroup_id)
        
        for e in group.testing_elements():
            c = child_element(e)
            if c is not None:
                assert parent_element(c) == e, f"{group} to subgroup {sg} with id {subgroup_id} | Element {e} != {parent_element(c)} | {c}"
                
        for e in sg.testing_elements():
            
            assert child_element(parent_element(e)) == e, f"Element {e} from subgroup {sg.name}: {parent_element(e)}, {child_element(parent_element(e))}"
            
            repr_a = repr(parent_element(e))
            repr_b = restrict_repr(e)

            np.set_printoptions(precision=2, threshold=2 * repr_a.size**2, suppress=True, linewidth=10*repr_a.size + 3)
            self.assertTrue(np.allclose(repr_a, repr_b), msg=f"{group.name} | {repr.name} | {subgroup_id} | {e}:\n{repr_a}\ndifferent from\n {repr_b}\n")
            if not np.allclose(repr_a, repr_b):
                print(f"{repr.name} | {e}:\n{repr_a}\ndifferent from\n {repr_b}\n")
                
        for _ in range(10):
            e = sg.sample()
            
            assert child_element(parent_element(
                e)) == e, f"Element {e} from subgroup {sg.name}: {parent_element(e)}, {child_element(parent_element(e))}"

            repr_a = repr(parent_element(e))
            repr_b = restrict_repr(e)

            np.set_printoptions(precision=2, threshold=2 * repr_a.size ** 2, suppress=True,
                                linewidth=10 * repr_a.size + 3)
            self.assertTrue(np.allclose(repr_a, repr_b),
                            msg=f"{group.name} | {repr.name} | {subgroup_id} | {e}:\n{repr_a}\ndifferent from\n {repr_b}\n")
            if not np.allclose(repr_a, repr_b):
                print(f"{repr.name} | {e}:\n{repr_a}\ndifferent from\n {repr_b}\n")


if __name__ == '__main__':
    unittest.main()
