import unittest
from unittest import TestCase

from escnn.group import *

from collections import defaultdict

import numpy as np


class TestTensorProductRepresentation(TestCase):
    
    def test_tensor_irreps_cyclic_even(self):
        self.check_irrep_tensor_product(cyclic_group(2))
        self.check_irrep_tensor_product(cyclic_group(4))
        self.check_irrep_tensor_product(cyclic_group(16))
    
    def test_tensor_irreps_cyclic_odd(self):
        self.check_irrep_tensor_product(cyclic_group(1))
        self.check_irrep_tensor_product(cyclic_group(3))
        self.check_irrep_tensor_product(cyclic_group(9))
        self.check_irrep_tensor_product(cyclic_group(13))
    
    def test_tensor_irreps_dihedral_even(self):
        self.check_irrep_tensor_product(dihedral_group(2))
        self.check_irrep_tensor_product(dihedral_group(4))
        self.check_irrep_tensor_product(dihedral_group(16))
    
    def test_tensor_irreps_dihedral_odd(self):
        self.check_irrep_tensor_product(dihedral_group(1))
        self.check_irrep_tensor_product(dihedral_group(3))
        self.check_irrep_tensor_product(dihedral_group(9))
    
    def test_tensor_irreps_so2(self):
        G = so2_group(4)
        irreps = [G.irrep(l) for l in range(5)]
        self.check_irrep_tensor_product(so2_group(8), irreps)
    
    def test_tensor_irreps_o2(self):
        G = o2_group(4)
        irreps = [G.irrep(0, 0)] + [G.irrep(1, l) for l in range(5)]
        self.check_irrep_tensor_product(o2_group(8), irreps)
    
    def test_tensor_irreps_ico(self):
        self.check_irrep_tensor_product(ico_group())
    
    def test_tensor_irreps_so3(self):
        G = so3_group(4)
        irreps = [G.irrep(l) for l in range(5)]
        self.check_irrep_tensor_product(so3_group(8), irreps)
    
    def test_tensor_irreps_o3(self):
        G = o3_group(4)
        irreps = [G.irrep(j, l) for j in range(2) for l in range(5)]
        self.check_irrep_tensor_product(o3_group(8), irreps)
        
    def test_tensor_regular_so3(self):
        G = so3_group(4)
        self.check_tensor_product(G.bl_regular_representation(2), G.bl_regular_representation(1))

    def test_tensor_cob_so3(self):
        G = so3_group(4)
        r1 = directsum([G.irrep(0), G.irrep(1), G.irrep(2)], np.random.randn(9, 9))
        self.check_tensor_product(r1, r1)

    def check_irrep_tensor_product(self, group: Group, irreps=None):
        if irreps is None:
            irreps = group.irreps()
        for irr1 in irreps:
            for irr2 in irreps:
                assert irr1.group == group
                assert irr2.group == group
                
                self.check_tensor_product(irr1, irr2)

    def check_tensor_product(self, rho1: Representation, rho2: Representation):
        
        assert rho1.group == rho2.group
        
        G = rho1.group
        
        rho1x2 = rho1.tensor(rho2)
        
        self.assertTrue(rho1x2.group == G)
        self.assertTrue(rho1x2.size == rho1.size * rho2.size)

        for g in G.testing_elements():
            rho12_g = np.kron(rho1(g), rho2(g))
            self.assertTrue(np.allclose(rho12_g, rho1x2(g)))


if __name__ == '__main__':
    unittest.main()
