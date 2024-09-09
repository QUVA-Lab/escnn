import unittest
from unittest import TestCase

from .utils import *

from escnn.group import *
import numpy as np


class TestDirectProduct(TestCase):
    
    def test_trivial_x_trivial(self):
        H = trivial_group()
        G = direct_product(H, H)
        self.check_group(G, H, H)
        
    def test_cn_x_cn(self):
        for n in [1, 2, 3, 5]:
            G1 = cyclic_group(n)
            for m in [2, 3, 5]:
                G2 = cyclic_group(m)
                
                G = direct_product(G1, G2)
                self.check_group(G, G1, G2)

    def test_dn_x_dn(self):
        for n in [1, 2, 3, 4]:
            G1 = dihedral_group(n)
            for m in [1, 2, 3, 4]:
                G2 = dihedral_group(m)
            
                G = direct_product(G1, G2)
                self.check_group(G, G1, G2)

    def test_cn_x_dn(self):
        for n in [1, 2, 3, 4]:
            G1 = cyclic_group(n)
            for m in [1, 2, 3, 4]:
                G2 = dihedral_group(m)
            
                G = direct_product(G1, G2)
                self.check_group(G, G1, G2)

    def test_so2_x_cn(self):
        G1 = so2_group(2)
        for m in [1, 2, 3, 4]:
            G2 = cyclic_group(m)
        
            G = direct_product(G1, G2)
            self.check_group(G, G1, G2)
        
            G = direct_product(G2, G1)
            self.check_group(G, G2, G1)

    def test_o2_x_cn(self):
        G1 = o2_group(2)
        for m in [1, 2, 3, 4]:
            G2 = cyclic_group(m)
        
            G = direct_product(G1, G2)
            self.check_group(G, G1, G2)
        
            G = direct_product(G2, G1)
            self.check_group(G, G2, G1)

    def test_so2_x_dn(self):
        G1 = so2_group(2)
        for m in [1, 2, 3, 4]:
            G2 = dihedral_group(m)
        
            G = direct_product(G1, G2)
            self.check_group(G, G1, G2)
            
            G = direct_product(G2, G1)
            self.check_group(G, G2, G1)

    def test_o2_x_dn(self):
        G1 = o2_group(2)
        for m in [1, 2, 3, 4]:
            G2 = dihedral_group(m)
        
            G = direct_product(G1, G2)
            self.check_group(G, G1, G2)
        
            G = direct_product(G2, G1)
            self.check_group(G, G2, G1)

    def test_ico_x_c2(self):
        G1 = ico_group()
        G2 = cyclic_group(2)
    
        G = direct_product(G1, G2)
        self.check_group(G, G1, G2)
    
        G = direct_product(G2, G1)
        self.check_group(G, G2, G1)


    def check_group(self, group: DirectProductGroup, G1: Group, G2: Group):
        
        assert group.G1 == G1
        assert group.G2 == G2

        check_singleton(self, direct_product, G1, G2)
        check_generators(self, group)
        check_operations(self, group)
        check_irreps(self, group)
        check_regular_repr(self, group)
        

if __name__ == '__main__':
    unittest.main()
