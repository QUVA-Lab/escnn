import unittest
from unittest import TestCase

from .utils import *

from escnn.group import *
import numpy as np


class TestDoubleGroup(TestCase):
    
    def test_trivial(self):
        H = trivial_group()
        G = double_group(H)
        self.check_group(G, H)
        
    def test_cn(self):
        for n in [1, 2, 3, 5]:
            G1 = cyclic_group(n)
            G = double_group(G1)
            self.check_group(G, G1)

    def test_dn(self):
        for n in [1, 2, 3, 4]:
            G1 = dihedral_group(n)
            G = double_group(G1)
            self.check_group(G, G1)

    def test_so2(self):
        G1 = so2_group(2)
        G = double_group(G1)
        self.check_group(G, G1)

    def test_o2(self):
        G1 = o2_group(2)
        G = double_group(G1)
        self.check_group(G, G1)

    def test_so3(self):
        G1 = so3_group(2)
        G = double_group(G1)
        self.check_group(G, G1)


    def check_group(self, group: DoubleGroup, G1: Group):
        
        assert group.G1 == G1
        assert group.G2 == G1
        
        check_singleton(self, double_group, G1)
        check_generators(self, group)
        check_operations(self, group)
        check_irreps(self, group)
        check_regular_repr(self, group)

        

if __name__ == '__main__':
    unittest.main()
