import unittest
from unittest import TestCase

from escnn.group import *

import numpy as np
import math


class TestGroups(TestCase):
    
    def test_cyclic_odd(self):
        g = CyclicGroup(15)
        self.check_parametrization(g)

    def test_cyclic_even(self):
        g = CyclicGroup(16)
        self.check_parametrization(g)

    def test_dihedral_odd(self):
        g = DihedralGroup(15)
        self.check_parametrization(g)

    def test_dihedral_even(self):
        g = DihedralGroup(16)
        self.check_parametrization(g)

    def test_so2(self):
        g = SO2(4)
        self.check_parametrization(g)

    def test_o2(self):
        g = O2(4)
        self.check_parametrization(g)

    def test_so3(self):
        g = SO3(5)
        self.check_parametrization(g)
        
    def test_o3(self):
        g = O3(5)
        self.check_parametrization(g)

    def test_ico(self):
        g = Icosahedral()
        self.check_parametrization(g)

    def test_octa(self):
        g = Octahedral()
        self.check_parametrization(g)

    # Check some direct products

    def test_c2xso2(self):
        g = cylinder_group(3)
        self.check_parametrization(g)

    def test_c2xo2(self):
        g = full_cylinder_group(3)
        self.check_parametrization(g)

    def test_c2xcn(self):
        for n in [3, 4, 7]:
            g = cylinder_discrete_group(n)
            self.check_parametrization(g)

    def test_c2xdn(self):
        for n in [3, 4, 7]:
            g = full_cylinder_discrete_group(n)
            self.check_parametrization(g)

    ####################################################################################################################

    def check_parametrization(self, group: Group):
        
        if group.order() > 0:
            elements = group.testing_elements()
        else:
            elements = [group.sample() for _ in range(400)]
    
        for p1 in group.PARAMETRIZATIONS:
            for g in elements:
                
                g1 = g.to(p1)
                h = group.element(g1, p1)
                self.assertTrue(g == h, f'[{p1}]: {g1} | {g} != {h} ')


if __name__ == '__main__':
    unittest.main()
