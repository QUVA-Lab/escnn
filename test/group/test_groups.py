import unittest
from unittest import TestCase

from escnn.group import *

from escnn.group.utils import cycle_isclose

import math

import numpy as np


class TestGroups(TestCase):
    
    def test_cyclic_odd(self):
        g = CyclicGroup(15)
        self.check_group(g)

    def test_cyclic_even(self):
        g = CyclicGroup(16)
        self.check_group(g)

    def test_dihedral_odd(self):
        g = DihedralGroup(15)
        self.check_group(g)

    def test_dihedral_even(self):
        g = DihedralGroup(16)
        self.check_group(g)

    def test_so2(self):
        g = SO2(4)
        self.check_group(g)

    def test_o2(self):
        g = O2(4)
        self.check_group(g)

    def test_so3(self):
        g = SO3(5)
        self.check_group(g)
        
    def test_o3(self):
        g = O3(5)
        self.check_group(g)

    def test_ico(self):
        g = Icosahedral()
        self.check_group(g)

    def test_octa(self):
        g = Octahedral()
        self.check_group(g)

        # Check that the `_is_element` method of Octahedral() is consistent with the `is_element` method based
        # on the rotation order and the rotation axis

        from escnn.group.groups.octa import _is_axis_aligned

        def is_element(q):
            theta = 2 * np.arccos(np.clip(q[3], -1., 1.))

            ans_axis = False
            if cycle_isclose(theta, 0., 2 * np.pi, atol=1e-6, rtol=0.):
                ans_axis = True
            elif cycle_isclose(theta, 0., 2 * np.pi / 2, atol=1e-6, rtol=0.):
                ans_axis = _is_axis_aligned(q[:3], 2)
            elif cycle_isclose(theta, 0., 2 * np.pi / 4, atol=1e-6, rtol=0.):
                ans_axis = _is_axis_aligned(q[:3], 4)
            elif cycle_isclose(theta, 0., 2 * np.pi / 3, atol=1e-6, rtol=0.):
                ans_axis = _is_axis_aligned(q[:3], 3)
            return ans_axis

        for e in g.elements:
            assert g._is_element(e.value, e.param)
            assert is_element(e.to('Q'))

        so3 = so3_group(1)
        for e in so3.grid('rand', 30):
            assert g._is_element(e.to('Q'), 'Q') == is_element(e.to('Q'))

    def check_group(self, group: Group):
        
        self.check_generators(group)
        
        e = group.identity
        
        for a in group.testing_elements():
            
            self.assertTrue(a @ e == a)
            self.assertTrue(e @ a == a)
            
            i = ~a
            self.assertTrue(a @ i, e)
            self.assertTrue(i @ a, e)
            
            for b in group.testing_elements():
                for c in group.testing_elements():
    
                    ab = a @ b
                    bc = b @ c
                    a_bc = a @ bc
                    ab_c = ab @ c
                    self.assertTrue(a_bc == ab_c, f"{a_bc} != {ab_c}")

    def check_generators(self, group: Group):
        if group.order() > 0:
            generators = group.generators
            self.assertTrue(len(generators) > 0)
        else:
            with self.assertRaises(ValueError):
                generators = group.generators
            return
    
        identity = group.identity
        
        added = set()
        elements = set()
    
        added.add(identity)
        elements.add(identity)
        
        while len(added) > 0:
            new = set()
            for g in generators:
                for e in added:
                    new |= {g @ e, ~g @ e}
            added = new - elements
            elements |= added
    
        self.assertTrue(
            len(elements) == group.order(),
            'Error! The set of generators does not generate the whole group'
        )
    
        for a in elements:
            self.assertIn(~a, elements)
            for b in elements:
                self.assertIn(a @ b, elements)


if __name__ == '__main__':
    unittest.main()
