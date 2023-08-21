import unittest
from unittest import TestCase
from utils import *

from escnn.group import *
from escnn.group.utils import cycle_isclose

import numpy as np


class TestGroups(TestCase):
    
    def test_cyclic_odd(self):
        self.check_group(CyclicGroup, 15)

    def test_cyclic_even(self):
        self.check_group(CyclicGroup, 16)

    def test_dihedral_odd(self):
        self.check_group(DihedralGroup, 15)

    def test_dihedral_even(self):
        self.check_group(DihedralGroup, 16)

    def test_so2(self):
        self.check_group(SO2, 4)

    def test_o2(self):
        self.check_group(O2, 4)

    def test_so3(self):
        self.check_group(SO3, 5)
        
    def test_o3(self):
        self.check_group(O3, 5)

    def test_ico(self):
        self.check_group(Icosahedral)

    def test_octa(self):
        g = self.check_group(Octahedral)

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


    def check_group(self, factory, *args, **kwargs):
        g = check_singleton(self, factory, *args, **kwargs)
        check_generators(self, g)
        check_operations(self, g)
        return g



if __name__ == '__main__':
    unittest.main()
