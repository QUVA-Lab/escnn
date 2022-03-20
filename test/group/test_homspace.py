import unittest
from unittest import TestCase

from escnn.group import *


class TestHomSpace(TestCase):
    
    def test_SO3(self):
        G = so3_group(3)
        G.homspace((False, -1))._unit_test_basis()
        G.homspace((False, -1))._unit_test_full_basis()
        G.homspace((True, -1))._unit_test_basis()
        G.homspace((True, -1))._unit_test_full_basis()
        G.homspace((False, 7))._unit_test_basis()
        G.homspace((False, 7))._unit_test_full_basis()
        G.homspace((False, 4))._unit_test_basis()
        G.homspace((False, 4))._unit_test_full_basis()
        G.homspace((True, 7))._unit_test_basis()
        G.homspace((True, 7))._unit_test_full_basis()
        G.homspace((True, 4))._unit_test_basis()
        G.homspace((True, 4))._unit_test_full_basis()
        G.homspace((False, 1))._unit_test_basis()
        G.homspace((False, 1))._unit_test_full_basis()
        G.homspace((True, 1))._unit_test_basis()
        G.homspace((True, 1))._unit_test_full_basis()
        G.homspace('ico')._unit_test_basis()
        G.homspace('ico')._unit_test_full_basis()

    def test_O3(self):
        G = o3_group(3)
        G.homspace('so3')._unit_test_basis()
        G.homspace('so3')._unit_test_full_basis()
        G.homspace((False, False, -1))._unit_test_basis()
        G.homspace((False, False, -1))._unit_test_full_basis()
        G.homspace((False, True, -1))._unit_test_basis()
        G.homspace((False, True, -1))._unit_test_full_basis()
        G.homspace((False, False, 7))._unit_test_basis()
        G.homspace((False, False, 7))._unit_test_full_basis()
        G.homspace((False, False, 4))._unit_test_basis()
        G.homspace((False, False, 4))._unit_test_full_basis()
        G.homspace((False, True, 7))._unit_test_basis()
        G.homspace((False, True, 7))._unit_test_full_basis()
        G.homspace((False, True, 4))._unit_test_basis()
        G.homspace((False, True, 4))._unit_test_full_basis()
        G.homspace((False, False, 1))._unit_test_basis()
        G.homspace((False, False, 1))._unit_test_full_basis()
        G.homspace((False, True, 1))._unit_test_basis()
        G.homspace((False, True, 1))._unit_test_full_basis()
        G.homspace((False, 'ico'))._unit_test_basis()
        G.homspace((False, 'ico'))._unit_test_full_basis()

        G.homspace((True, False, 1))._unit_test_basis()
        G.homspace((True, False, 1))._unit_test_full_basis()
        G.homspace((True, True, -1))._unit_test_basis()
        G.homspace((True, True, -1))._unit_test_full_basis()
        G.homspace((True, True, 1))._unit_test_basis()
        G.homspace((True, True, 1))._unit_test_full_basis()
        G.homspace((True, True, 4))._unit_test_basis()
        G.homspace((True, True, 4))._unit_test_full_basis()
        G.homspace((True, True, 7))._unit_test_basis()
        G.homspace((True, True, 7))._unit_test_full_basis()
        G.homspace((True, 'ico'))._unit_test_basis()
        G.homspace((True, 'ico'))._unit_test_full_basis()

        # not implemented irrep restriction yet
        # G.homspace(('cone', -1))._unit_test_basis()
        # G.homspace(('cone', -1))._unit_test_full_basis()()
        # G.homspace(('cone', 4))._unit_test_basis()
        # G.homspace(('cone', 4))._unit_test_full_basis()()
        # G.homspace(('cone', 1))._unit_test_basis()
        # G.homspace(('cone', 1))._unit_test_full_basis()()

    def test_SO2(self):
        G = so2_group(3)
        G.homspace(1)._unit_test_basis()
        G.homspace(1)._unit_test_full_basis()
        G.homspace(2)._unit_test_basis()
        G.homspace(2)._unit_test_full_basis()
        G.homspace(3)._unit_test_basis()
        G.homspace(3)._unit_test_full_basis()
        G.homspace(8)._unit_test_basis()
        G.homspace(8)._unit_test_full_basis()

    def test_O2(self):
        G = o2_group(3)
        G.homspace((None, 1))._unit_test_basis()
        G.homspace((None, 1))._unit_test_full_basis()
        G.homspace((None, 2))._unit_test_basis()
        G.homspace((None, 2))._unit_test_full_basis()
        G.homspace((None, 3))._unit_test_basis()
        G.homspace((None, 3))._unit_test_full_basis()
        G.homspace((None, 8))._unit_test_basis()
        G.homspace((None, 8))._unit_test_full_basis()
        G.homspace((0., 1))._unit_test_basis()
        G.homspace((0., 1))._unit_test_full_basis()
        G.homspace((0., 2))._unit_test_basis()
        G.homspace((0., 2))._unit_test_full_basis()
        G.homspace((0., 3))._unit_test_basis()
        G.homspace((0., 3))._unit_test_full_basis()
        G.homspace((0., 8))._unit_test_basis()
        G.homspace((0., 8))._unit_test_full_basis()

    def test_cyclicgroup(self):
        for n in [2, 3, 4, 5, 8, 12, 15, 20]:
            for j in range(3, n):
                if n % j == 0:
                    cyclic_group(n).homspace(j)._unit_test_basis()
                    cyclic_group(n).homspace(j)._unit_test_full_basis()

    def test_dihedralgroup(self):
        for n in [2, 3, 4, 5, 8, 12, 15, 20]:
            for j in range(3, n):
                if n % j == 0:
                    cyclic_group(n).homspace(j)._unit_test_basis()
                    cyclic_group(n).homspace(j)._unit_test_full_basis()


if __name__ == '__main__':
    unittest.main()
