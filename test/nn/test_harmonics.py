import unittest
from unittest import TestCase

from escnn.group import *
from escnn.nn import HarmonicPolynomialR3


class TestHarmonicPolynomialsR3(TestCase):
    
    def test_equivariance_so3(self):
        hp = HarmonicPolynomialR3(L=5, group='so3')
        self.assertEqual(hp.gspace.fibergroup, so3_group())
        hp.check_equivariance()

    def test_equivariance_o3(self):
        hp = HarmonicPolynomialR3(L=5, group='o3')
        self.assertEqual(hp.gspace.fibergroup, o3_group())
        hp.check_equivariance()


if __name__ == '__main__':
    unittest.main()
