import unittest
from unittest import TestCase

from escnn.nn import *
from escnn.gspaces import *
from escnn.group import *

import torch
import numpy as np

import random


class TestHarmonicPolynomials(TestCase):
    
    def test_r3(self):

        hp = HarmonicPolynomialR3(L=3)
        hp.check_equivariance()


if __name__ == '__main__':
    unittest.main()
