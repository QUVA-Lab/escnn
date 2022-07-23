import unittest
from unittest import TestCase

from escnn.nn import *
from escnn.gspaces import *

import torch
import random


class TestSequential(TestCase):

    def test_iter(self):
        N = 8
        g = flipRot2dOnR2(N)

        r = g.type(*[g.regular_repr] * 3)

        modules = [
            R2Conv(r, r, 3, bias=False, initialize=False),
            ReLU(r),
            R2Conv(r, r, 3, bias=False, initialize=False),
            ReLU(r),
            R2Conv(r, r, 3, bias=False, initialize=False),
        ]
        module = SequentialModule(*modules)

        self.assertEquals(len(modules), len(module))

        for i, module in enumerate(module):
            self.assertEquals(module, modules[i])

    def test_get_item(self):
        N = 8
        g = flipRot2dOnR2(N)
        
        r = g.type(*[g.regular_repr]*3)

        modules = [
            R2Conv(r, r, 3, bias=False, initialize=False),
            ReLU(r),
            R2Conv(r, r, 3, bias=False, initialize=False),
            ReLU(r),
            R2Conv(r, r, 3, bias=False, initialize=False),
        ]
        module = SequentialModule(*modules)

        self.assertEquals(len(modules), len(module))

        for i in range(len(module)):
            self.assertEquals(module[i], modules[i])

if __name__ == '__main__':
    unittest.main()
