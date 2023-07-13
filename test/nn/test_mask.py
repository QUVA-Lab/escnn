import unittest
import torch

from unittest import TestCase
from escnn.gspaces import rot2dOnR2, rot3dOnR3
from escnn.nn import FieldType, GeometricTensor, MaskModule


class TestMask(TestCase):

    def test_mask_module_r2(self):
        gspace = rot2dOnR2()
        in_type = FieldType(gspace, [gspace.trivial_repr])

        # Use a very low standard deviation to make the transition between 
        # masked/unmasked more dramatic.
        mask = MaskModule(in_type, 5, margin=0, sigma=0.1)

        x = GeometricTensor(
                torch.ones(1, 1, 5, 5),
                in_type,
        )
        y = mask(x)

        unmasked_indices = [
                (1, 1),
                (1, 3),
                (3, 1),
                (3, 3),
        ]
        masked_indices = [
                (0, 0),
                (0, 4),
                (4, 0),
                (4, 4),
        ]

        assert y.type == in_type

        for i,j in unmasked_indices:
            self.assertGreater(y.tensor[0, 0, i, j], 1 - 1e5)
        for i,j in masked_indices:
            self.assertLess(y.tensor[0, 0, i, j], 1e-5)

    def test_mask_module_r3(self):
        gspace = rot3dOnR3()
        in_type = FieldType(gspace, [gspace.trivial_repr])

        # Use a very low standard deviation to make the transition between 
        # masked/unmasked more dramatic.
        mask = MaskModule(in_type, 5, margin=0, sigma=0.1)

        x = GeometricTensor(
                torch.ones(1, 1, 5, 5, 5),
                in_type,
        )
        y = mask(x)

        unmasked_indices = [
                (1, 1, 1),
                (1, 1, 3),
                (1, 3, 1),
                (1, 3, 3),
                (3, 1, 1),
                (3, 1, 3),
                (3, 3, 1),
                (3, 3, 3),
        ]
        masked_indices = [
                (0, 0, 0),
                (0, 0, 4),
                (0, 4, 0),
                (0, 4, 4),
                (4, 0, 0),
                (4, 0, 4),
                (4, 4, 0),
                (4, 4, 4),
        ]

        assert y.type == in_type

        for i,j,k in unmasked_indices:
            self.assertGreater(y.tensor[0, 0, i, j, k], 1 - 1e5)
        for i,j,k in masked_indices:
            self.assertLess(y.tensor[0, 0, i, j, k], 1e-5)

    def test_equivariance_r3(self):
        gspace = rot3dOnR3()
        in_type = FieldType(gspace, [gspace.trivial_repr])

        S = 17
        mask = MaskModule(in_type, S, margin=2, sigma=2.)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        mask.check_equivariance(device=device)

    def test_equivariance_r2(self):
        gspace = rot2dOnR2()
        in_type = FieldType(gspace, [gspace.trivial_repr])

        S = 17
        mask = MaskModule(in_type, S, margin=2, sigma=2.)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        mask.check_equivariance(device=device)


if __name__ == '__main__':
    unittest.main()




