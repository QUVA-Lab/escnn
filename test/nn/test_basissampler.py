import unittest
from unittest import TestCase

import numpy as np

from escnn.gspaces import *
from escnn.nn import *
from escnn.nn.modules.basismanager import BlocksBasisSampler

import torch


class TestBasisSampler(TestCase):
    
    def test_conv2d(self):
        gspaces = [
            rot2dOnR2(4),
            flipRot2dOnR2(4),
            flipRot2dOnR2(-1),
        ]
    
        for gspace in gspaces:

            reprs = gspace.irreps[:4]

            try:
                reg = gspace.regular_repr
                reprs = [reg] + reprs
            except ValueError:
                pass

            for i in range(len(reprs) - 1):
                for j in range(len(reprs) - 1):

                    t1 = reprs[:i + 1]
                    t2 = reprs[:j + 1]

                    t1 = FieldType(gspace, t1)
                    t2 = FieldType(gspace, t2)

                    sigma = None
                    fco = 2.
                    layer = R2PointConv(t1, t2,
                                     sigma=sigma,
                                     width=2.,
                                     n_rings=3,
                                     frequencies_cutoff=fco,
                                     bias=False)
                    self.compare(layer.basissampler, d=2)

    def test_conv3d(self):
        gspaces = [
            flipRot3dOnR3(),
            rot3dOnR3(),
            # # fullIcoOnR3(),
            # icoOnR3(),
            octaOnR3(),
            dihedralOnR3(),
            rot2dOnR3(),
            conicalOnR3(),
            # fullCylindricalOnR3(),
            # cylindricalOnR3(),
            mirOnR3(),
            invOnR3(),
            trivialOnR3(),
        ]
    
        for gspace in gspaces:
        
            reprs = gspace.irreps[:4]
        
            try:
                reg = gspace.regular_repr
                reprs = [reg] + reprs
            except ValueError:
                pass
            
            for i in range(len(reprs) - 1):
                for j in range(len(reprs) - 1):
                    print(gspace, len(reprs))
                    t1 = reprs[:i + 1]
                    t2 = reprs[:j + 1]
                
                    t1 = FieldType(gspace, t1)
                    t2 = FieldType(gspace, t2)

                    sigma = None
                    fco = 2.
                    layer = R3PointConv(t1, t2,
                                        sigma=sigma,
                                        width=2.,
                                        n_rings=3,
                                        frequencies_cutoff=fco,
                                        bias=False)
                    self.compare(layer.basissampler, d=3)

    def test_many_block_discontinuous(self):
        gspace = rot3dOnR3()
        t1 = FieldType(gspace, list(gspace.representations.values()) * 4)
        t2 = FieldType(gspace, list(gspace.representations.values()) * 4)
        sigma = None
        fco = 2.
        layer = R3PointConv(t1, t2,
                            sigma=sigma,
                            width=2.,
                            n_rings=3,
                            frequencies_cutoff=fco,
                            bias=False)
        self.compare(layer.basissampler, d=3)

    def test_many_block_sorted(self):
        gspace = rot3dOnR3()
        t1 = FieldType(gspace, list(gspace.representations.values()) * 4).sorted()
        t2 = FieldType(gspace, list(gspace.representations.values()) * 4).sorted()
        sigma = None
        fco = 2.
        layer = R3PointConv(t1, t2,
                            sigma=sigma,
                            width=2.,
                            n_rings=3,
                            frequencies_cutoff=fco,
                            bias=False)
        self.compare(layer.basissampler, d=3)

    def compare(self, basis: BlocksBasisSampler, d: int):

        for i, attr1 in enumerate(basis.get_basis_info()):
            attr2 = basis.get_element_info(i)
            self.assertEquals(attr1, attr2)
            self.assertEquals(attr1['id'], i)

        for _ in range(5):
            P = 20
            pos = torch.randn(P, d)
            x = torch.randn(P, basis._input_size)

            distance = torch.norm(pos.unsqueeze(1) - pos, dim=2, keepdim=False)
            thr = sorted(distance.view(-1).tolist())[
                int(P ** 2 // 16)
            ]
            edge_index = torch.nonzero(distance < thr).T.contiguous()

            row, cols = edge_index
            edge_delta = pos[row] - pos[cols]

            x_j = x[cols]

            w = torch.randn(basis.dimension())

            f1 = basis(w, edge_delta)
            f2 = basis(w, edge_delta)
            self.assertTrue(torch.allclose(f1, f2))
            self.assertEquals(f1.shape[2], basis._input_size)
            self.assertEquals(f1.shape[1], basis._output_size)

            y1 = basis.compute_messages(w, x_j, edge_delta, conv_first=False)
            y2 = basis.compute_messages(w, x_j, edge_delta, conv_first=True)

            np.set_printoptions(precision=7, suppress=True, linewidth=100000000000, threshold=10000000)
            self.assertTrue(
                torch.allclose(y1, y2, atol=1e-5, rtol=1e-5),
                f"Error: outputs do not match!\n"
                f"\t{basis._in_reprs}\n"
                f"\t{basis._out_reprs}\n"
                "Max Abs Error\n"
                f"{torch.max(torch.abs(y1-y2)).item()}\n"
            )


if __name__ == '__main__':
    unittest.main()
