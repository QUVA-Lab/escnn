import unittest
from unittest import TestCase

from escnn.gspaces import *
from escnn.group import *
from escnn.nn import *
from escnn.nn.modules.basismanager import BlocksBasisExpansion

import torch


class TestBasisExpansion(TestCase):
    
    def test_linear(self):
        gspaces = [
            no_base_space(G)
            for G in [
                cyclic_group(1),
                cyclic_group(4),
                dihedral_group(3),
                so2_group(2),
                o2_group(2),
                so3_group(2),
                o3_group(2),
                ico_group(),
            ]
        ]
        
        for gspace in gspaces:

            reprs = gspace.fibergroup.irreps() if gspace.fibergroup.order() > 0 else [gspace.fibergroup.irrep(*irr) for irr in gspace.fibergroup.bl_irreps(4)]

            try:
                reg = gspace.regular_repr
                reprs = [reg] + reprs
            except ValueError:
                pass
            
            for i in range(len(reprs)-1):
                for j in range(len(reprs)-1):
                    t1 = reprs[:i+1]
                    t2 = reprs[:j+1]
                    
                    t1 = FieldType(gspace, t1)
                    t2 = FieldType(gspace, t2)
                    
                    layer = Linear(t1, t2, bias=False, initialize=False)
                    self.compare(layer.basisexpansion)

    def test_conv2d(self):
        gspaces = [
            rot2dOnR2(4),
            flipRot2dOnR2(4),
            flipRot2dOnR2(-1),
        ]
    
        for gspace in gspaces:

            reprs = gspace.fibergroup.irreps() if gspace.fibergroup.order() > 0 else [gspace.fibergroup.irrep(*irr) for irr in gspace.fibergroup.bl_irreps(4)]

            try:
                reg = gspace.regular_repr
                reprs = [reg] + reprs
            except ValueError:
                pass

            print(gspace, len(reprs))
            
            for i in range(len(reprs) - 1):
                for j in range(len(reprs) - 1):
                    t1 = reprs[:i + 1]
                    t2 = reprs[:j + 1]
                
                    t1 = FieldType(gspace, t1)
                    t2 = FieldType(gspace, t2)
                
                    layer = R2Conv(t1, t2, 5, bias=False, initialize=False)
                    self.compare(layer.basisexpansion)

    def test_conv3d(self):
        gspaces = [
            flipRot3dOnR3(),
            rot3dOnR3(),
            # # fullIcoOnR3(),
            icoOnR3(),
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

            reprs = gspace.fibergroup.irreps() if gspace.fibergroup.order() > 0 else [gspace.fibergroup.irrep(*irr) for irr in gspace.fibergroup.bl_irreps(4)]

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
                
                    layer = R3Conv(t1, t2, 5, bias=False, initialize=False)
                    self.compare(layer.basisexpansion)

    def test_many_block_discontinuous(self):
        gspace = rot3dOnR3()
        reprs = [gspace.irrep(*irr) for irr in gspace.fibergroup.bl_irreps(3)]
        t1 = t2 = FieldType(gspace, reprs * 4)
        layer = R3Conv(t1, t2, 5, bias=False, initialize=False)
        self.compare(layer.basisexpansion)

    def test_many_block_sorted(self):
        gspace = rot3dOnR3()
        reprs = [gspace.irrep(*irr) for irr in gspace.fibergroup.bl_irreps(3)]
        t1 = t2 = FieldType(gspace, reprs * 4).sorted()
        layer = R3Conv(t1, t2, 5, bias=False, initialize=False)
        self.compare(layer.basisexpansion)

    def compare(self, basis: BlocksBasisExpansion):
        
        for i, attr1 in enumerate(basis.get_basis_info()):
            attr2 = basis.get_element_info(i)
            self.assertEquals(attr1, attr2)
            self.assertEquals(attr1['id'], i)
        
        for _ in range(5):
            w = torch.randn(basis.dimension())
            
            f1 = basis(w)
            f2 = basis(w)
            assert torch.allclose(f1, f2)
            self.assertEquals(f1.shape[1], basis._input_size)
            self.assertEquals(f1.shape[0], basis._output_size)


if __name__ == '__main__':
    unittest.main()
