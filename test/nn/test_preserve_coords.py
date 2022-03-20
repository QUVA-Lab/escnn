import unittest
from unittest import TestCase

from escnn.nn import *
from escnn.gspaces import *

import torch
import random


class TestCoordsPreserved(TestCase):
    
    def test_dihedral_norm_relu(self):
        N = 8
        g = flipRot2dOnR2(N)
        
        r = FieldType(g, list(g.representations.values()) * 4)
        
        nnl = NormNonLinearity(r, function='n_relu')
        
        self.check_coords_preserved(nnl)
    
    def test_dihedral_norm_sigmoid(self):
        N = 8
        g = flipRot2dOnR2(N)
        
        r = FieldType(g, list(g.representations.values()) * 4)
        
        nnl = NormNonLinearity(r, function='n_sigmoid')
        
        self.check_coords_preserved(nnl)
    
    def test_dihedral_pointwise_relu(self):
        N = 8
        g = flipRot2dOnR2(N)
        
        reprs = [r for r in g.representations.values() if 'pointwise' in r.supported_nonlinearities]
        
        r = FieldType(g, reprs)
        
        nnl = PointwiseNonLinearity(r, function='p_relu')
        
        self.check_coords_preserved(nnl)
    
    def test_dihedral_pointwise_sigmoid(self):
        N = 8
        g = flipRot2dOnR2(N)
        
        reprs = [r for r in g.representations.values() if 'pointwise' in r.supported_nonlinearities]
        
        r = FieldType(g, reprs)
        
        nnl = PointwiseNonLinearity(r, function='p_sigmoid')
        
        self.check_coords_preserved(nnl)
    
    def test_dihedral_gated_one_input_shuffled_gated(self):
        N = 8
        g = flipRot2dOnR2(N)
        
        reprs = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities] * 3
        ngates = len(reprs)
        reprs += [g.trivial_repr] * ngates
        
        gates = ['gated'] * ngates + ['gate'] * ngates
        
        r = FieldType(g, reprs)
        
        nnl = GatedNonLinearity1(r, gates=gates)
        self.check_coords_preserved(nnl)
    
    def test_dihedral_gated_one_input_sorted_gated(self):
        N = 8
        g = flipRot2dOnR2(N)
        
        reprs = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities] * 3
        
        r = FieldType(g, reprs).sorted()
        
        ngates = len(r)
        
        reprs = [g.trivial_repr] * ngates
        
        gates = ['gated'] * ngates + ['gate'] * ngates
        
        r = r + FieldType(g, reprs)
        
        nnl = GatedNonLinearity1(r, gates=gates)
        self.check_coords_preserved(nnl)
    
    def test_dihedral_gated_one_input_all_shuffled(self):
        N = 8
        g = flipRot2dOnR2(N)
        
        reprs = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities] * 2
        
        ngates = len(reprs)
        
        reprs += [g.trivial_repr] * ngates
        
        gates = ['gated'] * ngates + ['gate'] * ngates
        
        t = list(zip(reprs, gates))
        
        random.shuffle(t)
        
        reprs, gates = zip(*t)
        
        r = FieldType(g, reprs)
        
        nnl = GatedNonLinearity1(r, gates=gates)
        self.check_coords_preserved(nnl)

    def test_dihedral_gated_two_inputs_shuffled_gated(self):
        N = 8
        g = flipRot2dOnR2(N)
    
        gated = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities] * 3
        ngates = len(gated)
        gates = [g.trivial_repr] * ngates

        gates = FieldType(g, gates)
        gated = FieldType(g, gated)
    
        nnl = GatedNonLinearity2((gates, gated))
        self.check_coords_preserved(nnl)

    def test_dihedral_gated_two_inputs_sorted_gated(self):
        N = 8
        g = flipRot2dOnR2(N)
    
        gated = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities] * 2
        ngates = len(gated)
        gates = [g.trivial_repr] * ngates
    
        gates = FieldType(g, gates)
        gated = FieldType(g, gated).sorted()
    
        nnl = GatedNonLinearity2((gates, gated))
        self.check_coords_preserved(nnl)

    def test_dihedral_concat_relu(self):
        N = 8
        g = flipRot2dOnR2(N)
        
        reprs = [r for r in g.representations.values() if 'concatenated' in r.supported_nonlinearities]
        
        for rep in reprs:
            r = FieldType(g, [rep])
            nnl = ConcatenatedNonLinearity(r, function='c_relu')
            self.check_coords_preserved(nnl)

    def test_dihedral_induced_norm_relu(self):
    
        N = 9
        g = flipRot2dOnR2(N)
    
        sg_id = (None, N)
        so2, _, _ = g.fibergroup.subgroup(sg_id)
        
        r = FieldType(g, [g.induced_repr(sg_id, so2.irrep(k)) for k in range(1, int(N // 2))] * 4).sorted()
        nnl = InducedNormNonLinearity(r, function='n_relu')
        self.check_coords_preserved(nnl)

    def test_cyclic_vectorfield(self):
        N = 8
        g = rot2dOnR2(N)
    
        reprs = [r for r in g.representations.values() if 'vectorfield' in r.supported_nonlinearities] * 8
    
        r = FieldType(g, reprs)
        nnl = VectorFieldNonLinearity(r)
        self.check_coords_preserved(nnl)

    def test_dihedral_indnormpool(self):
        N = 8
        g = flipRot2dOnR2(N)
        
        sgid = False, 4
        
        _g = g.restrict(sgid)[0]
    
        reprs = [
            g.induced_repr(sgid, rep)
            for rep in _g.irreps
        ]
    
        r = FieldType(g, reprs)
        nnl = InducedNormPool(r)
        self.check_coords_preserved(nnl)
        
    def test_cyclic_normpool(self):
        N = 8
        g = rot2dOnR2(N)

        reprs = list(g.representations.values())

        r = FieldType(g, reprs)
        nnl = NormPool(r)
        self.check_coords_preserved(nnl)

    def test_cyclic_gpool(self):
        N = 8
        g = rot2dOnR2(N)
    
        reprs = [g.regular_repr]*3
    
        r = FieldType(g, reprs)
        nnl = GroupPooling(r)
        self.check_coords_preserved(nnl)

    def test_cyclic_restriction(self):
        N = 8
        g = rot2dOnR2(N)
    
        reprs = list(g.representations.values())
    
        r = FieldType(g, reprs)
        nnl = RestrictionModule(r, 2)
        self.check_coords_preserved(nnl)

    def test_cyclic_reshuffle(self):
        N = 8
        g = rot2dOnR2(N)
    
        reprs = list(g.representations.values())
    
        r = FieldType(g, reprs)
        nnl = ReshuffleModule(r, list(range(len(reprs))))
        self.check_coords_preserved(nnl)

    def test_cyclic_disentangle(self):
        N = 8
        g = rot2dOnR2(N)
    
        reprs = list(g.representations.values())
    
        r = FieldType(g, reprs)
        nnl = DisentangleModule(r)
        self.check_coords_preserved(nnl)

    def test_cyclic_sequential(self):
        N = 8
        g = rot2dOnR2(N)
    
        reprs = [g.regular_repr, g.trivial_repr]*2
    
        r = FieldType(g, reprs)
        nnl = SequentialModule(
            PointwiseDropout(r),
            ReLU(r),
            GroupPooling(r),
        )
        self.check_coords_preserved(nnl)

    def check_coords_preserved(self, module: EquivariantModule):
    
        B = 3
        P = 9
        
        double = not isinstance(module.in_type, FieldType)

        if not double:
            in_type = module.in_type
        else:
            in_type = module.in_type[0]

        pos = torch.randn(P, in_type.gspace.dimensionality)
        x = torch.randn(P, in_type.size)
        x = GeometricTensor(x, in_type, pos)
        
        if double:
            g = torch.randn(P, module.in_type[1].size)
            g = GeometricTensor(g, module.in_type[1], pos)
            
            y = module(x, g)
        else:
            y = module(x)

        self.assertTrue(x.coords is y.coords)
        

if __name__ == '__main__':
    unittest.main()
