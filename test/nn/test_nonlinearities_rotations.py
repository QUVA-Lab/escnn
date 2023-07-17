import unittest
from unittest import TestCase

from escnn.group import *
from escnn.nn import *
from escnn.gspaces import *

import torch
import numpy as np

import random


class TestNonLinearitiesRotations(TestCase):
    
    def test_cyclic_norm_relu(self):
        N = 8
        g = rot2dOnR2(N)
        
        r = FieldType(g, list(g.representations.values()) * 4)
        
        nnl = NormNonLinearity(r, function='n_relu')

        nnl.check_equivariance()
    
    def test_cyclic_norm_sigmoid(self):
        N = 8
        g = rot2dOnR2(N)
        
        r = FieldType(g, list(g.representations.values()) * 4)
        
        nnl = NormNonLinearity(r, function='n_sigmoid')
        
        nnl.check_equivariance()
    
    def test_cyclic_pointwise_relu(self):
        N = 8
        g = rot2dOnR2(N)
        
        reprs = [r for r in g.representations.values() if 'pointwise' in r.supported_nonlinearities]
        
        r = FieldType(g, reprs)
        
        # nnl = PointwiseNonLinearity(r, function='p_relu')
        nnl = ReLU(r)
        
        nnl.check_equivariance()
    
    def test_cyclic_pointwise_sigmoid(self):
        N = 8
        g = rot2dOnR2(N)
        
        reprs = [r for r in g.representations.values() if 'pointwise' in r.supported_nonlinearities]
        
        r = FieldType(g, reprs)
        
        nnl = PointwiseNonLinearity(r, function='p_sigmoid')
        
        nnl.check_equivariance()

    def test_cyclic_pointwise_leakyrelu(self):
        N = 8
        g = rot2dOnR2(N)

        reprs = [r for r in g.representations.values() if 'pointwise' in r.supported_nonlinearities]

        r = FieldType(g, reprs)

        nnl = LeakyReLU(r)

        nnl.check_equivariance()

    def test_cyclic_gated_uniform_sigmoid(self):
        N = 8
        g = rot2dOnR2(N)
        C = 10

        for repr in g.representations.values():

            if np.allclose(repr.change_of_basis, np.eye(repr.size)):
                ngates = len(repr.irreps)
                repr_and_gates = directsum([g.trivial_repr]*ngates) + repr
                in_type = g.type(*[repr_and_gates]*C)
                nnl = GatedNonLinearityUniform(in_type)

                self.assertEqual(nnl.out_type.size, C*repr.size)
                self.assertTrue(nnl.out_type.uniform)
                self.assertEqual(nnl.out_type.representations[0].irreps, repr.irreps)
                self.assertTrue(np.allclose(nnl.out_type.representations[0].change_of_basis, repr.change_of_basis))

                nnl.check_equivariance()
            else:
                print('Change of basis non-supported')
                print(repr)

    def test_cyclic_gated_uniform_swish(self):
        N = 8
        g = rot2dOnR2(N)
        C = 10

        representations = list(g.representations.values())

        representations += [
            directsum([g.irrep(l) for l in range(3)]),
            directsum([g.irrep(l) for l in range(3)]*2),
            directsum([g.irrep(0), g.irrep(1), g.irrep(1), g.irrep(2), g.irrep(0)]),
        ]

        for repr in representations:

            if np.allclose(repr.change_of_basis, np.eye(repr.size)):
                ngates = len(repr.irreps)
                repr_and_gates = directsum([g.trivial_repr]*ngates) + repr
                in_type = g.type(*[repr_and_gates]*C)
                nnl = GatedNonLinearityUniform(in_type, gate=torch.nn.functional.silu)

                self.assertEqual(nnl.out_type.size, C*repr.size)
                self.assertTrue(nnl.out_type.uniform)
                self.assertEqual(nnl.out_type.representations[0].irreps, repr.irreps)
                self.assertTrue(np.allclose(nnl.out_type.representations[0].change_of_basis, repr.change_of_basis))

                nnl.check_equivariance()
            else:
                print('Change of basis non-supported')
                print(repr)

    def test_o2_gated_uniform_swish(self):
        g = flipRot2dOnR2(-1, 4)
        C = 10

        representations = list(g.representations.values())

        representations += [
            directsum([g.irrep(1, l) for l in range(3)]),
            directsum([g.irrep(1, l) for l in range(3)]*2),
            directsum([g.irrep(0, 0), g.irrep(1, 1), g.irrep(1, 1), g.irrep(1, 2), g.irrep(1, 0), g.irrep(0, 0)]),
        ]

        for repr in representations:

            if np.allclose(repr.change_of_basis, np.eye(repr.size)):
                ngates = len(repr.irreps)
                repr_and_gates = directsum([g.trivial_repr]*ngates) + repr
                in_type = g.type(*[repr_and_gates]*C)
                nnl = GatedNonLinearityUniform(in_type, gate=torch.nn.functional.silu)

                self.assertEqual(nnl.out_type.size, C*repr.size)
                self.assertTrue(nnl.out_type.uniform)
                self.assertEqual(nnl.out_type.representations[0].irreps, repr.irreps)
                self.assertTrue(np.allclose(nnl.out_type.representations[0].change_of_basis, repr.change_of_basis))

                nnl.check_equivariance()
            else:
                print('Change of basis non-supported')
                print(repr)

    def test_cyclic_gated_one_input_shuffled_gated(self):
        N = 8
        g = rot2dOnR2(N)
        
        reprs = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities]
        ngates = len(reprs)
        reprs += [g.trivial_repr] * ngates
        
        gates = ['gated'] * ngates + ['gate'] * ngates
        
        r = FieldType(g, reprs)
        
        nnl = GatedNonLinearity1(r, gates=gates)
        nnl.check_equivariance()
    
    def test_cyclic_gated_one_input_sorted_gated(self):
        N = 8
        g = rot2dOnR2(N)
        
        reprs = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities]
        
        r = FieldType(g, reprs).sorted()
        
        ngates = len(r)
        
        reprs = [g.trivial_repr] * ngates
        
        gates = ['gated'] * ngates + ['gate'] * ngates
        
        r = r + FieldType(g, reprs)
        
        nnl = GatedNonLinearity1(r, gates=gates)
        nnl.check_equivariance()
    
    def test_cyclic_gated_one_input_all_shuffled(self):
        N = 8
        g = rot2dOnR2(N)
        
        reprs = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities]
        
        ngates = len(reprs)
        
        reprs += [g.trivial_repr] * ngates
        
        gates = ['gated'] * ngates + ['gate'] * ngates
        
        t = list(zip(reprs, gates))
        
        random.shuffle(t)
        
        reprs, gates = zip(*t)
        
        r = FieldType(g, reprs)
        
        nnl = GatedNonLinearity1(r, gates=gates)
        nnl.check_equivariance()

    def test_cyclic_gated_two_inputs_shuffled_gated(self):
        N = 8
        g = rot2dOnR2(N)
    
        gated = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities]
        ngates = len(gated)
        gates = [g.trivial_repr] * ngates
    
        gates = FieldType(g, gates)
        gated = FieldType(g, gated)
    
        nnl = GatedNonLinearity2((gates, gated))
        nnl.check_equivariance()

    def test_cyclic_gated_two_inputs_sorted_gated(self):
        N = 8
        g = rot2dOnR2(N)
    
        gated = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities]

        gated = FieldType(g, gated).sorted()
    
        ngates = len(gated)
    
        gates = [g.trivial_repr] * ngates
        gates = FieldType(g, gates)
    
        nnl = GatedNonLinearity2((gates, gated))
        nnl.check_equivariance()

    def test_cyclic_concat_relu(self):
        N = 8
        g = rot2dOnR2(N)
    
        reprs = [r for r in g.representations.values() if 'concatenated' in r.supported_nonlinearities]
    
        for rep in reprs:
            print(rep.name)
            r = FieldType(g, [rep])
            nnl = ConcatenatedNonLinearity(r, function='c_relu')
            nnl.check_equivariance()

    def test_cyclic_vectorfield(self):
        N = 8
        g = rot2dOnR2(N)
    
        reprs = [r for r in g.representations.values() if 'vectorfield' in r.supported_nonlinearities] * 8
    
        r = FieldType(g, reprs)
        nnl = VectorFieldNonLinearity(r)
        nnl.check_equivariance(atol=2e-6)

    def test_cyclic_induced_norm_relu(self):
    
        N = 15
        g = rot2dOnR2(N)
    
        sg_id = 5
        sg, _, _ = g.fibergroup.subgroup(sg_id)
    
        r = FieldType(g, [g.induced_repr(sg_id, sg.irrep(k)) for k in range(1, int(sg.order() // 2))] * 4).sorted()
        nnl = InducedNormNonLinearity(r, function='n_relu')
        nnl.check_equivariance()

    def test_so2_norm_relu(self):
        
        g = rot2dOnR2(-1, 10)
    
        r = FieldType(g, list(g.representations.values()) * 4)
    
        nnl = NormNonLinearity(r, function='n_relu')
    
        nnl.check_equivariance()

    def test_so2_norm_sigmoid(self):
        g = rot2dOnR2(-1, 10)
    
        r = FieldType(g, list(g.representations.values()) * 4)
    
        nnl = NormNonLinearity(r, function='n_sigmoid')
    
        nnl.check_equivariance()

    def test_so2_pointwise_relu(self):
        g = rot2dOnR2(-1, 10)
    
        reprs = [r for r in g.representations.values() if 'pointwise' in r.supported_nonlinearities]
    
        r = FieldType(g, reprs)
    
        nnl = PointwiseNonLinearity(r, function='p_relu')
    
        nnl.check_equivariance()

    def test_so2_pointwise_sigmoid(self):
        g = rot2dOnR2(-1, 10)
    
        reprs = [r for r in g.representations.values() if 'pointwise' in r.supported_nonlinearities]
    
        r = FieldType(g, reprs)
    
        nnl = PointwiseNonLinearity(r, function='p_sigmoid')
    
        nnl.check_equivariance()

    def test_so2_gated_one_input_shuffled_gated(self):
        g = rot2dOnR2(-1, 10)
    
        reprs = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities]
        ngates = len(reprs)
        reprs += [g.trivial_repr] * ngates
    
        gates = ['gated'] * ngates + ['gate'] * ngates
    
        r = FieldType(g, reprs)
    
        nnl = GatedNonLinearity1(r, gates=gates)
        nnl.check_equivariance()

    def test_so2_gated_one_input_sorted_gated(self):
        g = rot2dOnR2(-1, 10)
    
        reprs = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities]
    
        r = FieldType(g, reprs).sorted()
    
        ngates = len(r)
    
        reprs = [g.trivial_repr] * ngates
    
        gates = ['gated'] * ngates + ['gate'] * ngates
    
        r = r + FieldType(g, reprs)
    
        nnl = GatedNonLinearity1(r, gates=gates)
        nnl.check_equivariance()

    def test_so2_gated_one_input_all_shuffled(self):
        g = rot2dOnR2(-1, 10)
    
        reprs = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities]
    
        ngates = len(reprs)
    
        reprs += [g.trivial_repr] * ngates
    
        gates = ['gated'] * ngates + ['gate'] * ngates
    
        t = list(zip(reprs, gates))
    
        random.shuffle(t)
    
        reprs, gates = zip(*t)
    
        r = FieldType(g, reprs)
    
        nnl = GatedNonLinearity1(r, gates=gates)
        nnl.check_equivariance()

    def test_so2_gated_two_inputs_shuffled_gated(self):
        g = rot2dOnR2(-1, 10)
    
        gated = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities]
        ngates = len(gated)
        gates = [g.trivial_repr] * ngates

        gates = FieldType(g, gates)
        gated = FieldType(g, gated)
    
        nnl = GatedNonLinearity2((gates, gated))
        nnl.check_equivariance()

    def test_so2_gated_two_inputs_sorted_gated(self):
        g = rot2dOnR2(-1, 10)
    
        gated = [r for r in g.representations.values() if 'gated' in r.supported_nonlinearities]

        gated = FieldType(g, gated).sorted()
    
        ngates = len(gated)

        gates = [g.trivial_repr] * ngates
        gates = FieldType(g, gates)
    
        nnl = GatedNonLinearity2((gates, gated))
        nnl.check_equivariance()

    def test_cyclic_gated1_error(self):
        N = 8
        g = rot2dOnR2(N)
        
        for r in g.representations.values():
            if 'gated' not in r.supported_nonlinearities:
                r1 = FieldType(g, [r, g.trivial_repr])
                gates = ['gated', 'gate']
                self.assertRaises(AssertionError, GatedNonLinearity1, r1, gates=gates)
    
        for r in g.representations.values():
            if 'gate' not in r.supported_nonlinearities:
                r1 = FieldType(g, [g.trivial_repr, r])
                gates = ['gated', 'gate']
                self.assertRaises(AssertionError, GatedNonLinearity1, r1, gates=gates)

    def test_cyclic_gated2_error(self):
        N = 8
        g = rot2dOnR2(N)
    
        for r in g.representations.values():
            if 'gated' not in r.supported_nonlinearities:
                gated = FieldType(g, [r])
                gates = FieldType(g, [g.trivial_repr])
                self.assertRaises(AssertionError, GatedNonLinearity2, (gates, gated))
    
        for r in g.representations.values():
            if 'gate' not in r.supported_nonlinearities:
                gated = FieldType(g, [g.trivial_repr])
                gates = FieldType(g, [r])
                self.assertRaises(AssertionError, GatedNonLinearity2, (gates, gated))

    def test_cyclic_norm_error(self):
        N = 8
        g = rot2dOnR2(N)
        
        for r in g.representations.values():
        
            if 'norm' not in r.supported_nonlinearities:
                r1 = FieldType(g, [r])
                self.assertRaises(AssertionError, NormNonLinearity, r1)

    def test_cyclic_pointwise_error(self):
        N = 8
        g = rot2dOnR2(N)
        
        for r in g.representations.values():
        
            if 'pointwise' not in r.supported_nonlinearities:
                r1 = FieldType(g, [r])
                self.assertRaises(AssertionError, PointwiseNonLinearity, r1)

    def test_cyclic_concat_error(self):
        N = 8
        g = rot2dOnR2(N)
        
        for r in g.representations.values():

            if 'concatenated' not in r.supported_nonlinearities:
                r1 = FieldType(g, [r])
                self.assertRaises(AssertionError, ConcatenatedNonLinearity, r1)


if __name__ == '__main__':
    unittest.main()
