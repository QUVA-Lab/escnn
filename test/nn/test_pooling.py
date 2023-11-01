import unittest
from unittest import TestCase

from escnn.nn import *
from escnn.gspaces import *

import torch

import numpy as np


class TestPooling(TestCase):
    
    def test_pointwise_maxpooling(self):
        
        N = 8
        g = rot2dOnR2(N)
        
        r = FieldType(g, [repr for repr in g.representations.values() if 'pointwise' in repr.supported_nonlinearities] * 3)
        
        mpl = PointwiseMaxPool2D(r, kernel_size=(3, 1), stride=(2, 2))

        x = torch.randn(3, r.size, 10, 15)

        x = GeometricTensor(x, r)

        for el in g.testing_elements:
            
            out1 = mpl(x).transform_fibers(el)
            out2 = mpl(x.transform_fibers(el))
    
            errs = (out1.tensor - out2.tensor).detach().numpy()
            errs = np.abs(errs).reshape(-1)
            print(el, errs.max(), errs.mean(), errs.var())
    
            assert torch.allclose(out1.tensor, out2.tensor, atol=1e-6, rtol=1e-5), \
                'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}' \
                    .format(el, errs.max(), errs.mean(), errs.var())

    def test_pointwise_maxpooling3D(self):

        g = icoOnR3()

        r = FieldType(g,
                      [repr for repr in g.representations.values() if 'pointwise' in repr.supported_nonlinearities] * 3)

        mpl = PointwiseMaxPool3D(r, kernel_size=(3, 1, 2), stride=(2, 2, 2))

        x = torch.randn(3, r.size, 10, 15, 13)

        x = GeometricTensor(x, r)

        for el in g.testing_elements:
            out1 = mpl(x).transform_fibers(el)
            out2 = mpl(x.transform_fibers(el))

            errs = (out1.tensor - out2.tensor).detach().numpy()
            errs = np.abs(errs).reshape(-1)
            print(el, errs.max(), errs.mean(), errs.var())

            assert torch.allclose(out1.tensor, out2.tensor, atol=1e-6, rtol=1e-5), \
                'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}' \
                    .format(el, errs.max(), errs.mean(), errs.var())

    def test_norm_maxpooling(self):
    
        N = 8
        g = rot2dOnR2(N)
    
        r = FieldType(g, list(g.representations.values()) * 3)
        
        print(r.size)
    
        mpl = NormMaxPool(r, kernel_size=(3, 1), stride=(2, 2))
    
        x = torch.randn(3, r.size, 10, 15)
    
        x = GeometricTensor(x, r)
    
        for el in g.testing_elements:
            out1 = mpl(x).transform_fibers(el)
            out2 = mpl(x.transform_fibers(el))
        
            errs = (out1.tensor - out2.tensor).detach().numpy()
            errs = np.abs(errs).reshape(-1)
            print(el, errs.max(), errs.mean(), errs.var())
        
            assert torch.allclose(out1.tensor, out2.tensor, atol=1e-6, rtol=1e-5), \
                'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}' \
                    .format(el, errs.max(), errs.mean(), errs.var())

    def test_orientation_pooling(self):
        
        N = 8
        g = rot2dOnR2(N)
        
        r = FieldType(g, [repr for repr in g.representations.values() if 'pointwise' in repr.supported_nonlinearities] * 3)
        
        mpl = GroupPooling(r)

        x = torch.randn(3, r.size, 10, 15)

        x = GeometricTensor(x, r)

        for el in g.testing_elements:
            
            out1 = mpl(x).transform_fibers(el)
            out2 = mpl(x.transform_fibers(el))
    
            errs = (out1.tensor - out2.tensor).detach().numpy()
            errs = np.abs(errs).reshape(-1)
            print(el, errs.max(), errs.mean(), errs.var())
    
            assert torch.allclose(out1.tensor, out2.tensor, atol=1e-6, rtol=1e-5), \
                'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}' \
                    .format(el, errs.max(), errs.mean(), errs.var())

    def test_norm_pooling(self):
    
        N = 8
        g = rot2dOnR2(N)
    
        r = FieldType(g, list(g.representations.values()) * 3)
    
        mpl = NormPool(r)
    
        x = torch.randn(3, r.size, 10, 15)
    
        x = GeometricTensor(x, r)
    
        for el in g.testing_elements:
            out1 = mpl(x).transform_fibers(el)
            out2 = mpl(x.transform_fibers(el))
        
            errs = (out1.tensor - out2.tensor).detach().numpy()
            errs = np.abs(errs).reshape(-1)
            print(el, errs.max(), errs.mean(), errs.var())
        
            assert torch.allclose(out1.tensor, out2.tensor, atol=1e-6, rtol=1e-5), \
                'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}' \
                    .format(el, errs.max(), errs.mean(), errs.var())

    def test_induced_norm_pooling(self):
    
        N = 8
        g = flipRot2dOnR2(-1, 6)
        
        sgid = (None, -1)
        sg, _, _ = g.restrict(sgid)
    
        r = FieldType(g, list(g.induced_repr(sgid, r) for r in sg.representations.values() if not r.is_trivial()) * 3)
    
        mpl = InducedNormPool(r)
        W, H = 10, 15
        x = torch.randn(3, r.size, W, H)
    
        x = GeometricTensor(x, r)
    
        for el in g.testing_elements:
            
            expected_out, _ = x.tensor.view(-1, len(r), 2, 2, W, H).norm(dim=3).max(dim=2)
            
            out1 = mpl(x)
            
            self.assertTrue(torch.allclose(expected_out, out1.tensor))
            
            out1 = out1.transform_fibers(el)
            out2 = mpl(x.transform_fibers(el))
        
            errs = (out1.tensor - out2.tensor).detach().numpy()
            errs = np.abs(errs).reshape(-1)
            print(el, errs.max(), errs.mean(), errs.var())
        
            assert torch.allclose(out1.tensor, out2.tensor, atol=1e-6, rtol=1e-5), \
                'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}' \
                    .format(el, errs.max(), errs.mean(), errs.var())

    # Not all modules are properly equivariant so it doesn't make sense to check their equivariance error
    # At least, let's check that they pool correctly.

    def test_PointwiseMaxPool2D(self):
        gs = rot2dOnR2()
        ft = gs.type(gs.trivial_repr)

        m = PointwiseMaxPool2D(ft, 3, 2, 0)

        x = torch.Tensor([
            [0, 4, 0, 1, 0, 0],
            [1, 0, 1, 0, 1, 1],
            [1, 2, 0, 3, 0, 2],
            [0, 0, 1, 0, 1, 1],
            [1, 0, 1, 3, 2, 0],
        ])
        x = x.view(1, 1, 5, 6)
        x = ft(x)

        m.eval()
        y = m(x)

        # Manually verified.
        y_expected = torch.Tensor([
            [4, 3], 
            [2, 3],
        ])

        torch.testing.assert_close(y.tensor, y_expected.view(1, 1, 2, 2))

        self.assertEqual(y.shape, m.evaluate_output_shape(x.shape))

    def test_PointwiseMaxPoolAntialiased2D(self):
        gs = rot2dOnR2()
        ft = gs.type(gs.trivial_repr)

        m = PointwiseMaxPoolAntialiased2D(ft, 3, 2, 0, sigma=0.4)

        x = torch.Tensor([
            [0, 4, 0, 1, 0, 0],
            [1, 0, 1, 0, 1, 1],
            [1, 2, 0, 3, 0, 2],
            [0, 0, 1, 0, 1, 1],
            [1, 0, 1, 3, 2, 0],
        ])
        x = x.view(1, 1, 5, 6)
        x = ft(x)

        m.eval()
        y = m(x)

        # Manually verified.
        y_expected = torch.Tensor([
            [3.6075, 2.9159], 
            [1.8805, 2.8788],
        ])

        torch.testing.assert_close(
                y.tensor,
                y_expected.view(1, 1, 2, 2),
                atol=1e-4,
                rtol=0,
        )

        self.assertEqual(y.shape, m.evaluate_output_shape(x.shape))

    def test_PointwiseMaxPool3D(self):
        gs = rot2dOnR3()
        ft = gs.type(gs.trivial_repr)

        m = PointwiseMaxPool3D(ft, 3, 2, 0)

        x = torch.randn(5, ft.size, *[9+i for i in range(gs.dimensionality)])

        x = torch.Tensor([
            [[0, 0, 0, 0, 1, 0, 0],
             [2, 2, 1, 0, 2, 1, 1],
             [0, 0, 1, 0, 1, 3, 1],
             [1, 0, 2, 1, 2, 0, 1],
             [1, 0, 0, 0, 1, 2, 2],
             [3, 2, 2, 0, 1, 0, 1]],

            [[0, 0, 0, 2, 1, 0, 2],
             [2, 1, 0, 1, 0, 0, 1],
             [1, 0, 1, 3, 0, 1, 2],
             [2, 1, 0, 1, 1, 0, 1],
             [1, 1, 0, 1, 1, 1, 0],
             [1, 0, 1, 1, 4, 1, 1]],

            [[1, 1, 2, 1, 0, 1, 1],
             [1, 1, 1, 1, 1, 0, 1],
             [0, 1, 0, 2, 0, 1, 2],
             [2, 1, 2, 0, 0, 2, 0],
             [2, 1, 0, 2, 2, 2, 3],
             [2, 1, 1, 0, 0, 1, 0]],

            [[0, 2, 1, 2, 5, 1, 0],
             [2, 1, 1, 5, 0, 1, 0],
             [0, 2, 1, 0, 3, 1, 0],
             [1, 1, 2, 0, 2, 1, 0],
             [1, 0, 1, 2, 1, 0, 0],
             [0, 1, 2, 0, 2, 2, 2]],

            [[1, 1, 0, 1, 1, 0, 2],
             [1, 0, 1, 2, 1, 0, 1],
             [3, 1, 0, 0, 0, 0, 0],
             [1, 0, 0, 1, 2, 0, 0],
             [3, 1, 0, 1, 1, 1, 1],
             [1, 3, 1, 1, 2, 2, 0]],
        ])
        x = x.view(1, 1, 5, 6, 7)
        x = ft(x)

        m.eval()
        y = m(x)

        # Manually verified.
        y_expected = torch.Tensor([
            [[2, 3, 3],
             [2, 3, 3]],
            [[3, 5, 5],
             [3, 3, 3]],
        ])

        torch.testing.assert_close(y.tensor, y_expected.view(1, 1, 2, 2, 3))

        self.assertEqual(y.shape, m.evaluate_output_shape(x.shape))

    def test_PointwiseMaxPoolAntialiased3D(self):
        gs = rot2dOnR3()
        ft = gs.type(gs.trivial_repr)

        m = PointwiseMaxPoolAntialiased3D(ft, 3, 2, 0, sigma=0.4)

        x = torch.Tensor([
            [[0, 0, 0, 0, 1, 0, 0],
             [2, 2, 1, 0, 2, 1, 1],
             [0, 0, 1, 0, 1, 3, 1],
             [1, 0, 2, 1, 2, 0, 1],
             [1, 0, 0, 0, 1, 2, 2],
             [3, 2, 2, 0, 1, 0, 1]],

            [[0, 0, 0, 2, 1, 0, 2],
             [2, 1, 0, 1, 0, 0, 1],
             [1, 0, 1, 3, 0, 1, 2],
             [2, 1, 0, 1, 1, 0, 1],
             [1, 1, 0, 1, 1, 1, 0],
             [1, 0, 1, 1, 4, 1, 1]],

            [[1, 1, 2, 1, 0, 1, 1],
             [1, 1, 1, 1, 1, 0, 1],
             [0, 1, 0, 2, 0, 1, 2],
             [2, 1, 2, 0, 0, 2, 0],
             [2, 1, 0, 2, 2, 2, 3],
             [2, 1, 1, 0, 0, 1, 0]],

            [[0, 2, 1, 2, 5, 1, 0],
             [2, 1, 1, 5, 0, 1, 0],
             [0, 2, 1, 0, 3, 1, 0],
             [1, 1, 2, 0, 2, 1, 0],
             [1, 0, 1, 2, 1, 0, 0],
             [0, 1, 2, 0, 2, 2, 2]],

            [[1, 1, 0, 1, 1, 0, 2],
             [1, 0, 1, 2, 1, 0, 1],
             [3, 1, 0, 0, 0, 0, 0],
             [1, 0, 0, 1, 2, 0, 0],
             [3, 1, 0, 1, 1, 1, 1],
             [1, 3, 1, 1, 2, 2, 0]],
        ])
        x = x.view(1, 1, 5, 6, 7)
        x = ft(x)

        m.eval()
        y = m(x)

        # Not manually verified.
        y_expected = torch.Tensor([
            [[1.8076, 2.8401, 2.7224],
             [1.9131, 2.9177, 2.7999]],
            [[2.6897, 4.6042, 4.3470],
             [2.6943, 2.8881, 2.7657]],
        ])

        torch.testing.assert_close(
                y.tensor,
                y_expected.view(1, 1, 2, 2, 3),
                atol=1e-4,
                rtol=0,
        )

        self.assertEqual(y.shape, m.evaluate_output_shape(x.shape))

    def test_PointwiseAvgPool2D(self):
        gs = rot2dOnR2()
        ft = gs.type(gs.trivial_repr)

        m = PointwiseAvgPool2D(ft, 3, 2, 0)

        x = torch.Tensor([
            [0, 4, 0, 1, 0, 0],
            [1, 0, 1, 0, 1, 1],
            [1, 2, 0, 3, 0, 2],
            [0, 0, 1, 0, 1, 1],
            [1, 0, 1, 3, 2, 0],
        ])
        x = x.view(1, 1, 5, 6)
        x = ft(x)

        m.eval()
        y = m(x)

        # Manually verified.
        y_expected = torch.Tensor([
            [ 9/9,  6/9], 
            [ 6/9, 11/9],
        ])

        torch.testing.assert_close(y.tensor, y_expected.view(1, 1, 2, 2))

        self.assertEqual(y.shape, m.evaluate_output_shape(x.shape))

    def test_PointwiseAvgPoolAntialiased2D(self):
        gs = rot2dOnR2()
        ft = gs.type(gs.trivial_repr)

        m = PointwiseAvgPoolAntialiased2D(ft, 0.6, 2, 0)

        x = torch.Tensor([
            [0, 4, 0, 1, 0, 0],
            [1, 0, 1, 0, 1, 1],
            [1, 2, 0, 3, 0, 2],
            [0, 0, 1, 0, 1, 1],
            [1, 0, 1, 3, 2, 0],
        ])
        x = x.view(1, 1, 5, 6)
        x = ft(x)

        m.eval()
        y = m(x)

        # Manually verified.
        y_expected = torch.Tensor([
            [0.7773],
        ])

        torch.testing.assert_close(
                y.tensor,
                y_expected.view(1, 1, 1, 1),
                atol=1e-4,
                rtol=0,
        )

        self.assertEqual(y.shape, m.evaluate_output_shape(x.shape))

    def test_PointwiseAvgPool3D(self):
        gs = rot2dOnR3()
        ft = gs.type(gs.trivial_repr)

        m = PointwiseAvgPool3D(ft, 3, 2, 0)

        x = torch.Tensor([
            [[0, 0, 0, 0, 1, 0, 0],
             [2, 2, 1, 0, 2, 1, 1],
             [0, 0, 1, 0, 1, 3, 1],
             [1, 0, 2, 1, 2, 0, 1],
             [1, 0, 0, 0, 1, 2, 2],
             [3, 2, 2, 0, 1, 0, 1]],

            [[0, 0, 0, 2, 1, 0, 2],
             [2, 1, 0, 1, 0, 0, 1],
             [1, 0, 1, 3, 0, 1, 2],
             [2, 1, 0, 1, 1, 0, 1],
             [1, 1, 0, 1, 1, 1, 0],
             [1, 0, 1, 1, 4, 1, 1]],

            [[1, 1, 2, 1, 0, 1, 1],
             [1, 1, 1, 1, 1, 0, 1],
             [0, 1, 0, 2, 0, 1, 2],
             [2, 1, 2, 0, 0, 2, 0],
             [2, 1, 0, 2, 2, 2, 3],
             [2, 1, 1, 0, 0, 1, 0]],

            [[0, 2, 1, 2, 5, 1, 0],
             [2, 1, 1, 5, 0, 1, 0],
             [0, 2, 1, 0, 3, 1, 0],
             [1, 1, 2, 0, 2, 1, 0],
             [1, 0, 1, 2, 1, 0, 0],
             [0, 1, 2, 0, 2, 2, 2]],

            [[1, 1, 0, 1, 1, 0, 2],
             [1, 0, 1, 2, 1, 0, 1],
             [3, 1, 0, 0, 0, 0, 0],
             [1, 0, 0, 1, 2, 0, 0],
             [3, 1, 0, 1, 1, 1, 1],
             [1, 3, 1, 1, 2, 2, 0]],
        ])
        x = x.view(1, 1, 5, 6, 7)
        x = ft(x)

        m.eval()
        y = m(x)

        # Manually verified.
        y_expected = torch.Tensor([
            [[19/27, 22/27, 24/27],
             [21/27, 24/27, 32/27]],
            [[26/27, 32/27, 23/27],
             [27/27, 25/27, 25/27]],
        ])

        torch.testing.assert_close(y.tensor, y_expected.view(1, 1, 2, 2, 3))

        self.assertEqual(y.shape, m.evaluate_output_shape(x.shape))

    def test_PointwiseAvgPoolAntialiased3D(self):
        gs = rot2dOnR3()
        ft = gs.type(gs.trivial_repr)

        m = PointwiseAvgPoolAntialiased3D(ft, 0.6, 2, 0)

        x = torch.Tensor([
            [[0, 0, 0, 0, 1, 0, 0],
             [2, 2, 1, 0, 2, 1, 1],
             [0, 0, 1, 0, 1, 3, 1],
             [1, 0, 2, 1, 2, 0, 1],
             [1, 0, 0, 0, 1, 2, 2],
             [3, 2, 2, 0, 1, 0, 1]],

            [[0, 0, 0, 2, 1, 0, 2],
             [2, 1, 0, 1, 0, 0, 1],
             [1, 0, 1, 3, 0, 1, 2],
             [2, 1, 0, 1, 1, 0, 1],
             [1, 1, 0, 1, 1, 1, 0],
             [1, 0, 1, 1, 4, 1, 1]],

            [[1, 1, 2, 1, 0, 1, 1],
             [1, 1, 1, 1, 1, 0, 1],
             [0, 1, 0, 2, 0, 1, 2],
             [2, 1, 2, 0, 0, 2, 0],
             [2, 1, 0, 2, 2, 2, 3],
             [2, 1, 1, 0, 0, 1, 0]],

            [[0, 2, 1, 2, 5, 1, 0],
             [2, 1, 1, 5, 0, 1, 0],
             [0, 2, 1, 0, 3, 1, 0],
             [1, 1, 2, 0, 2, 1, 0],
             [1, 0, 1, 2, 1, 0, 0],
             [0, 1, 2, 0, 2, 2, 2]],

            [[1, 1, 0, 1, 1, 0, 2],
             [1, 0, 1, 2, 1, 0, 1],
             [3, 1, 0, 0, 0, 0, 0],
             [1, 0, 0, 1, 2, 0, 0],
             [3, 1, 0, 1, 1, 1, 1],
             [1, 3, 1, 1, 2, 2, 0]],
        ])
        x = x.view(1, 1, 5, 6, 7)
        x = ft(x)

        m.eval()
        y = m(x)

        # Not manually verified.
        y_expected = torch.Tensor([
            [0.8444, 0.7675],
        ])

        torch.testing.assert_close(
                y.tensor,
                y_expected.view(1, 1, 1, 1, 2),
                atol=1e-4,
                rtol=0,
        )

        self.assertEqual(y.shape, m.evaluate_output_shape(x.shape))

    def test_PointwiseAdaptiveAvgPool2D(self):
        gs = rot2dOnR2()
        ft = gs.type(gs.trivial_repr)

        m = PointwiseAdaptiveAvgPool2D(ft, 2)

        x = torch.Tensor([
            [0, 4, 0, 1, 0, 0],
            [1, 0, 1, 0, 1, 1],
            [1, 2, 0, 3, 0, 2],
            [0, 0, 1, 0, 1, 1],
            [1, 0, 1, 3, 2, 0],
        ])
        x = x.view(1, 1, 5, 6)
        x = ft(x)

        m.eval()
        y = m(x)

        # Manually verified, assuming kernel_size=3 and stride=(2, 3)
        y_expected = torch.Tensor([
            [ 9/9,  8/9], 
            [ 6/9, 12/9],
        ])

        torch.testing.assert_close(y.tensor, y_expected.view(1, 1, 2, 2))

        self.assertEqual(y.shape, m.evaluate_output_shape(x.shape))

    def test_PointwiseAdaptiveAvgPool3D(self):
        gs = rot2dOnR3()
        ft = gs.type(gs.trivial_repr)

        m = PointwiseAdaptiveAvgPool3D(ft, (2, 2, 3))

        x = torch.Tensor([
            [[0, 0, 0, 0, 1, 0, 0],
             [2, 2, 1, 0, 2, 1, 1],
             [0, 0, 1, 0, 1, 3, 1],
             [1, 0, 2, 1, 2, 0, 1],
             [1, 0, 0, 0, 1, 2, 2],
             [3, 2, 2, 0, 1, 0, 1]],

            [[0, 0, 0, 2, 1, 0, 2],
             [2, 1, 0, 1, 0, 0, 1],
             [1, 0, 1, 3, 0, 1, 2],
             [2, 1, 0, 1, 1, 0, 1],
             [1, 1, 0, 1, 1, 1, 0],
             [1, 0, 1, 1, 4, 1, 1]],

            [[1, 1, 2, 1, 0, 1, 1],
             [1, 1, 1, 1, 1, 0, 1],
             [0, 1, 0, 2, 0, 1, 2],
             [2, 1, 2, 0, 0, 2, 0],
             [2, 1, 0, 2, 2, 2, 3],
             [2, 1, 1, 0, 0, 1, 0]],

            [[0, 2, 1, 2, 5, 1, 0],
             [2, 1, 1, 5, 0, 1, 0],
             [0, 2, 1, 0, 3, 1, 0],
             [1, 1, 2, 0, 2, 1, 0],
             [1, 0, 1, 2, 1, 0, 0],
             [0, 1, 2, 0, 2, 2, 2]],

            [[1, 1, 0, 1, 1, 0, 2],
             [1, 0, 1, 2, 1, 0, 1],
             [3, 1, 0, 0, 0, 0, 0],
             [1, 0, 0, 1, 2, 0, 0],
             [3, 1, 0, 1, 1, 1, 1],
             [1, 3, 1, 1, 2, 2, 0]],
        ])
        x = x.view(1, 1, 5, 6, 7)
        x = ft(x)

        m.eval()
        y = m(x)

        # Manually verified, assuming kernel_size=3 and stride=(2,3,2).
        y_expected = torch.Tensor([
            [[19/27, 22/27, 24/27],
             [30/27, 26/27, 30/27]],
            [[26/27, 32/27, 23/27],
             [31/27, 28/27, 29/27]],
        ])

        torch.testing.assert_close(y.tensor, y_expected.view(1, 1, 2, 2, 3))

        self.assertEqual(y.shape, m.evaluate_output_shape(x.shape))

    def test_PointwiseAdaptiveMaxPool2D(self):
        gs = rot2dOnR2()
        ft = gs.type(gs.trivial_repr)

        m = PointwiseAdaptiveMaxPool2D(ft, 2)

        x = torch.Tensor([
            [0, 4, 0, 1, 0, 0],
            [1, 0, 1, 0, 1, 1],
            [1, 2, 0, 3, 0, 2],
            [0, 0, 1, 0, 1, 1],
            [1, 0, 1, 3, 2, 0],
        ])
        x = x.view(1, 1, 5, 6)
        x = ft(x)

        m.eval()
        y = m(x)

        # Manually verified, assuming kernel_size=3 and stride=(2, 3)
        y_expected = torch.Tensor([
            [4, 3], 
            [2, 3],
        ])

        torch.testing.assert_close(y.tensor, y_expected.view(1, 1, 2, 2))

        self.assertEqual(y.shape, m.evaluate_output_shape(x.shape))

    def test_PointwiseAdaptiveMaxPool3D(self):
        gs = rot2dOnR3()
        ft = gs.type(gs.trivial_repr)

        m = PointwiseAdaptiveMaxPool3D(ft, (2, 2, 3))

        x = torch.Tensor([
            [[0, 0, 0, 0, 1, 0, 0],
             [2, 2, 1, 0, 2, 1, 1],
             [0, 0, 1, 0, 1, 3, 1],
             [1, 0, 2, 1, 2, 0, 1],
             [1, 0, 0, 0, 1, 2, 2],
             [3, 2, 2, 0, 1, 0, 1]],

            [[0, 0, 0, 2, 1, 0, 2],
             [2, 1, 0, 1, 0, 0, 1],
             [1, 0, 1, 3, 0, 1, 2],
             [2, 1, 0, 1, 1, 0, 1],
             [1, 1, 0, 1, 1, 1, 0],
             [1, 0, 1, 1, 4, 1, 1]],

            [[1, 1, 2, 1, 0, 1, 1],
             [1, 1, 1, 1, 1, 0, 1],
             [0, 1, 0, 2, 0, 1, 2],
             [2, 1, 2, 0, 0, 2, 0],
             [2, 1, 0, 2, 2, 2, 3],
             [2, 1, 1, 0, 0, 1, 0]],

            [[0, 2, 1, 2, 5, 1, 0],
             [2, 1, 1, 5, 0, 1, 0],
             [0, 2, 1, 0, 3, 1, 0],
             [1, 1, 2, 0, 2, 1, 0],
             [1, 0, 1, 2, 1, 0, 0],
             [0, 1, 2, 0, 2, 2, 2]],

            [[1, 1, 0, 1, 1, 0, 2],
             [1, 0, 1, 2, 1, 0, 1],
             [3, 1, 0, 0, 0, 0, 0],
             [1, 0, 0, 1, 2, 0, 0],
             [3, 1, 0, 1, 1, 1, 1],
             [1, 3, 1, 1, 2, 2, 0]],
        ])
        x = x.view(1, 1, 5, 6, 7)
        x = ft(x)

        m.eval()
        y = m(x)

        # Manually verified, assuming kernel_size=3 and stride=(2,3,2).
        y_expected = torch.Tensor([
            [[2, 3, 3],
             [3, 4, 4]],
            [[3, 5, 5],
             [3, 2, 3]],
        ])

        torch.testing.assert_close(y.tensor, y_expected.view(1, 1, 2, 2, 3))

        self.assertEqual(y.shape, m.evaluate_output_shape(x.shape))

    def test_output_shape_2d(self):
        # The main purpose of this function is to make sure that all of the 
        # arguments to the pooling module have the expected effect, i.e. they 
        # aren't ignored or misinterpreted.

        gs = rot2dOnR2()
        ft = gs.type(gs.trivial_repr)

        cases = [
                # Pointwise pooling:
                # Use `PointwiseMaxPool2D` as a representative of all the 
                # `_PointwisePoolND` subclasses.
                dict(
                    module=PointwiseMaxPool2D(
                        ft,
                        kernel_size=3,
                        stride=2,
                    ),
                    in_shape=(2, 1, 5, 5),
                    out_shape=(2, 1, 2, 2),
                ),
                dict(
                    module=PointwiseMaxPool2D(
                        ft,
                        kernel_size=5,
                        stride=2,
                    ),
                    in_shape=(2, 1, 7, 7),
                    out_shape=(2, 1, 2, 2),
                ),
                dict(
                    module=PointwiseMaxPool2D(
                        ft,
                        kernel_size=3,
                        stride=1,
                    ),
                    in_shape=(2, 1, 4, 4),
                    out_shape=(2, 1, 2, 2),
                ),
                dict(
                    module=PointwiseMaxPool2D(
                        ft,
                        kernel_size=3,
                        stride=None,  # Means same as kernel size
                    ),
                    in_shape=(2, 1, 6, 6),
                    out_shape=(2, 1, 2, 2),
                ),
                dict(
                    module=PointwiseMaxPool2D(
                        ft,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    ),
                    in_shape=(2, 1, 3, 3),
                    out_shape=(2, 1, 2, 2),
                ),
                dict(
                    module=PointwiseMaxPool2D(
                        ft,
                        kernel_size=3,
                        stride=2,
                        dilation=2,
                    ),
                    in_shape=(2, 1, 7, 7),
                    out_shape=(2, 1, 2, 2),
                ),
                dict(
                    module=PointwiseMaxPool2D(
                        ft,
                        kernel_size=3,
                        stride=2,
                        ceil_mode=True,
                    ),
                    in_shape=(2, 1, 4, 4),
                    out_shape=(2, 1, 2, 2),
                ),

                # Antialiased average pooling:
                dict(
                    module=PointwiseAvgPoolAntialiased2D(
                        ft,
                        sigma=0.6,  # kernel size: 5
                        stride=2,
                    ),
                    in_shape=(2, 1, 3, 3),
                    out_shape=(2, 1, 2, 2),
                ),
                dict(
                    module=PointwiseAvgPoolAntialiased2D(
                        ft,
                        sigma=0.4,  # kernel size: 3
                        stride=2,
                    ),
                    in_shape=(2, 1, 3, 3),
                    out_shape=(2, 1, 2, 2),
                ),
                dict(
                    module=PointwiseAvgPoolAntialiased2D(
                        ft,
                        sigma=0.6,  # kernel size: 5
                        stride=1,
                    ),
                    in_shape=(2, 1, 2, 2),
                    out_shape=(2, 1, 2, 2),
                ),
                dict(
                    module=PointwiseAvgPoolAntialiased2D(
                        ft,
                        sigma=0.6,  # kernel size: 5
                        stride=2,
                        padding=0,
                    ),
                    in_shape=(2, 1, 7, 7),
                    out_shape=(2, 1, 2, 2),
                ),

                # Antialiased max pooling:
                dict(
                    module=PointwiseMaxPoolAntialiased2D(
                        ft,
                        kernel_size=3,
                    ),
                    in_shape=(2, 1, 6, 6),
                    out_shape=(2, 1, 2, 2),
                ),
                dict(
                    module=PointwiseMaxPoolAntialiased2D(
                        ft,
                        kernel_size=5,
                    ),
                    in_shape=(2, 1, 10, 10),
                    out_shape=(2, 1, 2, 2),
                ),
                dict(
                    module=PointwiseMaxPoolAntialiased2D(
                        ft,
                        kernel_size=3,
                        # This value of sigma reduces the size of the blur 
                        # filter from 5 (the default) to 3.  But because the 
                        # blur filter is hard-coded to use "relative padding", 
                        # the size of the blur filter has no effect on the size 
                        # of the output.
                        sigma=0.3,
                    ),
                    in_shape=(2, 1, 6, 6),
                    out_shape=(2, 1, 2, 2),
                ),
                dict(
                    module=PointwiseMaxPoolAntialiased2D(
                        ft,
                        kernel_size=3,
                        stride=2,
                    ),
                    in_shape=(2, 1, 5, 5),
                    out_shape=(2, 1, 2, 2),
                ),
                dict(
                    module=PointwiseMaxPoolAntialiased2D(
                        ft,
                        kernel_size=3,
                        padding=1
                    ),
                    in_shape=(2, 1, 4, 4),
                    out_shape=(2, 1, 2, 2),
                ),
        ]

        for case in cases:
            f = case['module']

            with self.subTest(f):
                x = ft(torch.zeros(*case['in_shape']))
                y = f(x)

                self.assertEqual(y.shape, case['out_shape'])
                self.assertEqual(y.shape, f.evaluate_output_shape(x.shape))

if __name__ == '__main__':
    unittest.main()
