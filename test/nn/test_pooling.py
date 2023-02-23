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
    # At least, let's check they forward without raising any issues

    def test_PointwiseMaxPool2D(self):
        gs = rot2dOnR2()
        ft = gs.type(gs.trivial_repr)

        m = PointwiseMaxPool2D(ft, 3, 2, 0)

        x = torch.randn(5, ft.size, *[9+i for i in range(gs.dimensionality)])
        x = ft(x)

        m.eval()
        y = m(x)

        self.assertEqual(y.shape, m.evaluate_output_shape(x.shape))

    def test_PointwiseMaxPoolAntialiased2D(self):
        gs = rot2dOnR2()
        ft = gs.type(gs.trivial_repr)

        m = PointwiseMaxPoolAntialiased2D(ft, 3, 2, 0, sigma=0.4)

        x = torch.randn(5, ft.size, *[9+i for i in range(gs.dimensionality)])
        x = ft(x)

        m.eval()
        y = m(x)

        self.assertEqual(y.shape, m.evaluate_output_shape(x.shape))

    def test_PointwiseMaxPool3D(self):
        gs = rot2dOnR3()
        ft = gs.type(gs.trivial_repr)

        m = PointwiseMaxPool3D(ft, 3, 2, 0)

        x = torch.randn(5, ft.size, *[9+i for i in range(gs.dimensionality)])
        x = ft(x)

        m.eval()
        y = m(x)

        self.assertEqual(y.shape, m.evaluate_output_shape(x.shape))

    def test_PointwiseMaxPoolAntialiased3D(self):
        gs = rot2dOnR3()
        ft = gs.type(gs.trivial_repr)

        m = PointwiseMaxPoolAntialiased3D(ft, 3, 2, 0, sigma=0.4)

        x = torch.randn(5, ft.size, *[9+i for i in range(gs.dimensionality)])
        x = ft(x)

        m.eval()
        y = m(x)

        self.assertEqual(y.shape, m.evaluate_output_shape(x.shape))

    def test_PointwiseAvgPool2D(self):
        gs = rot2dOnR2()
        ft = gs.type(gs.trivial_repr)

        m = PointwiseAvgPool2D(ft, 3, 2, 0)

        x = torch.randn(5, ft.size, *[9+i for i in range(gs.dimensionality)])
        x = ft(x)

        m.eval()
        y = m(x)

        self.assertEqual(y.shape, m.evaluate_output_shape(x.shape))

    def test_PointwiseAvgPoolAntialiased2D(self):
        gs = rot2dOnR2()
        ft = gs.type(gs.trivial_repr)

        m = PointwiseAvgPoolAntialiased2D(ft, 0.6, 2, 0)

        x = torch.randn(5, ft.size, *[9+i for i in range(gs.dimensionality)])
        x = ft(x)

        m.eval()
        y = m(x)

        self.assertEqual(y.shape, m.evaluate_output_shape(x.shape))

    def test_PointwiseAvgPool3D(self):
        gs = rot2dOnR3()
        ft = gs.type(gs.trivial_repr)

        m = PointwiseAvgPool3D(ft, 3, 2, 0)

        x = torch.randn(5, ft.size, *[9+i for i in range(gs.dimensionality)])
        x = ft(x)

        m.eval()
        y = m(x)

        self.assertEqual(y.shape, m.evaluate_output_shape(x.shape))

    def test_PointwiseAvgPoolAntialiased3D(self):
        gs = rot2dOnR3()
        ft = gs.type(gs.trivial_repr)

        m = PointwiseAvgPoolAntialiased3D(ft, 0.6, 2, 0)

        x = torch.randn(5, ft.size, *[9+i for i in range(gs.dimensionality)])
        x = ft(x)

        m.eval()
        y = m(x)

        self.assertEqual(y.shape, m.evaluate_output_shape(x.shape))

    def test_PointwiseAdaptiveAvgPool2D(self):
        gs = rot2dOnR2()
        ft = gs.type(gs.trivial_repr)

        m = PointwiseAdaptiveAvgPool2D(ft, 3)

        x = torch.randn(5, ft.size, *[9+i for i in range(gs.dimensionality)])
        x = ft(x)

        m.eval()
        y = m(x)

        self.assertEqual(y.shape, m.evaluate_output_shape(x.shape))

    def test_PointwiseAdaptiveAvgPool3D(self):
        gs = rot2dOnR3()
        ft = gs.type(gs.trivial_repr)

        m = PointwiseAdaptiveAvgPool3D(ft, 3)

        x = torch.randn(5, ft.size, *[9+i for i in range(gs.dimensionality)])
        x = ft(x)

        m.eval()
        y = m(x)

        self.assertEqual(y.shape, m.evaluate_output_shape(x.shape))

    def test_PointwiseAdaptiveMaxPool2D(self):
        gs = rot2dOnR2()
        ft = gs.type(gs.trivial_repr)

        m = PointwiseAdaptiveMaxPool2D(ft, 3)

        x = torch.randn(5, ft.size, *[9+i for i in range(gs.dimensionality)])
        x = ft(x)

        m.eval()
        y = m(x)

        self.assertEqual(y.shape, m.evaluate_output_shape(x.shape))

    def test_PointwiseAdaptiveMaxPool3D(self):
        gs = rot2dOnR3()
        ft = gs.type(gs.trivial_repr)

        m = PointwiseAdaptiveMaxPool3D(ft, 3)

        x = torch.randn(5, ft.size, *[9+i for i in range(gs.dimensionality)])
        x = ft(x)

        m.eval()
        y = m(x)

        self.assertEqual(y.shape, m.evaluate_output_shape(x.shape))


if __name__ == '__main__':
    unittest.main()
