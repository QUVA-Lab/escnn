import unittest
from unittest import TestCase

import escnn.nn.init as init
from escnn.nn import *
from escnn.gspaces import *
from escnn.group import *
from escnn.kernels import *

import numpy as np
import math

import torch


class TestConvolution(TestCase):
    
    def test_cyclic(self):
        N = 8
        g = rot2dOnR2(N)
        
        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()) * 2)
        # r1 = FieldType(g, [g.trivial_repr])
        # r2 = FieldType(g, [g.regular_repr])
        
        s = 7
        sigma = None
        # fco = lambda r: 1. * r * np.pi
        fco = None
        
        cl = R2Conv(r1, r2, s,
                    sigma=sigma,
                    frequencies_cutoff=fco,
                    bias=True)
        cl.bias.data = 20*torch.randn_like(cl.bias.data)

        for _ in range(1):
            init.generalized_he_init(cl.weights.data, cl.basisexpansion, cache=True)
            cl.eval()
            cl.check_equivariance(device = 'cuda' if torch.cuda.is_available() else 'cpu')

        cl.train()
        for _ in range(1):
            cl.check_equivariance(device = 'cuda' if torch.cuda.is_available() else 'cpu')

        cl.eval()
        
        for _ in range(5):
            init.generalized_he_init(cl.weights.data, cl.basisexpansion, cache=True)
            cl.eval()
            filter = cl.filter.clone()
            cl.check_equivariance(device = 'cuda' if torch.cuda.is_available() else 'cpu')
            self.assertTrue(torch.allclose(filter, cl.filter))

    def test_so2(self):
        N = 7
        g = rot2dOnR2(-1, N)

        reprs = [g.irrep(*irr) for irr in g.fibergroup.bl_irreps(3)] # + [g.fibergroup.bl_regular_representation(3)]
        r1 = r2 = g.type(*reprs)

        s = 7
        # sigma = 0.6
        # fco = lambda r: 1. * r * np.pi
        # fco = lambda r: 2 * r
        sigma = None
        fco = None
        cl = R2Conv(r1, r2, s,
                    sigma=sigma,
                    frequencies_cutoff=fco,
                    bias=True)
        
        for _ in range(3):
            # cl.basisexpansion._init_weights()
            init.generalized_he_init(cl.weights.data, cl.basisexpansion, cache=True)
            cl.eval()
            cl.check_equivariance(device = 'cuda' if torch.cuda.is_available() else 'cpu')

    def test_dihedral(self):
        N = 8
        g = flipRot2dOnR2(N, axis=np.pi/3)

        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()))
        # r1 = FieldType(g, [g.trivial_repr])
        # r2 = FieldType(g, [g.fibergroup.irrep(1, 0)])
        # r2 = FieldType(g, [irr for irr in g.fibergroup.irreps.values() if irr.size == 1])
        # r2 = FieldType(g, [g.regular_repr])
    
        s = 7
        # sigma = 0.6
        # fco = lambda r: 1. * r * np.pi
        # fco = lambda r: 2 * r
        sigma = None
        fco = None
        cl = R2Conv(r1, r2, s,
                    sigma=sigma,
                    frequencies_cutoff=fco,
                    bias=True)

        for _ in range(3):
            # cl.basisexpansion._init_weights()
            init.generalized_he_init(cl.weights.data, cl.basisexpansion, cache=True)
            cl.eval()
            cl.check_equivariance(device = 'cuda' if torch.cuda.is_available() else 'cpu')

    def test_o2(self):
        N = 3
        s = 7
        # sigma = 0.6
        # fco = lambda r: 1. * r * np.pi
        # fco = lambda r: 2 * r
        sigma = None
        fco = None

        g = flipRot2dOnR2(-1, max(s, 2*N))
        reprs = [g.irrep(*irr) for irr in g.fibergroup.bl_irreps(N)] + [g.fibergroup.bl_regular_representation(N)]
        r1 = r2 = g.type(*reprs)

        cl = R2Conv(r1, r2, s,
                    sigma=sigma,
                    frequencies_cutoff=fco,
                    bias=True)

        for _ in range(3):
            # cl.basisexpansion._init_weights()
            init.generalized_he_init(cl.weights.data, cl.basisexpansion, cache=True)
            cl.eval()
            cl.check_equivariance(device = 'cuda' if torch.cuda.is_available() else 'cpu')

    def test_flip(self):
        # g = flip2dOnR2(axis=np.pi/3)
        g = flip2dOnR2(axis=np.pi/2)

        r1 = FieldType(g, list(g.representations.values()))
        r2 = FieldType(g, list(g.representations.values()) * 3).sorted()
    
        s = 9
        # sigma = 0.6
        # fco = lambda r: 1. * r * np.pi
        # fco = lambda r: 2 * r
        sigma = None
        fco = None
        cl = R2Conv(r1, r2, s,
                    sigma=sigma,
                    frequencies_cutoff=fco,
                    bias=True)
        
        for _ in range(32):
            # cl.basisexpansion._init_weights()
            init.generalized_he_init(cl.weights.data, cl.basisexpansion, cache=True)
            cl.eval()
            cl.check_equivariance(device = 'cuda' if torch.cuda.is_available() else 'cpu')

    def test_1x1_conv2d(self):
        for gs in [
            trivialOnR2(),
            flip2dOnR2(np.pi/2.),
            rot2dOnR2(-1, 5),
            flipRot2dOnR2(-1, 5),
        ]:
            print(gs)

            t = gs.type(*[psi for psi in gs.irreps if psi.attributes['frequency']<3])

            try:
                cl = R2Conv(t, t, kernel_size=1, bias=False, recompute=True)
            except:
                print(gs)
                raise

            init.generalized_he_init(cl.weights.data, cl.basisexpansion, cache=True)
            cl.eval()
            cl.check_equivariance(device = 'cuda' if torch.cuda.is_available() else 'cpu')

    def test_1x1_conv3d(self):
        for gs in [
            trivialOnR3(),
            mirOnR3(),
            rot3dOnR3(5),
            flipRot3dOnR3(5),
        ]:
            print(gs)

            t = gs.type(*[psi for psi in gs.irreps if psi.attributes['frequency']<3])
            try:
                cl = R3Conv(t, t, kernel_size=1, bias=False, recompute=True)
            except:
                print(gs)
                raise

            init.generalized_he_init(cl.weights.data, cl.basisexpansion, cache=True)
            cl.eval()
            cl.check_equivariance(device = 'cuda' if torch.cuda.is_available() else 'cpu')

    def test_padding_mode_r2conv(self):
        g = flipRot2dOnR2(4, axis=np.pi / 2)
    
        r1 = FieldType(g, [g.trivial_repr])
        r2 = FieldType(g, [g.regular_repr])
    
        for mode in ['circular', 'reflect', 'replicate']:
            for s in [3, 5, 7]:
                padding = s // 2
                cl = R2Conv(r1, r2, s, bias=True, padding=padding, padding_mode=mode, initialize=False)

                device='cuda' if torch.cuda.is_available() else 'cpu'
                cl.to(device)

                for i in range(5):
                    init.generalized_he_init(cl.weights.data, cl.basisexpansion, cache=True)
                    cl.eval()
                    cl.check_equivariance(device=device)

    def test_padding_modes_r3conv(self):
        g = octaOnR3()

        r1 = FieldType(g, [g.trivial_repr])
        r2 = FieldType(g, [g.regular_repr])

        for mode in ['circular', 'reflect', 'replicate']:
            for s in [3, 5, 7]:
                padding = s // 2
                cl = R3Conv(r1, r2, s, bias=True, padding=padding, padding_mode=mode, initialize=False)

                print(mode, s)
                device='cuda' if torch.cuda.is_available() else 'cpu'
                cl.to(device)

                for i in range(3):
                    init.generalized_he_init(cl.weights.data, cl.basisexpansion, cache=True)
                    cl.eval()
                    cl.check_equivariance(device=device)

    def test_output_shape(self):
        g = flipRot2dOnR2(4, axis=np.pi / 2)
    
        r1 = FieldType(g, [g.trivial_repr])
        r2 = FieldType(g, [g.regular_repr])
        
        S = 17
        
        x = torch.randn(1, r1.size, S, S)
        x = GeometricTensor(x, r1)
        
        with torch.no_grad():
            for k in [3, 5, 7, 9, 4, 8]:
                for p in [0, 1, 2, 4]:
                    for s in [1, 2, 3]:
                        for mode in ['zeros', 'circular', 'reflect', 'replicate']:
                            cl = R2Conv(r1, r2, k, padding=p, stride=s, padding_mode=mode, initialize=False).eval()
                            y = cl(x)
                            _S = math.floor((S + 2*p - k) / s + 1)
                            self.assertEqual(y.shape, (1, r2.size, _S, _S))
                            self.assertEqual(y.shape, cl.evaluate_output_shape(x.shape))

    def test_so3(self):
        g = rot3dOnR3(3)

        for irr1 in g.fibergroup.bl_irreps(3):
            for irr2 in g.fibergroup.bl_irreps(3):
                print(irr1, irr2)

                r1 = g.type(g.irrep(*irr1))
                r2 = g.type(g.irrep(*irr2))

                s = 5
                # sigma = 0.6
                # fco = lambda r: 1. * r * np.pi
                # fco = lambda r: 2 * r
                sigma = None
                fco = None
                try:
                    cl = R3Conv(r1, r2, s,
                                sigma=sigma,
                                frequencies_cutoff=fco,
                                bias=True)
                except ValueError:
                    continue

                for i in range(1):
                    # cl.basisexpansion._init_weights()
                    # init.generalized_he_init(cl.weights.data, cl.basisexpansion, cache=True)
                    cl.weights.data.normal_()
                    cl.eval()
                    cl.check_equivariance(device = 'cuda' if torch.cuda.is_available() else 'cpu')

    def test_octa(self):
        g = octaOnR3()

        r1 = FieldType(g, g.irreps)
        r2 = FieldType(g, g.irreps)

        s = 7
        # sigma = 0.6
        # fco = lambda r: 1. * r * np.pi
        # fco = lambda r: 2 * r
        sigma = None
        fco = None
        cl = R3Conv(r1, r2, s,
                    sigma=sigma,
                    frequencies_cutoff=fco,
                    bias=True,
                    initialize=False
                    )

        for _ in range(3):
            init.generalized_he_init(cl.weights.data, cl.basisexpansion, cache=True)
            # cl.weights.data.normal_()
            cl.eval()
            cl.check_equivariance(device = 'cuda' if torch.cuda.is_available() else 'cpu')

    def test_ico(self):
        g = icoOnR3()
    
        r1 = FieldType(g, g.irreps)
        r2 = FieldType(g, g.irreps)
    
        s = 7
        # sigma = 0.6
        # fco = lambda r: 1. * r * np.pi
        # fco = lambda r: 2 * r
        sigma = None
        fco = None
        cl = R3Conv(r1, r2, s,
                    sigma=sigma,
                    frequencies_cutoff=fco,
                    bias=True)
    
        for _ in range(3):
            # cl.basisexpansion._init_weights()
            # init.generalized_he_init(cl.weights.data, cl.basisexpansion, cache=True)
            cl.weights.data.normal_()
            cl.eval()
            cl.check_equivariance(device = 'cuda' if torch.cuda.is_available() else 'cpu')

    def test_so3_exact(self):
        g = rot3dOnR3(3)
    
        reprs = [g.irrep(*irr) for irr in g.fibergroup.bl_irreps(3)]
        r1 = r2 = g.type(*reprs)

        s = 3
        # sigma = 0.6
        # fco = lambda r: 1. * r * np.pi
        # fco = lambda r: 2 * r
        sigma = None
        fco = 2.
        cl = R3Conv(r1, r2, s, padding=0,
                    sigma=sigma,
                    frequencies_cutoff=fco,
                    bias=True)

        atol = 1e-4
        rtol = 1e-4
        
        for _ in range(3):
            # cl.basisexpansion._init_weights()
            # init.generalized_he_init(cl.weights.data, cl.basisexpansion, cache=True)
            cl.weights.data.normal_()
            cl.eval()
            
            x = torch.randn(1, cl.in_type.size, 5, 5, 5)
            x = GeometricTensor(x, cl.in_type)

            for el in g.fibergroup.grid('cube'):
    
                out1 = cl(x).transform(el).tensor.detach().numpy()
                out2 = cl(x.transform(el)).tensor.detach().numpy()
    
                out1 = out1.reshape(-1)
                out2 = out2.reshape(-1)
    
                errs = np.abs(out1 - out2)
    
                esum = np.maximum(np.abs(out1), np.abs(out2))
                esum[esum < 1e-7] = 1.
    
                tol = rtol * esum + atol
    
                self.assertTrue(
                    np.all(errs < tol),
                    'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}'.format(el, errs.max(), errs.mean(), errs.var())
                )

    def test_o3_exact(self):
        g = flipRot3dOnR3(2)

        reprs = [g.irrep(*irr) for irr in g.fibergroup.bl_irreps(3)]
        r1 = r2 = g.type(*reprs)

        s = 3
        # sigma = 0.6
        # fco = lambda r: 1. * r * np.pi
        # fco = lambda r: 2 * r
        sigma = None
        fco = 2.
        cl = R3Conv(r1, r2, s, padding=0,
                    sigma=sigma,
                    frequencies_cutoff=fco,
                    bias=True)
    
        atol = 1e-4
        rtol = 1e-4
    
        for _ in range(3):
            # cl.basisexpansion._init_weights()
            # init.generalized_he_init(cl.weights.data, cl.basisexpansion, cache=True)
            cl.weights.data.normal_()
            cl.eval()
        
            x = torch.randn(1, cl.in_type.size, 5, 5, 5)
            x = GeometricTensor(x, cl.in_type)
        
            for el in g.fibergroup.grid('cube'):
                out1 = cl(x).transform(el).tensor.detach().numpy()
                out2 = cl(x.transform(el)).tensor.detach().numpy()
            
                out1 = out1.reshape(-1)
                out2 = out2.reshape(-1)
            
                errs = np.abs(out1 - out2)
            
                esum = np.maximum(np.abs(out1), np.abs(out2))
                esum[esum < 1e-7] = 1.
            
                tol = rtol * esum + atol
            
                self.assertTrue(
                    np.all(errs < tol),
                    'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}'.format(
                        el, errs.max(), errs.mean(), errs.var())
                )

    def test_ico_sparse(self):
        g = icoOnR3()

        r1 = FieldType(g, g.irreps)
        r2 = FieldType(g, g.irreps)

        s = 7
        # sigma = 0.6
        # fco = lambda r: 1. * r * np.pi
        # fco = lambda r: 2 * r
        sigma = None
        cl = R3IcoConv(r1, r2, s,
                    sigma=sigma,
                    samples='ico',
                    bias=True)

        for _ in range(3):
            # cl.basisexpansion._init_weights()
            # init.generalized_he_init(cl.weights.data, cl.basisexpansion, cache=True)
            cl.weights.data.normal_()
            cl.eval()

            # Check exact equivariance to tethrahedron subgroup
            with torch.no_grad():
                x = torch.randn(1, cl.in_type.size, s, s, s)
                x = GeometricTensor(x, cl.in_type)
                for el in so3_group().grid('tetra'):
                    el = g.fibergroup.element(el.to('MAT'), 'MAT')

                    out1 = cl(x).transform(el).tensor.detach().numpy()
                    out2 = cl(x.transform(el)).tensor.detach().numpy()

                    out1 = out1.reshape(-1)
                    out2 = out2.reshape(-1)

                    errs = np.abs(out1 - out2)

                    esum = np.maximum(np.abs(out1), np.abs(out2))
                    esum[esum < 1e-7] = 1.

                    rtol = 1e-3
                    atol = 1e-3
                    tol = rtol * esum + atol

                    self.assertTrue(
                        np.all(errs < tol),
                        'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}'.format(
                            el, errs.max(), errs.mean(), errs.var())
                    )

            # check equivariance to icosahedron group
            # this basis is quite unstable so it doesn't pass the equivariance check
            cl.check_equivariance(device = 'cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    unittest.main()
