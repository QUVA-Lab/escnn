import unittest
from unittest import TestCase

from escnn.nn import *
from escnn.gspaces import *
from escnn.group import *

import torch
import numpy as np

import random


class TestBatchnorms(TestCase):
    
    def test_o2_iid_bnorm(self):
        N = 3
        g = flipRot2dOnR2(-1, maximum_frequency=N)
        
        reprs = list(g.representations.values())
        
        irreps_sum = []
        for irr in g.irreps:
            irreps_sum += [irr] * int(irr.size // irr.sum_of_squares_constituents)
        reprs += [directsum(irreps_sum)]
        
        r = FieldType(g, reprs)
        
        bn = IIDBatchNorm2d(r, affine=False, momentum=1.)
        
        self.check_bn(bn)
    
    def test_so2_iid_bnorm(self):
        N = 3
        g = rot2dOnR2(-1, maximum_frequency=N)
        
        reprs = list(g.representations.values())
        
        irreps_sum = []
        for irr in g.irreps:
            irreps_sum += [irr] * int(irr.size // irr.sum_of_squares_constituents)
        reprs += [directsum(irreps_sum)]
        
        r = FieldType(g, reprs)

        bn = IIDBatchNorm2d(r, affine=False, momentum=1.)
        
        self.check_bn(bn)
    
    def test_dihedral_general_bnorm(self):
        N = 8
        g = flipRot2dOnR2(N)
        
        r = FieldType(g, list(g.representations.values()))
        
        bn = GNormBatchNorm(r, affine=False, momentum=1.)
        
        self.check_bn(bn)

    def test_cyclic_general_bnorm(self):
        N = 32
        g = rot2dOnR2(N)

        r = FieldType(g, list(g.representations.values()))

        bn = GNormBatchNorm(r, affine=False, momentum=1.)

        self.check_bn(bn)

    def test_so3_general_bnorm(self):
        g = rot3dOnR3()

        r = FieldType(g, [g.fibergroup.bl_regular_representation(2)] * 3)

        bn = GNormBatchNorm(r, affine=False, momentum=1.)

        self.check_bn(bn)

    def test_so3_norm_bnorm(self):
        g = rot3dOnR3()
        
        r = FieldType(g, [g.irrep(l) for l in range(1, 4)]*2)
        
        bn = NormBatchNorm(r, affine=False, momentum=1.)
        
        self.check_bn(bn)

    def test_cyclic_induced_norm(self):
        N = 8
        g = rot2dOnR2(N)

        sg_id = 4

        sg, _, _ = g.fibergroup.subgroup(sg_id)

        r = FieldType(g, [g.induced_repr(sg_id, r) for r in sg.irreps() if not r.is_trivial()])

        bn = InducedNormBatchNorm(r, affine=False, momentum=1.)

        self.check_bn(bn)

    def test_dihedral_induced_norm(self):
        N = 8
        g = flipRot2dOnR2(N)
        
        sg_id = (None, N)
        
        sg, _, _ = g.fibergroup.subgroup(sg_id)
    
        r = FieldType(g, [g.induced_repr(sg_id, r) for r in sg.irreps() if not r.is_trivial()])
    
        bn = InducedNormBatchNorm(r, affine=False, momentum=1.)
    
        self.check_bn(bn)

    def test_dihedral_inner_norm(self):
        N = 8
        g = flipRot2dOnR2(N)
        
        g.fibergroup._build_quotient_representations()
        
        reprs = []
        for r in g.representations.values():
            if 'pointwise' in r.supported_nonlinearities:
                reprs.append(r)
        
        r = FieldType(g, reprs)
    
        bn = InnerBatchNorm(r, affine=False, momentum=1.)
    
        self.check_bn(bn)

    def test_ico_inner_norm(self):
        g = icoOnR3()

        for sg in [
            (False, 1),
            (False, 2),
            (False, 3),
            (False, 5),
            (True, 1),
            # (True, 2),
            # (True, 3),
            # (True, 5),
        ]:
            print(sg)
            g.fibergroup.quotient_representation(sg)

        reprs = []
        for r in g.representations.values():
            if 'pointwise' in r.supported_nonlinearities:
                reprs.append(r)

        r = FieldType(g, reprs)

        bn = InnerBatchNorm(r, affine=False, momentum=1.)

        self.check_bn(bn)

    def check_bn(self, bn: EquivariantModule):

        d = bn.in_type.gspace.dimensionality
        D = 500
        x = 15*torch.randn(D, bn.in_type.size, *[1]*d)
        
        p = 0
        trivials = []
        for irr in bn.in_type.irreps:
            irr = bn.in_type.fibergroup.irrep(*irr)
            if irr.is_trivial():
                trivials.append(p)
            p += irr.size
        
        if len(trivials) > 0:
            bias = 2*torch.randn(D, len(trivials), *[1]*d) + 5.
            Q = torch.tensor(bn.in_type.representation.change_of_basis[:, trivials], dtype=torch.float)
            bias = torch.einsum('ij,bj...->bi...', Q, bias)
            x += bias
        
        G = list(bn.in_type.gspace.testing_elements)
        for i in range(D):
            g = G[np.random.randint(len(G))]
            central_slice = i, slice(None), *[0]*d
            x[central_slice] = torch.tensor(bn.in_type.representation(g), dtype=torch.float) @ x[central_slice]

        x = GeometricTensor(x, bn.in_type)
        
        bn.reset_running_stats()
        bn.train()
        
        # for name, size in bn._sizes:
        #     running_var = getattr(bn, f"{name}_running_var")
        #     running_mean = getattr(bn, f"{name}_running_mean")
        #     if running_mean.numel() > 0:
        #         self.assertTrue(torch.allclose(running_mean, torch.zeros_like(running_mean)))
        #     self.assertTrue(torch.allclose(running_var, torch.ones_like(running_var)))

        bn(x)
        
        bn.eval()
        # for name, size in bn._sizes:
        #     running_var = getattr(bn, f"{name}_running_var")
        #     running_mean = getattr(bn, f"{name}_running_mean")
        #     print(name)
        #     if running_mean.numel() > 0:
        #         print(running_mean.abs().max())
        #     print(running_var.abs().max())
        
        ys = []
        xs = []
        for g in bn.in_type.gspace.testing_elements:
            x_g = x.transform_fibers(g)
            y_g = bn(x_g)
            xs.append(x_g.tensor)
            ys.append(y_g.tensor)
        
        xs = torch.cat(xs, dim=0)
        ys = torch.cat(ys, dim=0)
        
        np.set_printoptions(precision=5, suppress=True, threshold=1000000, linewidth=3000)
        
        print([r.name for r in bn.out_type.representations])
        
        mean = xs.mean(0)
        std = xs.std(0)
        print('Pre-normalization stats')
        print(mean.view(-1).detach().numpy())
        print(std.view(-1).detach().numpy())

        mean = ys.mean(0)
        std = ys.std(0)
        print('Post-normalization stats')
        print(mean.view(-1).detach().numpy())
        print(std.view(-1).detach().numpy())
        
        self.assertTrue(torch.allclose(mean, torch.zeros_like(mean), rtol=4e-2, atol=2e-2))
        self.assertTrue(torch.allclose(std, torch.ones_like(std), rtol=4e-2, atol=2e-2), msg=f"""
            Standard deviations after normalization: \n {std.cpu().numpy().reshape(-1)}
        """)

    def test_gbnorm_so2(self):
        g = rot2dOnR2()
        r = g.type(*[g.fibergroup.bl_regular_representation(2)])
        self.check_gbatchnorm(r)

    def test_gbnorm_so3(self):
        g = rot3dOnR3()
        r = g.type(*[g.fibergroup.bl_regular_representation(2)]*3)
        self.check_gbatchnorm(r)

    def check_gbatchnorm(self, ft: FieldType, affine: bool=True):

        bnorm1 = GNormBatchNorm(ft, affine=affine, momentum=1.)
        d = ft.gspace.dimensionality

        ft2 = ft.gspace.type(*[ft.fibergroup.irrep(*irr) for irr in ft.irreps])
        if d == 1:
            bnorm2 = IIDBatchNorm1d(ft2, affine=affine, momentum=1.)
        elif d == 2:
            bnorm2 = IIDBatchNorm2d(ft2, affine=affine, momentum=1.)
        elif d == 3:
            bnorm2 = IIDBatchNorm3d(ft2, affine=affine, momentum=1.)
        else:
            raise ValueError

        D = 500
        bnorm1.reset_running_stats()
        bnorm1.train()
        bnorm2.reset_running_stats()
        bnorm2.train()

        for _ in range(5):
            x = 15 * torch.randn(D, ft.size, *[1] * d)

            x1 = ft(x)
            x2 = ft2(x)

            bnorm1(x1)
            bnorm2(x2)

        bnorm1.eval()
        bnorm2.eval()

        bnorm1 = bnorm1.export()
        bnorm2 = bnorm2.export()

        self.assertTrue(torch.allclose(bnorm1.running_mean, bnorm2.running_mean, rtol=4e-2, atol=2e-2))
        self.assertTrue(torch.allclose(bnorm1.running_var, bnorm2.running_var, rtol=4e-2, atol=2e-2))
        if affine:
            self.assertTrue(torch.allclose(bnorm1.bias, bnorm2.bias, rtol=4e-2, atol=2e-2))
            self.assertTrue(torch.allclose(bnorm1.weight, bnorm2.weight, rtol=4e-2, atol=2e-2))

    def test_fieldnorm_so2(self):
        g = rot2dOnR2(maximum_frequency=4)
        F = 4
        r = g.type(g.fibergroup.bl_regular_representation(F), *[g.fibergroup.irrep(*i) for i in g.fibergroup.bl_irreps(F)])
        self.check_fieldnorm(r)

    def check_fieldnorm(self, ft: FieldType):

        bn = FieldNorm(ft, affine=False)

        d = bn.in_type.gspace.dimensionality
        D = 500
        x = 15 * torch.randn(D, bn.in_type.size, *[1] * d)

        p = 0
        trivials = []
        for irr in bn.in_type.irreps:
            irr = bn.in_type.fibergroup.irrep(*irr)
            if irr.is_trivial():
                trivials.append(p)
            p += irr.size

        if len(trivials) > 0:
            bias = 2 * torch.randn(D, len(trivials), *[1] * d) + 5.
            Q = torch.tensor(bn.in_type.representation.change_of_basis[:, trivials], dtype=torch.float)
            bias = torch.einsum('ij,bj...->bi...', Q, bias)
            x += bias

        for i in range(D):
            g = bn.in_type.gspace.fibergroup.sample()
            central_slice = i, slice(None), *[0] * d
            x[central_slice] = torch.tensor(bn.in_type.representation(g), dtype=torch.float) @ x[central_slice]

        x = GeometricTensor(x, bn.in_type)

        bn.reset_running_stats()
        bn.train()

        bn(x)
        # exit()

        bn.eval()

        ys = []
        xs = []
        x = x[:1]
        for i in range(10000):
            g = bn.in_type.gspace.fibergroup.sample()
        # for g in bn.in_type.gspace.fibergroup.testing_elements():

            x_g = x.transform_fibers(g)
            y_g = bn(x_g)
            xs.append(x_g.tensor)
            ys.append(y_g.tensor)

        xs = torch.cat(xs, dim=0)
        ys = torch.cat(ys, dim=0)

        np.set_printoptions(precision=5, suppress=True, threshold=1000000, linewidth=3000)

        mean = xs.mean(0)
        std = xs.std(0)
        print('Pre-normalization stats')
        print(mean.view(-1).detach().numpy())
        print(std.view(-1).detach().numpy())

        mean = ys.mean(0)
        std = ys.std(0)
        print('Post-normalization stats')
        print(mean.view(-1).detach().numpy())
        print(std.view(-1).detach().numpy())

        proj_std = torch.zeros(len(ft))
        stdones = torch.ones_like(proj_std)
        p=0
        for i, repr in enumerate(ft.representations):
            proj_std[i] = (std[p:p+repr.size]**2).mean().sqrt()
            if repr.is_trivial():
                stdones[i] = 0.
            p+=repr.size
        print(proj_std.view(-1).detach().numpy())

        self.assertTrue(torch.allclose(mean, torch.zeros_like(mean), rtol=4e-2, atol=2e-2))
        self.assertTrue(torch.allclose(proj_std, stdones, rtol=4e-2, atol=2e-2), msg=f"""
            Standard deviations after normalization: \n {proj_std.cpu().numpy().reshape(-1)}
        """)


if __name__ == '__main__':
    unittest.main()
