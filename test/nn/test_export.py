import unittest
from unittest import TestCase

import escnn.nn.init as init
from escnn.nn import *
from escnn.group import *
from escnn.gspaces import *

import torch
from torch.optim import SGD
from torch import nn

import numpy as np


class TestExport(TestCase):
    
    def test_R2Conv(self):
        
        for gs in [rot2dOnR2(9), flipRot2dOnR2(7), flip2dOnR2(), trivialOnR2()]:
            for ks in [1, 5]:
                for pd in [0, 1, 2]:
                    for st in [1, 2]:
                        for d in [1, 3]:
                            for gr in [1, 3]:
                                for i in range(2):
                                
                                    c_in = 1 + np.random.randint(4)
                                    c_out = 1 + np.random.randint(4)
                                    
                                    c_in *= gr
                                    c_out *= gr

                                    f_in = FieldType(gs, [gs.regular_repr]*c_in)
                                    f_out = FieldType(gs, [gs.regular_repr]*c_out)

                                    conv = R2Conv(
                                        f_in, f_out,
                                        kernel_size=ks,
                                        padding=pd,
                                        stride=st,
                                        dilation=d,
                                        groups=gr,
                                        bias=True,
                                    )
                                    
                                    self.check_exported(conv)

    def test_R2Conv_mix(self):
    
        for gs in [rot2dOnR2(9), flipRot2dOnR2(7), flip2dOnR2(), trivialOnR2()]:
            gs.fibergroup._build_quotient_representations()
            for ks in [1, 5]:
                for pd in [0, 1, 2]:
                    for st in [1, 2]:
                        for d in [1, 3]:
                            
                            ft = FieldType(gs, list(gs.representations.values()))
                        
                            conv = R2Conv(
                                ft, ft,
                                kernel_size=ks,
                                padding=pd,
                                stride=st,
                                dilation=d,
                                groups=1,
                                bias=True,
                            )
                        
                            self.check_exported(conv)

    def test_R2ConvTransposed(self):
    
        for gs in [rot2dOnR2(9), flipRot2dOnR2(7), flip2dOnR2(), trivialOnR2()]:
            for ks in [1, 5]:
                for pd in [0, 1, 2]:
                    for st in [1, 2]:
                        for d in [1, 3]:
                            for gr in [1, 3]:
                                for i in range(2):
                                    c_in = 1 + np.random.randint(4)
                                    c_out = 1 + np.random.randint(4)
                                
                                    c_in *= gr
                                    c_out *= gr
                                
                                    f_in = FieldType(gs, [gs.regular_repr] * c_in)
                                    f_out = FieldType(gs, [gs.regular_repr] * c_out)
                                
                                    conv = R2ConvTransposed(
                                        f_in, f_out,
                                        kernel_size=ks,
                                        padding=pd,
                                        stride=st,
                                        dilation=d,
                                        groups=gr,
                                        bias=True,
                                    )
                                
                                    self.check_exported(conv)

    def test_R2ConvTransposed_mix(self):
    
        for gs in [rot2dOnR2(9), flipRot2dOnR2(7), flip2dOnR2(), trivialOnR2()]:
            gs.fibergroup._build_quotient_representations()
            for ks in [1, 5]:
                for pd in [0, 1, 2]:
                    for st in [1, 2]:
                        for d in [1, 3]:
                            ft = FieldType(gs, list(gs.representations.values()))
                        
                            conv = R2ConvTransposed(
                                ft, ft,
                                kernel_size=ks,
                                padding=pd,
                                stride=st,
                                dilation=d,
                                groups=1,
                                bias=True,
                            )
                        
                            self.check_exported(conv)

    def test_Linear(self):

        for G in [cyclic_group(2), cyclic_group(7), dihedral_group(4), octa_group(), ico_group()]:
            gs = no_base_space(G)

            for i in range(2):
                c_in = 1 + np.random.randint(10)
                c_out = 1 + np.random.randint(10)

                f_in = FieldType(gs, [gs.regular_repr] * c_in)
                f_out = FieldType(gs, [gs.regular_repr] * c_out)

                conv = Linear(
                    f_in, f_out,
                    bias=True,
                )

                self.check_exported(conv)

    def test_IIDBatchNorm2d(self):

        for gs in [rot2dOnR2(9), flipRot2dOnR2(7), flip2dOnR2(), trivialOnR2()]:
            for i in range(4):
                c_in = 1 + np.random.randint(4)

                f_in = FieldType(gs, [gs.regular_repr] * c_in)

                batchnorm = IIDBatchNorm2d(f_in, affine=True)
                self.check_exported(batchnorm)

                batchnorm = IIDBatchNorm2d(f_in, affine=False)
                self.check_exported(batchnorm)

    def test_IIDBatchNorm3d(self):

        for gs in [rot3dOnR3(2), flipRot3dOnR3(2), rot2dOnR3(8), trivialOnR3()]:

            f_in = FieldType(gs, gs.irreps)

            batchnorm = IIDBatchNorm3d(f_in, affine=True)
            self.check_exported(batchnorm)

            batchnorm = IIDBatchNorm3d(f_in, affine=False)
            self.check_exported(batchnorm)

    def test_IIDBatchNorm_mix(self):

        for gs in [rot2dOnR2(9), flipRot2dOnR2(7), flip2dOnR2(), trivialOnR2()]:
            gs.fibergroup._build_quotient_representations()
            reprs = [r for r in gs.representations.values()]
        
            f_in = FieldType(gs, reprs)
        
            batchnorm = IIDBatchNorm2d(f_in, affine=True)
            self.check_exported(batchnorm)
        
            batchnorm = IIDBatchNorm2d(f_in, affine=False)
            self.check_exported(batchnorm)

    def test_GBatchNorm_mix(self):

        for gs in [rot2dOnR2(9), flipRot2dOnR2(7), flip2dOnR2(), trivialOnR2()]:
            gs.fibergroup._build_quotient_representations()
            reprs = [r for r in gs.representations.values()]

            f_in = FieldType(gs, reprs*2)

            batchnorm = IIDBatchNorm2d(f_in, affine=True)
            self.check_exported(batchnorm)

            batchnorm = IIDBatchNorm2d(f_in, affine=False)
            self.check_exported(batchnorm)

    def test_InnerBatchNorm(self):
    
        for gs in [rot2dOnR2(9), flipRot2dOnR2(7), flip2dOnR2(), trivialOnR2()]:
            for i in range(4):
                c_in = 1 + np.random.randint(4)
            
                f_in = FieldType(gs, [gs.regular_repr] * c_in)
            
                batchnorm = InnerBatchNorm(f_in, affine=True)
                self.check_exported(batchnorm)
            
                batchnorm = InnerBatchNorm(f_in, affine=False)
                self.check_exported(batchnorm)

    def test_InnerBatchNorm_mix(self):
    
        for gs in [rot2dOnR2(9), flipRot2dOnR2(7), flip2dOnR2(), trivialOnR2()]:
            gs.fibergroup._build_quotient_representations()
            reprs = [r for r in gs.representations.values() if 'pointwise' in r.supported_nonlinearities]
            
            f_in = FieldType(gs, reprs)
        
            batchnorm = InnerBatchNorm(f_in, affine=True)
            self.check_exported(batchnorm)
        
            batchnorm = InnerBatchNorm(f_in, affine=False)
            self.check_exported(batchnorm)

    def test_PointwiseDropout(self):
    
        for gs in [rot2dOnR2(9), flipRot2dOnR2(7), flip2dOnR2(), trivialOnR2()]:
            for p in [1e-2, 1e-1, 5e-1, 8e-1]:
                for i in range(4):
                    c_in = 1 + np.random.randint(4)
                
                    f_in = FieldType(gs, [gs.regular_repr] * c_in)
                
                    relu = PointwiseDropout(f_in, p=p)
                    self.check_exported(relu)

    def test_R2Upsampling(self):
    
        for gs in [rot2dOnR2(9), flipRot2dOnR2(7), flip2dOnR2(), trivialOnR2()]:
            for m in ['bilinear', 'nearest']:
                for sf in [1, 2, 4]:
                    for ac in [True, False]:
                        for i in range(3):
                        
                            f_in = FieldType(gs, list(gs.representations.values()))
                        
                            upsample = R2Upsampling(
                                f_in,
                                scale_factor=sf,
                                mode=m,
                                align_corners=ac
                            )
                            self.check_exported(upsample)

    def test_PointwiseAdaptiveAvgPool(self):
    
        for gs in [rot2dOnR2(9), flipRot2dOnR2(7), flip2dOnR2(), trivialOnR2()]:
            for os in [1, 5, 11]:
                for i in range(3):
                    c_in = 1 + np.random.randint(4)
                
                    f_in = FieldType(gs, [gs.regular_repr] * c_in)
                
                    avgpool = PointwiseAdaptiveAvgPool(
                        f_in,
                        output_size=os
                    )
                    self.check_exported(avgpool)

    def test_PointwiseAdaptiveMaxPool(self):
    
        for gs in [rot2dOnR2(9), flipRot2dOnR2(7), flip2dOnR2(), trivialOnR2()]:
            for os in [1, 5, 11]:
                for i in range(3):
                    c_in = 1 + np.random.randint(4)
                
                    f_in = FieldType(gs, [gs.regular_repr] * c_in)
                
                    maxpool = PointwiseAdaptiveMaxPool(
                        f_in,
                        output_size=os
                    )
                    self.check_exported(maxpool)

    def test_PointwiseMaxPool(self):
    
        for gs in [rot2dOnR2(9), flipRot2dOnR2(7), flip2dOnR2(), trivialOnR2()]:
            for ks in [2, 3, 5]:
                for pd in range(min(ks-1, 2)):
                    for st in [1, 2]:
                        for d in [1, 3]:
                            for i in range(3):
                                c_in = 1 + np.random.randint(4)
                            
                                f_in = FieldType(gs, [gs.regular_repr] * c_in)
                            
                                maxpool = PointwiseMaxPool(
                                    f_in,
                                    kernel_size=ks,
                                    stride=st,
                                    padding=pd,
                                    dilation=d
                                )
                                self.check_exported(maxpool)

    ######

    def test_R3Upsampling(self):

        for gs in [rot2dOnR3(maximum_frequency=3), mirOnR3(), trivialOnR3()]:
            for m in ['trilinear', 'nearest']:
                for sf in [1, 2]:
                    for ac in [True, False]:
                        for i in range(2):
                            f_in = FieldType(gs, list(gs.representations.values()))

                            upsample = R3Upsampling(
                                f_in,
                                scale_factor=sf,
                                mode=m,
                                align_corners=ac
                            )
                            self.check_exported(upsample)

    def test_PointwiseAdaptiveAvgPool3D(self):

        for gs in [rot2dOnR3(n=4), dihedralOnR3(n=4), trivialOnR3()]:
            for os in [1, 5, 11]:
                for i in range(3):
                    c_in = 1 + np.random.randint(4)

                    f_in = FieldType(gs, [gs.regular_repr] * c_in)

                    avgpool = PointwiseAdaptiveAvgPool3D(
                        f_in,
                        output_size=os
                    )
                    self.check_exported(avgpool)

    def test_PointwiseAdaptiveMaxPool3D(self):

        for gs in [rot2dOnR3(n=4), dihedralOnR3(n=4), trivialOnR3()]:
            for os in [1, 5, 11]:
                for i in range(3):
                    c_in = 1 + np.random.randint(4)

                    f_in = FieldType(gs, [gs.regular_repr] * c_in)

                    maxpool = PointwiseAdaptiveMaxPool3D(
                        f_in,
                        output_size=os
                    )
                    self.check_exported(maxpool)

    def test_PointwiseMaxPool3D(self):

        for gs in [rot2dOnR3(n=4), dihedralOnR3(n=4), trivialOnR3()]:
            for ks in [2, 3, 5]:
                for pd in range(min(ks - 1, 2)):
                    for st in [1, 2]:
                        for d in [1, 3]:
                            for i in range(3):
                                c_in = 1 + np.random.randint(4)

                                f_in = FieldType(gs, [gs.regular_repr] * c_in)

                                maxpool = PointwiseMaxPool3D(
                                    f_in,
                                    kernel_size=ks,
                                    stride=st,
                                    padding=pd,
                                    dilation=d
                                )
                                self.check_exported(maxpool)

    ######

    def test_ReLU(self):

        for gs in [rot2dOnR2(9), flipRot2dOnR2(7), flip2dOnR2(), trivialOnR2()]:
            for i in range(4):
                c_in = 1 + np.random.randint(4)

                f_in = FieldType(gs, [gs.regular_repr] * c_in)

                relu = ReLU(f_in)
                self.check_exported(relu)

    def test_ELU(self):
    
        for gs in [rot2dOnR2(9), flipRot2dOnR2(7), flip2dOnR2(), trivialOnR2()]:
            for i in range(4):
                c_in = 1 + np.random.randint(4)
            
                f_in = FieldType(gs, [gs.regular_repr] * c_in)
            
                elu = ELU(f_in)
                self.check_exported(elu)

    def test_GPool_unique(self):
    
        for gs in [rot2dOnR2(9), flipRot2dOnR2(7), flip2dOnR2(), trivialOnR2()]:
            gs.fibergroup._build_quotient_representations()
            for repr in gs.representations.values():
                if 'pointwise' in repr.supported_nonlinearities:
                
                    for i in range(4):
                        c_in = 1 + np.random.randint(8)
                    
                        f_in = FieldType(gs, [repr] * c_in)
                    
                        gpool = GroupPooling(f_in)
                        self.check_exported(gpool)

    def test_Sequential(self):
    
        for gs in [rot2dOnR2(9), flipRot2dOnR2(7), flip2dOnR2(), trivialOnR2()]:
            gs.fibergroup._build_quotient_representations()
            reprs = [r for r in gs.representations.values() if 'pointwise' in r.supported_nonlinearities]
    
            f_in = FieldType(gs, reprs)
            f_out = FieldType(gs, [gs.regular_repr]*1)
            
            for i in range(20):
                net = SequentialModule(
                    R2Conv(f_in, f_in, 7, bias=True),
                    InnerBatchNorm(f_in, affine=True),
                    ReLU(f_in, inplace=True),
                    PointwiseMaxPool(f_in, 3, 2, 1),
                    R2Conv(f_in, f_out, 3, bias=True),
                    InnerBatchNorm(f_out, affine=False),
                    ELU(f_out, inplace=True),
                    GroupPooling(f_out),
                )

                self.check_exported(net, atol=1e-6, rtol=5e-4)
                
                self.check_state_dict(net)

    def test_Sequential_example(self):
    
        s = rot2dOnR2(8)
        c_in = FieldType(s, [s.trivial_repr] * 3)
        c_hid = FieldType(s, [s.regular_repr] * 3)
        c_out = FieldType(s, [s.regular_repr] * 1)
    
        net = SequentialModule(
            R2Conv(c_in, c_hid, 5, bias=False),
            InnerBatchNorm(c_hid),
            ReLU(c_hid, inplace=True),
            PointwiseMaxPool(c_hid, kernel_size=3, stride=2, padding=1),
            R2Conv(c_hid, c_out, 3, bias=False),
            InnerBatchNorm(c_out),
            ELU(c_out, inplace=True),
            GroupPooling(c_out)
        )
    
        print(net)
        print(net.export())
        
        self.check_exported(net)
        
        self.check_state_dict(net)

    def check_exported(self, equivariant: EquivariantModule, atol=1e-8, rtol=1e-5):
        
        equivariant = train(equivariant)
    
        in_size = equivariant.in_type.size
        gspace = equivariant.in_type.gspace

        # D = 2 if isinstance(gspace, GSpace2D) else 3
        D = gspace.dimensionality

        conventional = equivariant.export()
        
        for _ in range(5):
            x = torch.randn(5, in_size, *(31,)*D)

            ye = equivariant(GeometricTensor(x, equivariant.in_type)).tensor
            yc = conventional(x)
            
            # print(torch.abs(ye-yc).max())
            
            self.assertTrue(torch.allclose(ye, yc, atol=atol, rtol=rtol))

    def check_state_dict(self, equivariant: EquivariantModule, atol=1e-8, rtol=1e-5):
    
        equivariant = train(equivariant)
        
        conventional1 = equivariant.export()

        equivariant = train(equivariant)

        conventional2 = equivariant.export()
        state_dict = conventional2.state_dict()
        
        conventional1.load_state_dict(state_dict)
        
        # check the two versions are equivalent
        in_size = equivariant.in_type.size
        for _ in range(20):
            x = torch.randn(5, in_size, 31, 31)
        
            y1 = conventional1(x)
            y2 = conventional2(x)
        
            # print(torch.abs(y1-y2).max())
        
            self.assertTrue(torch.allclose(y1, y2, atol=atol, rtol=rtol))


def train(equivariant: EquivariantModule):

    in_size = equivariant.in_type.size

    gspace = equivariant.in_type.gspace
    D = gspace.dimensionality

    equivariant.train()

    if len(list(equivariant.parameters())) > 0:
        sgd = SGD(equivariant.parameters(), lr=1e-3)
    else:
        sgd = None

    for i in range(5):
        x = torch.randn(5, in_size, *(31,) * D)
        x = GeometricTensor(x, equivariant.in_type)
    
        if sgd is not None:
            sgd.zero_grad()
    
        y = equivariant(x).tensor
        y = ((y - 1.) ** 2).mean()
    
        if sgd is not None:
            y.backward()
            sgd.step()
    
    return equivariant


if __name__ == '__main__':
    unittest.main()
