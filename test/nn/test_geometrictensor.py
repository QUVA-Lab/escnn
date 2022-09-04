import unittest
from unittest import TestCase

from escnn.nn import *
from escnn.gspaces import *
import torch
import random

import numpy as np


class TestGeometricTensor(TestCase):
    
    def test_sum(self):
        for N in [2, 4, 7, 16]:
            gs = rot2dOnR2(N)
            for irr in gs.irreps:
                type = FieldType(gs, [irr] * 3)
                for i in range(3):
                    t1 = GeometricTensor(torch.randn(10, type.size, 11, 11), type)
                    t2 = GeometricTensor(torch.randn(10, type.size, 11, 11), type)
                    
                    out1 = t1.tensor + t2.tensor
                    out2 = (t1 + t2).tensor
                    out3 = (t2 + t1).tensor
                    
                    self.assertTrue(torch.allclose(out1, out2))
                    self.assertTrue(torch.allclose(out3, out2))
    
    def test_isum(self):
        for N in [2, 4, 7, 16]:
            gs = rot2dOnR2(N)
            for irr in gs.irreps:
                type = FieldType(gs, [irr] * 3)
                for i in range(5):
                    t1 = GeometricTensor(torch.randn(10, type.size, 11, 11), type)
                    t2 = GeometricTensor(torch.randn(10, type.size, 11, 11), type)
                    
                    out1 = t1.tensor + t2.tensor
                    t1 += t2
                    out2 = t1.tensor
                    
                    self.assertTrue(torch.allclose(out1, out2))
    
    def test_sub(self):
        for N in [2, 4, 7, 16]:
            gs = rot2dOnR2(N)
            for irr in gs.irreps:
                type = FieldType(gs, [irr]*3)
                for i in range(3):
                    t1 = GeometricTensor(torch.randn(10, type.size, 11, 11), type)
                    t2 = GeometricTensor(torch.randn(10, type.size, 11, 11), type)
                    
                    out1 = t1.tensor - t2.tensor
                    out2 = (t1 - t2).tensor
                    
                    self.assertTrue(torch.allclose(out1, out2))

    def test_isub(self):
        for N in [2, 4, 7, 16]:
            gs = rot2dOnR2(N)
            for irr in gs.irreps:
                type = FieldType(gs, [irr] * 3)
                for i in range(5):
                    t1 = GeometricTensor(torch.randn(10, type.size, 11, 11), type)
                    t2 = GeometricTensor(torch.randn(10, type.size, 11, 11), type)
                
                    out1 = t1.tensor - t2.tensor
                    t1 -= t2
                    out2 = t1.tensor
                
                    self.assertTrue(torch.allclose(out1, out2))

    def test_mul(self):
        for N in [2, 4, 7, 16]:
            gs = rot2dOnR2(N)
            for irr in gs.irreps:
                type = FieldType(gs, [irr] * 3)
                for i in range(3):
                    t1 = GeometricTensor(torch.randn(10, type.size, 11, 11), type)
                    
                    s = 10*torch.randn(1)
                
                    out1 = t1.tensor * s
                    out2 = (s * t1).tensor
                    out3 = (t1 * s).tensor
                
                    self.assertTrue(torch.allclose(out1, out2))
                    self.assertTrue(torch.allclose(out3, out2))

    def test_imul(self):
        for N in [2, 4, 7, 16]:
            gs = rot2dOnR2(N)
            for irr in gs.irreps:
                type = FieldType(gs, [irr] * 3)
                for i in range(5):
                    t1 = GeometricTensor(torch.randn(10, type.size, 11, 11), type)
                    s = 10*torch.randn(1)

                    out1 = t1.tensor * s
                    t1 *= s
                    out2 = t1.tensor
                
                    self.assertTrue(torch.allclose(out1, out2))

    def test_slicing(self):
        for N in [2, 4, 7, 16]:
            gs = flipRot2dOnR2(N)
            for irr in gs.irreps:
                # with multiple fields
                F = 7
                type = FieldType(gs, [irr] * F)
                for i in range(3):
                    t = torch.randn(10, type.size, 11, 11)
                    gt = GeometricTensor(t, type)
        
                    # slice all dims except the channels
                    self.assertTrue(torch.allclose(
                        t[2:3, :, 2:7, 2:7],
                        gt[2:3, :, 2:7, 2:7].tensor,
                    ))
        
                    # slice only spatial dims
                    self.assertTrue(torch.allclose(
                        t[:, :, 2:7, 2:7],
                        gt[:, :, 2:7, 2:7].tensor,
                    ))
        
                    self.assertTrue(torch.allclose(
                        t[:, :, 2:7, 2:7],
                        gt[..., 2:7, 2:7].tensor,
                    ))
        
                    # slice only 1 spatial
                    self.assertTrue(torch.allclose(
                        t[..., 2:7],
                        gt[..., 2:7].tensor,
                    ))
        
                    # slice only batch
                    self.assertTrue(torch.allclose(
                        t[2:4],
                        gt[2:4, ...].tensor,
                    ))
                    self.assertTrue(torch.allclose(
                        t[2:4],
                        gt[2:4].tensor,
                    ))
        
                    # different ranges
                    self.assertTrue(torch.allclose(
                        t[:, :, 1:9:2, 0:8:3],
                        gt[..., 1:9:2, 0:8:3].tensor,
                    ))
        
                    # no slicing
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:].tensor,
                    ))
        
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, :, :, :].tensor,
                    ))
        
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, :, :].tensor,
                    ))
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, :].tensor,
                    ))
        
                    self.assertTrue(torch.allclose(
                        t,
                        gt[...].tensor,
                    ))
        
                    # slice channels with all fields of same type
                    self.assertTrue(torch.allclose(
                        t[:, 1 * irr.size:4 * irr.size:],
                        gt[:, 1:4, ...].tensor,
                    ))
                    # slice cover all channels
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, 0:7, ...].tensor,
                    ))
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, 0:7:1, ...].tensor,
                    ))
        
                    # with a larger step
                    start = 1
                    end = 6
                    step = 2
                    self.assertTrue(torch.allclose(
                        t[:,
                            [f * irr.size + i for f in range(start, end, step) for i in range(irr.size)]
                        ],
                        gt[:, start:end:step, ...].tensor,
                    ))
        
                    start = 0
                    end = 7
                    step = 3
                    self.assertTrue(torch.allclose(
                        t[:,
                            [f * irr.size + i for f in range(start, end, step) for i in range(irr.size)]
                        ],
                        gt[:, start:end:step, ...].tensor,
                    ))
        
                    # with negative step
                    start = 6
                    end = 1
                    step = -1
                    self.assertTrue(torch.allclose(
                        t[:,
                            [f * irr.size + i for f in range(start, end, step) for i in range(irr.size)]
                        ],
                        gt[:, start:end:step, ...].tensor,
                    ))
        
                    start = 6
                    end = 1
                    step = -2
                    self.assertTrue(torch.allclose(
                        t[:,
                            [f * irr.size + i for f in range(start, end, step) for i in range(irr.size)]
                        ],
                        gt[:, start:end:step, ...].tensor,
                    ))
                    
                    # 1 single field

                    start = 1
                    end = 2
                    step = 1
                    self.assertTrue(torch.allclose(
                        t[:,
                            [f * irr.size + i for f in range(start, end, step) for i in range(irr.size)]
                        ],
                        gt[:, start:end:step, ...].tensor,
                    ))
                    
                    # index only one field
                    f = 2
                    self.assertTrue(torch.allclose(
                        t[:,
                            [type.fields_start[f] + i for i in range(irr.size)]
                        ],
                        gt[:, f:f+1, ...].tensor,
                    ))
                    
                    # single index
                    f = 2
                    self.assertTrue(torch.allclose(
                        t[:,
                            [type.fields_start[f] + i for i in range(irr.size)]
                        ],
                        gt[:, f, ...].tensor,
                    ))
                    self.assertTrue(torch.allclose(
                        t[:,
                            [type.fields_start[f] + i for i in range(irr.size)]
                        ],
                        gt[:, f].tensor,
                    ))
                    
                    self.assertTrue(torch.allclose(
                        t[1:2],
                        gt[1, ...].tensor,
                    ))

                    self.assertTrue(torch.allclose(
                        t[..., 3:4],
                        gt[..., 3].tensor,
                    ))
                    self.assertTrue(torch.allclose(
                        t[..., 2:3, 3:4],
                        gt[..., 2, 3].tensor,
                    ))
                    self.assertTrue(torch.allclose(
                        t[3:4, ..., 2:3, 3:4],
                        gt[3, ..., 2, 3].tensor,
                    ))

                    self.assertTrue(torch.allclose(
                        t[1:2, :irr.size],
                        gt[1, 0, ...].tensor,
                    ))
                    
                    self.assertTrue(torch.allclose(
                        t[1:2, :irr.size, 4:5, 2:3],
                        gt[1, 0, 4, 2].tensor,
                    ))

                    # raise errors
                    with self.assertRaises(TypeError):
                        sliced = gt[2:5, 0:4, 1:7, 1:7, ...]
                        
                    with self.assertRaises(TypeError):
                        sliced = gt[[2, 4, 2], 0:4, ...]
                        
                    with self.assertRaises(TypeError):
                        sliced = gt[2, 0:4, range(3), range(3)]

                # with a single field
                F = 1
                type = FieldType(gs, [irr] * F)
                for i in range(3):
                    t = torch.randn(10, type.size, 11, 11)
                    gt = GeometricTensor(t, type)
    
                    # slice all dims except the channels
                    self.assertTrue(torch.allclose(
                        t[2:3, :, 2:7, 2:7],
                        gt[2:3, :, 2:7, 2:7].tensor,
                    ))
    
                    # slice only spatial dims
                    self.assertTrue(torch.allclose(
                        t[:, :, 2:7, 2:7],
                        gt[:, :, 2:7, 2:7].tensor,
                    ))
    
                    self.assertTrue(torch.allclose(
                        t[:, :, 2:7, 2:7],
                        gt[..., 2:7, 2:7].tensor,
                    ))
    
                    # slice only 1 spatial
                    self.assertTrue(torch.allclose(
                        t[..., 2:7],
                        gt[..., 2:7].tensor,
                    ))
    
                    # slice only batch
                    self.assertTrue(torch.allclose(
                        t[2:4],
                        gt[2:4, ...].tensor,
                    ))
                    self.assertTrue(torch.allclose(
                        t[2:4],
                        gt[2:4].tensor,
                    ))
    
                    # different ranges
                    self.assertTrue(torch.allclose(
                        t[:, :, 1:9:2, 0:8:3],
                        gt[..., 1:9:2, 0:8:3].tensor,
                    ))
    
                    # no slicing
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:].tensor,
                    ))
    
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, :, :, :].tensor,
                    ))
    
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, :, :].tensor,
                    ))
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, :].tensor,
                    ))
    
                    self.assertTrue(torch.allclose(
                        t,
                        gt[...].tensor,
                    ))
                    
                    # 1 single field
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, 0:1, ...].tensor,
                    ))
                    
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, 0, ...].tensor,
                    ))

                    # negative index
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, -1, ...].tensor,
                    ))
                    
                    # with negative step
                    start = 0
                    end = -2
                    step = -1
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, start:end:step, ...].tensor,
                    ))

            for i in range(3):
                reprs = list(gs.representations.values())*3
    
                random.shuffle(reprs)
                type = FieldType(gs, reprs)
                F = len(type)
    
                t = torch.randn(3, type.size, 3, 4)
                gt = GeometricTensor(t, type)
                
                # assignment should not be allowed
                with self.assertRaises(TypeError):
                    gt[2, 1:3, ...] = torch.randn(gt[2, 1:3, ...].shape)
    
                # no slicing
                self.assertTrue(torch.allclose(
                    t,
                    gt[:].tensor,
                ))
    
                self.assertTrue(torch.allclose(
                    t,
                    gt[:, :, :, :].tensor,
                ))
    
                self.assertTrue(torch.allclose(
                    t,
                    gt[:, :, :].tensor,
                ))
                self.assertTrue(torch.allclose(
                    t,
                    gt[:, :].tensor,
                ))
    
                self.assertTrue(torch.allclose(
                    t,
                    gt[...].tensor,
                ))
    
                # slice channels with all fields of different types
                self.assertTrue(torch.allclose(
                    t[:, type.fields_start[1]:type.fields_end[3]:],
                    gt[:, 1:4, ...].tensor,
                ))
    
                # slice cover all channels
                self.assertTrue(torch.allclose(
                    t,
                    gt[:, 0:F, ...].tensor,
                ))
                self.assertTrue(torch.allclose(
                    t,
                    gt[:, 0:F:1, ...].tensor,
                ))
    
                # with a larger step
                start = 1
                end = 6
                step = 2
                self.assertTrue(torch.allclose(
                    t[:,
                    [type.fields_start[f] + i for f in range(start, end, step) for i in range(type.representations[f].size)]
                    ],
                    gt[:, start:end:step, ...].tensor,
                ))
    
                start = 0
                end = 7
                step = 3
                self.assertTrue(torch.allclose(
                    t[:,
                    [type.fields_start[f] + i for f in range(start, end, step) for i in range(type.representations[f].size)]
                    ],
                    gt[:, start:end:step, ...].tensor,
                ))
    
                # with negative step
                start = 6
                end = 1
                step = -1
                self.assertTrue(torch.allclose(
                    t[:,
                    [type.fields_start[f] + i for f in range(start, end, step) for i in range(type.representations[f].size)]
                    ],
                    gt[:, start:end:step, ...].tensor,
                ))
    
                start = 6
                end = 1
                step = -2
                self.assertTrue(torch.allclose(
                    t[:,
                    [type.fields_start[f] + i for f in range(start, end, step) for i in range(type.representations[f].size)]
                    ],
                    gt[:, start:end:step, ...].tensor,
                ))

                # single index
                
                for f in range(F):
                
                    self.assertTrue(torch.allclose(
                        t[:,
                            [type.fields_start[f] + i for i in range(type.representations[f].size)]
                        ],
                        gt[:, f, ...].tensor,
                    ))
                    
                    self.assertTrue(torch.allclose(
                        t[:,
                            [type.fields_start[f] + i for i in range(type.representations[f].size)]
                        ],
                        gt[:, f].tensor,
                    ))

                    self.assertTrue(torch.allclose(
                        t[1:2,
                            [type.fields_start[f] + i for i in range(type.representations[f].size)]
                        ],
                        gt[1, f, ...].tensor,
                    ))

                    self.assertTrue(torch.allclose(
                        t[
                            1:2,
                            [type.fields_start[f] + i for i in range(type.representations[f].size)],
                            3:4,
                            4:5
                        ],
                        gt[1, f, 3, 4].tensor,
                    ))

    def test_advanced_indexing(self):
        for N in [2, 4, 7, 16]:
            gs = flipRot2dOnR2(N)
            for irr in gs.irreps:
                # with multiple fields
                F = 7
                type = FieldType(gs, [irr] * F)
                for i in range(3):
                    B = 10
                    D = 11
                    t = torch.randn(B, type.size, D, D)
                    gt = GeometricTensor(t, type)

                    # index all dims except the channels
                    idx1 = torch.randint(0, B, size=(5,))
                    idx2 = torch.randint(0, D, size=(5,))
                    idx3 = torch.randint(0, D, size=(5,))

                    # index only spatial dims
                    self.assertTrue(torch.allclose(
                        t[:, :, idx2, :],
                        gt[:, :, idx2, :].tensor,
                    ))

                    self.assertTrue(torch.allclose(
                        t[..., idx3],
                        gt[..., idx3].tensor,
                    ))

                    # index only batch
                    self.assertTrue(torch.allclose(
                        t[idx1],
                        gt[idx1, ...].tensor,
                    ))
                    self.assertTrue(torch.allclose(
                        t[idx1],
                        gt[idx1].tensor,
                    ))

                    ####

                    self.assertTrue(torch.allclose(
                        t,
                        gt[torch.arange(B), ...].tensor,
                    ))
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, :, torch.arange(D), :].tensor,
                    ))
                    self.assertTrue(torch.allclose(
                        t,
                        gt[..., torch.arange(D)].tensor,
                    ))

                    # index consecutive channels with all fields of same type
                    self.assertTrue(torch.allclose(
                        t[:, 1 * irr.size:4 * irr.size:],
                        gt[:, torch.arange(1, 4), ...].tensor,
                    ))
                    # index cover all channels
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, torch.arange(len(type)), ...].tensor,
                    ))

                    # with random indeces
                    idx = torch.randint(0, len(type), size=(8,))
                    t_idx = idx.reshape(-1, 1) * irr.size + torch.arange(irr.size).reshape(1, -1)
                    t_idx = t_idx.reshape(-1)
                    self.assertTrue(torch.allclose(
                        t[:, t_idx],
                        gt[:, idx, ...].tensor,
                    ))

                    self.assertTrue(torch.allclose(
                        t[:,
                        [f * irr.size + i for f in idx for i in range(irr.size)]
                        ],
                        gt[:, idx, ...].tensor,
                    ))

                    # with negative indeces
                    idx = torch.randint(-len(type), len(type), size=(8,))
                    t_idx = idx.reshape(-1, 1) * irr.size + torch.arange(irr.size).reshape(1, -1)
                    t_idx = t_idx.reshape(-1)
                    self.assertTrue(torch.allclose(
                        t[:, t_idx],
                        gt[:, idx, ...].tensor,
                    ))

                    self.assertTrue(torch.allclose(
                        t[:,
                        [f * irr.size + i for f in idx for i in range(irr.size)]
                        ],
                        gt[:, idx, ...].tensor,
                    ))

                    # indexing over multiple dimensions drops the dimension and won't match the GeometricTensor spatial requirements
                    with self.assertRaises(AssertionError):
                        gt[idx1, :, idx2, idx3]

                    with self.assertRaises(AssertionError):
                        gt[..., idx2, idx3]

                # with a single field
                F = 1
                type = FieldType(gs, [irr] * F)
                for i in range(3):
                    B = 10
                    D = 11
                    t = torch.randn(B, type.size, D, D)
                    gt = GeometricTensor(t, type)

                    # index all dims except the channels
                    idx1 = torch.randint(0, B, size=(5,))
                    idx2 = torch.randint(0, D, size=(5,))
                    idx3 = torch.randint(0, D, size=(5,))

                    # index only spatial dims
                    self.assertTrue(torch.allclose(
                        t[:, :, idx2, :],
                        gt[:, :, idx2, :].tensor,
                    ))

                    self.assertTrue(torch.allclose(
                        t[..., idx3],
                        gt[..., idx3].tensor,
                    ))

                    # index only batch
                    self.assertTrue(torch.allclose(
                        t[idx1],
                        gt[idx1, ...].tensor,
                    ))
                    self.assertTrue(torch.allclose(
                        t[idx1],
                        gt[idx1].tensor,
                    ))

                    ####

                    self.assertTrue(torch.allclose(
                        t,
                        gt[torch.arange(B), ...].tensor,
                    ))
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, :, torch.arange(D), :].tensor,
                    ))
                    self.assertTrue(torch.allclose(
                        t,
                        gt[..., torch.arange(D)].tensor,
                    ))

                    # index consecutive channels with all fields of same type
                    self.assertTrue(torch.allclose(
                        t[:, 0 * irr.size:1 * irr.size:],
                        gt[:, torch.arange(0, 1), ...].tensor,
                    ))
                    # index cover all channels
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, torch.arange(len(type)), ...].tensor,
                    ))

                    # with random indeces
                    idx = torch.randint(0, len(type), size=(8,))
                    t_idx = idx.reshape(-1, 1) * irr.size + torch.arange(irr.size).reshape(1, -1)
                    t_idx = t_idx.reshape(-1)
                    self.assertTrue(torch.allclose(
                        t[:, t_idx],
                        gt[:, idx, ...].tensor,
                    ))

                    self.assertTrue(torch.allclose(
                        t[:,
                        [f * irr.size + i for f in idx for i in range(irr.size)]
                        ],
                        gt[:, idx, ...].tensor,
                    ))

                    # with negative indeces
                    idx = torch.randint(-len(type), len(type), size=(8,))
                    t_idx = idx.reshape(-1, 1) * irr.size + torch.arange(irr.size).reshape(1, -1)
                    t_idx = t_idx.reshape(-1)
                    self.assertTrue(torch.allclose(
                        t[:, t_idx],
                        gt[:, idx, ...].tensor,
                    ))

                    self.assertTrue(torch.allclose(
                        t[:,
                        [f * irr.size + i for f in idx for i in range(irr.size)]
                        ],
                        gt[:, idx, ...].tensor,
                    ))

                    # indexing over multiple dimensions drops the dimension and won't match the GeometricTensor spatial requirements
                    with self.assertRaises(AssertionError):
                        gt[idx1, :, idx2, idx3]

                    with self.assertRaises(AssertionError):
                        gt[..., idx2, idx3]

            for i in range(3):
                reprs = list(gs.representations.values()) * 3

                random.shuffle(reprs)
                type = FieldType(gs, reprs)
                F = len(type)

                B = 3
                D = 4
                t = torch.randn(B, type.size, D, D)
                gt = GeometricTensor(t, type)

                idx = torch.randint(0, len(type), size=(8,))
                idx1 = torch.randint(0, B, size=(5,))
                idx2 = torch.randint(0, D, size=(5,))
                idx3 = torch.randint(0, D, size=(5,))

                # assignment should not be allowed
                with self.assertRaises(TypeError):
                    gt[2, idx, ...] = torch.randn(gt[2, idx, ...].shape)

                # no indexing
                self.assertTrue(torch.allclose(
                    t,
                    gt[torch.arange(B), ...].tensor,
                ))
                self.assertTrue(torch.allclose(
                    t,
                    gt[:, torch.arange(len(type)), ...].tensor,
                ))
                self.assertTrue(torch.allclose(
                    t,
                    gt[:, :, torch.arange(D), :].tensor,
                ))
                self.assertTrue(torch.allclose(
                    t,
                    gt[..., torch.arange(D)].tensor,
                ))

                # with random indices over the channels
                self.assertTrue(torch.allclose(
                    t[:,
                    [type.fields_start[f] + i for f in idx for i in range(type.representations[f].size)]
                    ],
                    gt[:, idx, ...].tensor,
                ))

                # with random indices over all dims
                self.assertTrue(torch.allclose(
                    t[idx1, ...],
                    gt[idx1, ...].tensor,
                ))
                self.assertTrue(torch.allclose(
                    t[:, :, idx2, :],
                    gt[:, :, idx2, :].tensor,
                ))
                self.assertTrue(torch.allclose(
                    t[..., idx3],
                    gt[..., idx3].tensor,
                ))


                # with negative indeces
                idx = torch.randint(-len(type), len(type), size=(8,))
                self.assertTrue(torch.allclose(
                    t[
                        :,
                        [type.fields_start[f] + i for f in idx for i in range(type.representations[f].size)],
                    ],
                    gt[:, idx].tensor,
                ))
                self.assertTrue(torch.allclose(
                    t[
                        :,
                        [type.fields_start[f] + i for f in idx for i in range(type.representations[f].size)],
                        ...
                    ],
                    gt[:, idx, ...].tensor,
                ))

                # single index
                for f in range(F):
                    idx = torch.tensor([f])
                    self.assertTrue(torch.allclose(
                        t[
                            :,
                            [type.fields_start[f] + i for f in idx for i in range(type.representations[f].size)],
                        ],
                        gt[:, idx, ...].tensor,
                    ))

    def test_boolean_indexing(self):
        for N in [2, 4, 7, 16]:
            gs = flipRot2dOnR2(N)
            for irr in gs.irreps:
                # with multiple fields
                F = 7
                type = FieldType(gs, [irr] * F)
                for i in range(3):
                    B = 10
                    D = 11
                    t = torch.randn(B, type.size, D, D)
                    gt = GeometricTensor(t, type)

                    # index all dims except the channels
                    idx1 = torch.rand(B) > .7
                    idx2 = torch.rand(D) > .7
                    idx3 = torch.rand(D) > .7

                    # index only spatial dims

                    self.assertTrue(torch.allclose(
                        t[:, :, idx2, :],
                        gt[:, :, idx2, :].tensor,
                    ))

                    self.assertTrue(torch.allclose(
                        t[..., idx3],
                        gt[..., idx3].tensor,
                    ))

                    # index only batch
                    self.assertTrue(torch.allclose(
                        t[idx1],
                        gt[idx1, ...].tensor,
                    ))
                    self.assertTrue(torch.allclose(
                        t[idx1],
                        gt[idx1].tensor,
                    ))

                    ####

                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, torch.ones(len(type), dtype=torch.bool), ...].tensor,
                    ))
                    self.assertTrue(torch.allclose(
                        t,
                        gt[torch.ones(B, dtype=torch.bool), ...].tensor,
                    ))
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, :, torch.ones(D, dtype=torch.bool), :].tensor,
                    ))
                    self.assertTrue(torch.allclose(
                        t,
                        gt[..., torch.ones(D, dtype=torch.bool)].tensor,
                    ))

                    # index consecutive channels with all fields of same type
                    self.assertTrue(torch.allclose(
                        t[:, 1 * irr.size:4 * irr.size:],
                        gt[:, (torch.arange(len(type)) < 4) & (torch.arange(len(type)) > 0), ...].tensor,
                    ))
                    # index cover all channels
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, torch.ones(len(type), dtype=torch.bool), ...].tensor,
                    ))

                    # with random indeces
                    idx = torch.rand(len(type)) > .3
                    if not idx.any():
                        idx[np.random.randint(len(type))] = 1
                    t_idx = idx.reshape(-1, 1).expand(-1, irr.size).reshape(-1)
                    self.assertTrue(torch.allclose(
                        t[:, t_idx],
                        gt[:, idx, ...].tensor,
                    ))

                    self.assertTrue(torch.allclose(
                        t[:,
                        [f * irr.size + i for f in range(len(type)) for i in range(irr.size) if idx[f]]
                        ],
                        gt[:, idx, ...].tensor,
                    ))

                    # indexing over multiple dimensions drops the dimension and won't match the GeometricTensor spatial requirements
                    with self.assertRaises((AssertionError, IndexError)):
                        gt[idx1, :, idx2, idx3]

                    with self.assertRaises((AssertionError, IndexError)):
                        gt[..., idx2, idx3]

                # with a single field
                F = 1
                type = FieldType(gs, [irr] * F)
                for i in range(3):
                    B = 10
                    D = 11
                    t = torch.randn(B, type.size, D, D)
                    gt = GeometricTensor(t, type)

                    # index all dims except the channels
                    idx1 = torch.rand(B) > .7
                    idx2 = torch.rand(D) > .7
                    idx3 = torch.rand(D) > .7

                    # index only spatial dims

                    self.assertTrue(torch.allclose(
                        t[:, :, idx2, :],
                        gt[:, :, idx2, :].tensor,
                    ))

                    self.assertTrue(torch.allclose(
                        t[..., idx3],
                        gt[..., idx3].tensor,
                    ))

                    # index only batch
                    self.assertTrue(torch.allclose(
                        t[idx1],
                        gt[idx1, ...].tensor,
                    ))
                    self.assertTrue(torch.allclose(
                        t[idx1],
                        gt[idx1].tensor,
                    ))

                    ####

                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, torch.ones(len(type), dtype=torch.bool), ...].tensor,
                    ))
                    self.assertTrue(torch.allclose(
                        t,
                        gt[torch.ones(B, dtype=torch.bool), ...].tensor,
                    ))
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, :, torch.ones(D, dtype=torch.bool), :].tensor,
                    ))
                    self.assertTrue(torch.allclose(
                        t,
                        gt[..., torch.ones(D, dtype=torch.bool)].tensor,
                    ))

                    # index consecutive channels with all fields of same type
                    self.assertTrue(torch.allclose(
                        t[:, 0 * irr.size:1 * irr.size:],
                        gt[:, (torch.arange(len(type)) < 1) & (torch.arange(len(type)) >= 0), ...].tensor,
                    ))
                    # index cover all channels
                    self.assertTrue(torch.allclose(
                        t,
                        gt[:, torch.ones(len(type), dtype=torch.bool), ...].tensor,
                    ))

                    # with random indeces
                    idx = torch.ones(1, dtype=torch.bool)
                    t_idx = idx.reshape(-1, 1).expand(-1, irr.size).reshape(-1)
                    self.assertTrue(torch.allclose(
                        t[:, t_idx],
                        gt[:, idx, ...].tensor,
                    ))

                    self.assertTrue(torch.allclose(
                        t[:,
                        [f * irr.size + i for f in range(len(type)) for i in range(irr.size) if idx[f]]
                        ],
                        gt[:, idx, ...].tensor,
                    ))

            for i in range(3):
                reprs = list(gs.representations.values()) * 3

                random.shuffle(reprs)
                type = FieldType(gs, reprs)
                F = len(type)

                B = 3
                D = 4
                t = torch.randn(B, type.size, D, D)
                gt = GeometricTensor(t, type)

                idx = torch.rand(len(type)) > .5
                idx1 = torch.rand(B) > .7
                idx2 = torch.rand(D) > .7
                idx3 = torch.rand(D) > .7

                # assignment should not be allowed
                with self.assertRaises(TypeError):
                    gt[2, idx, ...] = torch.randn(gt[2, idx, ...].shape)

                # no indexing
                self.assertTrue(torch.allclose(
                    t,
                    gt[:, torch.ones(len(type), dtype=torch.bool), ...].tensor,
                ))
                self.assertTrue(torch.allclose(
                    t,
                    gt[torch.ones(B, dtype=torch.bool), ...].tensor,
                ))
                self.assertTrue(torch.allclose(
                    t,
                    gt[:, :, torch.ones(D, dtype=torch.bool), :].tensor,
                ))
                self.assertTrue(torch.allclose(
                    t,
                    gt[..., torch.ones(D, dtype=torch.bool)].tensor,
                ))

                # with random indices over the channels
                self.assertTrue(torch.allclose(
                    t[:,
                    [type.fields_start[f] + i for f in range(len(type)) for i in range(type.representations[f].size) if idx[f]]
                    ],
                    gt[:, idx, ...].tensor,
                ))

                # with random indices over all dims
                self.assertTrue(torch.allclose(
                    t[idx1, ...],
                    gt[idx1, ...].tensor,
                ))
                self.assertTrue(torch.allclose(
                    t[:, :, idx2, :],
                    gt[:, :, idx2, :].tensor,
                ))
                self.assertTrue(torch.allclose(
                    t[..., idx3],
                    gt[..., idx3].tensor,
                ))

                # single index
                for f in range(F):
                    idx = torch.zeros(len(type), dtype=torch.bool)
                    idx[f] = 1
                    self.assertTrue(torch.allclose(
                        t[
                        :,
                        [type.fields_start[f] + i for f in range(len(type)) for i in range(type.representations[f].size)
                         if idx[f]]
                        ],
                        gt[:, idx, ...].tensor,
                    ))

    def test_rmul(self):
        for N in [2, 4, 7, 16]:
            gs = rot2dOnR2(N)
            for irr in gs.irreps:
                type = FieldType(gs, [irr] * 3)
                for i in range(3):
                    t1 = GeometricTensor(torch.randn(10, type.size, 11, 11), type)

                    for _ in range(5):
                        g = gs.fibergroup.sample()

                        out1 = g @ t1
                        out2 = t1.transform_fibers(g)

                        self.assertTrue(torch.allclose(out1.tensor, out2.tensor))

    def test_directsum(self):
        for N in [2, 4, 7, 16]:
            gs = flipRot2dOnR2(N)
            irreps = gs.irreps

            for i in range(5):
                type1 = gs.type(*[irreps[i] for i in np.random.randint(len(irreps), size=(3,))])
                type2 = gs.type(*[irreps[i] for i in np.random.randint(len(irreps), size=(3,))])

                t1 = torch.randn(10, type1.size, 11, 11)
                gt1 = type1(t1)
                t2 = torch.randn(10, type2.size, 11, 11)
                gt2 = type2(t2)

                type12 = type1 + type2
                t12 = torch.cat([t1, t2], dim=1)

                gt12 = tensor_directsum([gt1, gt2])

                self.assertEquals(gt12.type, type12)

                self.assertTrue(torch.allclose(gt12.tensor, t12, atol=1e-6, rtol=1e-4))

    def test_field_type_eq(self):
        gs = flipRot2dOnR2(5)
        irreps = gs.irreps

        for i in range(5):
            reprs = [irreps[i] for i in np.random.randint(len(irreps), size=(3,))]
            type1 = gs.type(*reprs)
            type2 = gs.type(*reprs)

            self.assertEquals(type1, type2)


if __name__ == '__main__':
    unittest.main()
