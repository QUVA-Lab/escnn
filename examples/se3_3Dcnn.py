
from typing import Tuple, Union
from collections import defaultdict

from escnn.group import *
from escnn.gspaces import *
from escnn.nn import *


import torch
from torch import nn
import numpy as np

from scipy import stats


import math


class ResBlock(EquivariantModule):

    def __init__(self, in_type: FieldType, channels: int, out_type: FieldType = None, stride: int = 1, features: str = '2_96'):

        super(ResBlock, self).__init__()

        self.in_type = in_type
        if out_type is None:
            self.out_type = self.in_type
        else:
            self.out_type = out_type

        self.gspace = self.in_type.gspace

        if features == 'ico':
            L = 2
            grid = {'type': 'ico'}
        elif features == '2_96':
            L = 2
            grid = {'type': 'thomson_cube', 'N': 4}
        elif features == '2_72':
            L = 2
            grid = {'type': 'thomson_cube', 'N': 3}
        elif features == '3_144':
            L = 3
            grid = {'type': 'thomson_cube', 'N': 6}
        elif features == '3_192':
            L = 3
            grid = {'type': 'thomson_cube', 'N': 8}
        elif features == '3_160':
            L = 3
            grid = {'type': 'thomson', 'N': 160}
        else:
            raise ValueError()

        so3: SO3 = self.in_type.fibergroup

        # number of samples for the discrete Fourier Transform
        S = len(so3.grid(**grid))

        # We try to keep the width of the model approximately constant
        _channels = channels / S
        print(f'{_channels} ({int(round(_channels))}) vs {channels} | {60 / S}')
        _channels = int(round(_channels))

        # Build the non-linear layer
        # Internally, this module performs an Inverse FT sampling the `_channels` continuous input features on the `S`
        # samples, apply ELU pointwise and, finally, recover `_channels` output features with discrete FT.
        ftelu = FourierELU(self.gspace, _channels, irreps=so3.bl_irreps(L), inplace=True, **grid)
        res_type = ftelu.in_type

        print(f'ResBlock: {in_type.size} -> {res_type.size} | {S*_channels}')

        self.res_block = SequentialModule(
            R3Conv(in_type, res_type, kernel_size=3, padding=1, bias=False, initialize=False),
            IIDBatchNorm3d(res_type, affine=True),
            ftelu,
            R3Conv(res_type, self.out_type, kernel_size=3, padding=1, stride=stride, bias=False, initialize=False),
        )

        if stride > 1:
            self.downsample = PointwiseAvgPoolAntialiased3D(in_type, .33, 2, 1)
        else:
            self.downsample = lambda x: x

        if self.in_type != self.out_type:
            self.skip = R3Conv(self.in_type, self.out_type, kernel_size=1, padding=0, bias=False)
        else:
            self.skip = lambda x: x

    def forward(self, input: GeometricTensor):

        assert input.type == self.in_type
        return self.skip(self.downsample(input)) + self.res_block(input)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:

        if self.in_type != self.out_type:
            return input_shape[:1] + (self.out_type.size, ) + input_shape[2:]
        else:
            return input_shape


class SE3CNN(nn.Module):

    def __init__(self, pool: str = "snub_cube", res_features: str = '2_96', init: str = 'delta'):

        super(SE3CNN, self).__init__()

        self.gs = rot3dOnR3()

        self.in_type = FieldType(self.gs, [self.gs.trivial_repr])

        self._init = init

        layer_types = [
            (FieldType(self.gs, [self.taylor_repr(2)] * 3), 240),
            (FieldType(self.gs, [self.taylor_repr(3)] * 2), 480),
            (FieldType(self.gs, [self.taylor_repr(3)] * 6), 480),
            (FieldType(self.gs, [self.taylor_repr(3)] * 12), 960),
            (FieldType(self.gs, [self.taylor_repr(3)] * 8), None),
        ]

        blocks = [
            R3Conv(self.in_type, layer_types[0][0], kernel_size=5, padding=2, bias=False, initialize=False)
        ]

        for i in range(len(layer_types) - 1):
            blocks.append(
                ResBlock(layer_types[i][0], layer_types[i][1], layer_types[i+1][0], 2, features=res_features)
            )

        # For pooling, we map the features to a spherical representation (bandlimited to freq 2)
        # Then, we apply pointwise ELU over a number of samples on the sphere and, finally, compute the average
        # # (i.e. recover only the frequency 0 component of the output features)
        if pool == "icosidodecahedron":
            # samples the 30 points of the icosidodecahedron
            # this is only perfectly equivarint to the 12 tethrahedron symmetries
            grid = self.gs.fibergroup.sphere_grid(type='ico')
        elif pool == "snub_cube":
            # samples the 24 points of the snub cube
            # this is perfectly equivariant to all 24 rotational symmetries of the cube
            grid = self.gs.fibergroup.sphere_grid(type='thomson_cube', N=1)
        else:
            raise ValueError(f"Pooling method {pool} not recognized")

        ftgpool = QuotientFourierELU(self.gs, (False, -1), 128, irreps=self.gs.fibergroup.bl_irreps(2), out_irreps=self.gs.fibergroup.bl_irreps(0), grid=grid)

        final_features = ftgpool.in_type
        blocks += [
            R3Conv(layer_types[-1][0], final_features, kernel_size=3, padding=0, bias=False, initialize=False),
            ftgpool,
        ]
        C = ftgpool.out_type.size

        self.blocks = SequentialModule(*blocks)

        H = 256
        self.classifier = nn.Sequential(
            nn.Linear(C, H, bias=False),

            nn.BatchNorm1d(H, affine=True),
            nn.ELU(inplace=True),
            nn.Dropout(.1),
            nn.Linear(H, H // 2, bias=False),

            nn.BatchNorm1d(H // 2, affine=True),
            nn.ELU(inplace=True),
            nn.Dropout(.1),
            nn.Linear(H//2, 10, bias=True),
        )

    def init(self):
        for name, m in self.named_modules():
            if isinstance(m, R3Conv):
                if self._init == 'he':
                    init.generalized_he_init(m.weights.data, m.basisexpansion, cache=True)
                elif self._init == 'delta':
                    init.deltaorthonormal_init(m.weights.data, m.basisexpansion)
                elif self._init == 'rand':
                    m.weights.data[:] = torch.randn_like(m.weights)
                else:
                    raise ValueError()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                o, i = m.weight.shape
                m.weight.data[:] = torch.tensor(stats.ortho_group.rvs(max(i, o))[:o, :i])
                if m.bias is not None:
                    m.bias.data.zero_()

    def taylor_repr(self, K: int):
        assert K >= 0

        if K == 0:
            return [self.gs.trivial_repr]

        SO3 = self.gs.fibergroup

        polinomials = [self.gs.trivial_repr, SO3.irrep(1)]

        for k in range(2, K+1):
            polinomials.append(
                polinomials[-1].tensor(SO3.irrep(1))
            )

        return directsum(polinomials, name=f'taylor_{K}')

    def bl_sphere(self, K: int):
        assert K >= 0

        if K == 0:
            return [self.gs.trivial_repr]

        return self.gs.fibergroup.bl_quotient_representation(K, (False, -1))

    def bl_regular(self, K: int):
        assert K >= 0

        if K == 0:
            return [self.gs.trivial_repr]

        return self.gs.fibergroup.bl_regular_representation(K)

    def forward(self, input: torch.Tensor):

        input = GeometricTensor(input, self.in_type)

        features = self.blocks(input)

        shape = features.shape
        features = features.tensor.reshape(shape[0], shape[1])

        out = self.classifier(features)

        return out


if __name__ == '__main__':

    # from datetime import datetime

    net = SE3CNN(pool='snub_cube', res_features='2_96', init='delta')
    nparams = sum(p.numel() for p in net.parameters())
    print(f"Number of parameters: {int(nparams/1000)}")

    import time

    net.init()

    device = 'cuda'

    torch.backends.cudnn.benchmark = True

    with torch.no_grad():
        model = net.to(device)
        net.eval()

        durations = []
        for step in range(100):
            img = torch.randn(8, 1, 33, 33, 33, device=device)
            torch.cuda.synchronize()
            start = time.time()
            net(img)
            torch.cuda.synchronize()
            end = time.time()
            durations.append((end - start) * 1000)
            # print((end - start) * 1000)
    WARMUP = 10
    filtered_durations = durations[WARMUP:]
    filtered_durations = np.array(filtered_durations)
    # avg_duration = sum(filtered_durations) / len(filtered_durations)
    # print(avg_duration)
    print(filtered_durations.mean(), filtered_durations.std())
    del net

