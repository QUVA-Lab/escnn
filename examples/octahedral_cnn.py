import torch

from escnn import gspaces
from escnn import nn

import numpy as np
np.set_printoptions(precision=3, linewidth=10000, suppress=True)


def main():
    # code based on https://github.com/QUVA-Lab/escnn/blob/master/examples/introduction.ipynb, which is SO(2) case
    octa_act = gspaces.octaOnR3()

    feat_type_in = nn.FieldType(octa_act, [octa_act.trivial_repr])
    feat_type_out = nn.FieldType(octa_act, [octa_act.regular_repr])

    conv = nn.R3Conv(feat_type_in, feat_type_out, kernel_size=3)
    relu = nn.ReLU(feat_type_out)
    x = torch.randn(1, 1, 3, 3, 3)
    x = feat_type_in(x)

    assert isinstance(x.tensor, torch.Tensor)
    y = conv(x)
    z = relu(y)
    assert z.type == feat_type_out

    # for each group element
    for g in octa_act.testing_elements:
        x_transformed = x.transform(g)
        z_from_x_transformed = relu(conv(x_transformed))
        z_transformed_from_x = z.transform(g)
        assert torch.allclose(z_from_x_transformed.tensor, z_transformed_from_x.tensor, atol=1e-5), g


if __name__ == "__main__":
    main()
