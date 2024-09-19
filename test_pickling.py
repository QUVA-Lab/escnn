import pickle
import tempfile
import escnn
from escnn import nn, gspaces
import torch


def test_pickling(r2_act):
    in_type = nn.FieldType(r2_act, 1 * [r2_act.trivial_repr])
    out_type = nn.FieldType(r2_act, 1 * [r2_act.regular_repr])

    module = torch.nn.Sequential(
        nn.R2Conv(in_type, out_type, 1),
    )

    with tempfile.TemporaryFile() as f:
        pickle.dump(module, f)


if __name__ == "__main__":
    r2_acts = [
        gspaces.trivialOnR2(),
        gspaces.rot2dOnR2(N=4),
        gspaces.flip2dOnR2(),
        gspaces.flipRot2dOnR2(N=4),
    ]
    for r2_act in r2_acts:
        print(str(r2_act))
        test_pickling(r2_act)

