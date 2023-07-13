
from escnn.nn import GeometricTensor
from escnn.nn import FieldType
from escnn.gspaces import GSpace
from .equivariant_module import EquivariantModule

from itertools import product, repeat
from typing import Tuple

import torch
import numpy as np

import math

__all__ = ["MaskModule"]


def build_mask(
        s,
        dim: int = 2,
        margin: float = 2.0,
        sigma: float = 2.0,
        dtype=torch.float32,
):
    mask = torch.zeros(1, 1, *repeat(s, dim), dtype=dtype)
    c = (s-1) / 2
    t = (c - margin/100.*c)**2
    for k in product(range(s), repeat=dim):
        r = sum((x - c)**2 for x in k)
        if r > t:
            mask[(..., *k)] = math.exp((t - r) / sigma**2)
        else:
            mask[(..., *k)] = 1.
    return mask


class MaskModule(EquivariantModule):

    def __init__(
            self,
            in_type: FieldType,
            S: int,
            margin: float = 0.,
            sigma: float = 2.,
    ):
        r"""
        
        Performs an element-wise multiplication of the input with a *mask* of shape :math:`S^n`, where :math:`n` is the
        dimensionality of the underlying space.
        
        The mask has value :math:`1` in all pixels with distance smaller than
        :math:`\frac{S - 1}{2} \times (1 - \frac{\mathrm{margin}}{100})` from the center of the mask and :math:`0`
        elsewhere. Values change smoothly between the two regions.
        
        This operation is useful to remove from an input image or feature map all the part of the signal defined on the
        pixels which lay outside the circle inscribed in the grid.
        Because a rotation would move these pixels outside the grid, this information would anyways be
        discarded when rotating an image. However, allowing a model to use this information might break the guaranteed
        equivariance as rotated and non-rotated inputs have different information content.
        
        .. note::
            The input tensors provided to this module must have the following dimensions: :math:`B \times C \times S^n`,
            where :math:`B` is the minibatch dimension, :math:`C` is the channels dimension, and :math:`S^n` are the
            :math:`n` spatial dimensions (corresponding to the Euclidean basespace :math:`\R^n`) associated with the
            given input field type, i.e. ``in_type.gspace.dimensionality``.
            Each Euclidean dimension must be of size :math:`S`.

            For example, if :math:`S=10` and the ``in_type.gspace.dimensionality=2``, then the input tensors
            should be of size :math:`B \times C \times 10 \times 10`.  If ``in_type.gspace.dimensionality=3``
            instead, then the input tensors should be of size
            :math:`B \times C \times 10 \times 10 \times 10`.
        
        Args:
            in_type (FieldType): input field type
            S (int): the shape of the mask and the expected inputs
            margin (float, optional): margin around the mask in percentage with respect to the radius of the mask
            sigma (float, optional): how quickly masked pixels should approach 0.  This can be thought of a standard
                deviation in units of pixels/voxels.  For example, the default value of 2 means that only 5% of the
                original signal will remain 4 px into the masked region.
        
        """
        super(MaskModule, self).__init__()

        self.dim: int = in_type.gspace.dimensionality
        self.S: int = S

        self.margin = margin
        self.mask = torch.nn.Parameter(
                build_mask(S, dim=self.dim, margin=margin, sigma=sigma),
                requires_grad=False,
        )

        self.in_type = self.out_type = in_type
        self.space: GSpace = in_type.gspace

    def forward(self, input: GeometricTensor) -> GeometricTensor:
        assert input.type == self.in_type

        assert input.tensor.shape[2:] == self.mask.shape[2:]

        out = input.tensor * self.mask
        return GeometricTensor(out, self.out_type, input.coords)

    def evaluate_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        return input_shape

    def check_equivariance(self, atol: float = 0.1, rtol: float = 0.1, assertion: bool = True, verbose: bool = True, device: str = 'cpu'):

        # np.set_printoptions(precision=5, threshold=30 *self.in_type.size**2, suppress=False, linewidth=30 *self.in_type.size**2)

        feature_map_size = self.S

        c = self.in_type.size

        from tqdm import tqdm
        from skimage.transform import resize

        import scipy
        x = scipy.datasets.face().transpose((2, 0, 1))[np.newaxis, 0:c, :, :]

        x = resize(
            x,
            (x.shape[0], x.shape[1],) + (feature_map_size,)*self.dim,
            anti_aliasing=True
        )

        assert x.shape[0] == 1, x.shape

        x = x / 255.0 - 0.5

        if x.shape[1] < c:
            to_stack = [x for i in range(c // x.shape[1])]
            if c % x.shape[1] > 0:
                to_stack += [x[:, :(c % x.shape[1]), ...]]

            x = np.concatenate(to_stack, axis=1)

        x = torch.FloatTensor(x)
        x = self.in_type(x)

        with torch.no_grad():
            self.to(device)

            gx = self.in_type(torch.cat([x.transform(el).tensor for el in self.space.testing_elements], dim=0))

            gx = gx.to(device)
            outs_2 = self(gx)
            outs_2 = outs_2.tensor.detach().cpu().numpy()
            assert outs_2.shape[-self.dim:] == (feature_map_size, ) * self.dim, (outs_2.shape, feature_map_size)

            out_1 = self(x.to(device)).to('cpu')
            outs_1 = torch.cat([out_1.transform(el).tensor for el in self.space.testing_elements], dim=0)
            del out_1
            outs_1 = outs_1.detach().cpu().numpy()
            assert outs_1.shape[-self.dim:] == (feature_map_size, ) * self.dim, (outs_1.shape, feature_map_size)

            errors = []

            for i, el in tqdm(enumerate(self.space.testing_elements)):

                # out1 = shrink(out_1.transform(el), last_downsampling).tensor.detach().numpy()

                out1 = outs_1[i:i+1]
                out2 = outs_2[i:i+1]

                b, c = out2.shape[:2]
                spatial_dims = out2.shape[2:]

                center_mask = np.stack(np.meshgrid(*[np.arange(0, _w) - _w // 2 for _w in spatial_dims]), axis=0)
                assert center_mask.shape == (len(spatial_dims), *spatial_dims), (center_mask.shape, *spatial_dims)
                center_mask = (center_mask ** 2).sum(0) < (spatial_dims[0] / 4) ** 2

                out1 = out1[..., center_mask]
                out2 = out2[..., center_mask]

                out1 = out1.reshape(-1)
                out2 = out2.reshape(-1)

                errs = np.abs(out1 - out2)

                esum = np.maximum(np.abs(out1), np.abs(out2))
                esum[esum == 0.0] = 1

                relerr = errs / esum

                if verbose:
                    print(el, relerr.max(), relerr.mean(), relerr.var(), errs.max(), errs.mean(), errs.var())

                tol = rtol * esum + atol

                if np.any(errs > tol) and verbose:
                    print(out1[errs > tol])
                    print(out2[errs > tol])
                    print(tol[errs > tol])

                if assertion:
                    assert np.all(
                        errs < tol), 'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}'.format(
                        el, errs.max(), errs.mean(), errs.var())

                errors.append((el, errs.mean()))

        return errors

