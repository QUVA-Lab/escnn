

from escnn.gspaces import *
from escnn.nn import FieldType
from escnn.nn import GeometricTensor

from abc import ABC

from .equivariant_module import EquivariantModule

from typing import Tuple, Optional, Union

import torch
import numpy as np

import math

import torch.nn.functional as F

__all__ = ["R2Upsampling", "R3Upsampling"]


class _RdUpsampling(EquivariantModule, ABC):

    def __init__(self,
                 in_type: FieldType,
                 d: int,
                 scale_factor: Optional[int] = None,
                 size: Optional[Union[int, Tuple[int, int]]] = None,
                 mode: str = None,
                 align_corners: bool = False
                 ):
        r"""

        Wrapper for :func:`torch.nn.functional.interpolate`. Check its documentation for further details.

        ``mode="nearest"`` is not equivariant; using this method may result in broken equivariance.
        For this reason, we suggest to use ``"bi/trilinear"``.

        .. warning ::
            The module supports a ``size`` parameter as an alternative to ``scale_factor``.
            However, the use of ``scale_factor`` should be *preferred*, since it guarantees both axes are scaled
            uniformly, which preserves rotation equivariance.
            A misuse of the parameter ``size`` can break the overall equivariance, since it might scale the two axes by
            two different factors.

        .. warning ::
            Even if the input tensor has a `coords` attribute, the output of this module will not have one.


        Args:
            in_type (FieldType): the input field type
            d (int): dimensionality of the base space (2 for images, 3 for volumes)
            size (optional, int or tuple): output spatial size.
            scale_factor (optional, int): multiplier for spatial size
            mode (str): algorithm used for upsampling: ``nearest`` / ``bilinear`` or ``trilinear`` depending on ``d``
            align_corners (bool): if ``True``, the corner pixels of the input and output tensors are aligned, and thus
                    preserving the values at those pixels. This only has effect when mode is ``bilinear``.
                    Default: ``False``

        """

        assert d in [2, 3], f"Only dimensionality 2 or 3 are currently suported by 'd={d}' found"

        assert isinstance(in_type.gspace, GSpace)
        assert in_type.gspace.dimensionality == d, (in_type.gspace.dimensionality, d)

        super(_RdUpsampling, self).__init__()

        self.d = d
        self.space = in_type.gspace
        self.in_type = in_type
        self.out_type = in_type

        assert size is None or scale_factor is None, \
            f'Only one of "size" and "scale_factor" can be set, but found scale_factor={scale_factor} and size={size}'

        self._size = (size,) * self.d if isinstance(size, int) else size
        assert self._size is None or (isinstance(self._size, tuple) and len(self._size) == self.d), self._size
        self._scale_factor = scale_factor
        self._mode = mode
        self._align_corners = align_corners if mode != "nearest" else None

        valid_modes = ["nearest"]
        if self.d == 2:
            valid_modes += ["bilinear"]
        elif self.d == 3:
            valid_modes += ["trilinear"]

        if mode not in valid_modes:
            raise ValueError(f'Error Upsampling mode {mode} not recognized! Mode should be one of {valid_modes}.')

    def forward(self, input: GeometricTensor):
        r"""

        Args:
            input (torch.Tensor): input feature map

        Returns:
             the result of the convolution

        """

        assert input.type == self.in_type
        assert len(input.shape) == 2 + self.d, (input.shape, self.d)

        if self._align_corners is None:
            output = F.interpolate(input.tensor,
                                 size=self._size,
                                 scale_factor=self._scale_factor,
                                 mode=self._mode)
        else:
            output = F.interpolate(input.tensor,
                                 size=self._size,
                                 scale_factor=self._scale_factor,
                                 mode=self._mode,
                                 align_corners=self._align_corners)

        return GeometricTensor(output, self.out_type, coords=None)

    def evaluate_output_shape(self, input_shape: Tuple) -> Tuple:
        assert len(input_shape) == 2 + self.d, (input_shape, self.d)
        assert input_shape[1] == self.in_type.size

        b, c = input_shape[:2]
        shape_i = input_shape[2:]

        if self._size is None:
            shape_o = tuple(
                math.floor(shape_i[i] * self._scale_factor) for i in range(self.d)
            )
        else:
            shape_o = self._size

        return (b, self.out_type.size, *shape_o)

    def export(self):
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.Upsample` module and set to "eval" mode.

        """

        self.eval()

        if self._align_corners is not None:
            upsample = torch.nn.Upsample(
                size=self._size,
                scale_factor=self._scale_factor,
                mode=self._mode,
                align_corners=self._align_corners
            )
        else:
            upsample = torch.nn.Upsample(
                size=self._size,
                scale_factor=self._scale_factor,
                mode=self._mode,
            )

        return upsample.eval()


class R2Upsampling(_RdUpsampling):
    
    def __init__(self,
                 in_type: FieldType,
                 scale_factor: Optional[int] = None,
                 size: Optional[Union[int, Tuple[int, int]]] = None,
                 mode: str = "bilinear",
                 align_corners: bool = False
                 ):
        r"""
        
        Wrapper for :func:`torch.nn.functional.interpolate`. Check its documentation for further details.
        
        Only ``"bilinear"`` and ``"nearest"`` methods are supported.
        However, ``"nearest"`` is not equivariant; using this method may result in broken equivariance.
        For this reason, we suggest to use ``"bilinear"`` (default value).

        .. warning ::
            The module supports a ``size`` parameter as an alternative to ``scale_factor``.
            However, the use of ``scale_factor`` should be *preferred*, since it guarantees both axes are scaled
            uniformly, which preserves rotation equivariance.
            A misuse of the parameter ``size`` can break the overall equivariance, since it might scale the two axes by
            two different factors.
        
        .. warning ::
            Even if the input tensor has a `coords` attribute, the output of this module will not have one.
        
        
        Args:
            in_type (FieldType): the input field type
            size (optional, int or tuple): output spatial size.
            scale_factor (optional, int): multiplier for spatial size
            mode (str): algorithm used for upsampling: ``nearest`` | ``bilinear``. Default: ``bilinear``
            align_corners (bool): if ``True``, the corner pixels of the input and output tensors are aligned, and thus
                    preserving the values at those pixels. This only has effect when mode is ``bilinear``.
                    Default: ``False``
            
        """

        assert isinstance(in_type.gspace, GSpace)
        assert in_type.gspace.dimensionality == 2

        if mode not in ["nearest", "bilinear"]:
            raise ValueError(f'Error Upsampling mode {mode} not recognized! Mode should be `nearest` or `bilinear`.')

        super(R2Upsampling, self).__init__(in_type, 2, scale_factor, size, mode, align_corners)

    def check_equivariance(self, atol: float = 0.1, rtol: float = 0.1):
        
        initial_size = 55

        c = self.in_type.size

        # x = torch.randn(3, c, initial_size, initial_size)

        import matplotlib.image as mpimg
        from skimage.transform import resize

        # x = mpimg.imread('../group/testimage.jpeg').transpose((2, 0, 1))[np.newaxis, 0:c, :, :]
        import scipy
        x = scipy.datasets.face().transpose((2, 0, 1))[np.newaxis, 0:c, :, :]

        x = x / 255.0
        x = resize(
            x,
            (x.shape[0], x.shape[1], initial_size, initial_size),
            anti_aliasing=True
        )

        if x.shape[1] < c:
            to_stack = [x for i in range(c // x.shape[1])]
            if c % x.shape[1] > 0:
                to_stack += [x[:, :(c % x.shape[1]), ...]]

            x = np.concatenate(to_stack, axis=1)

        x = GeometricTensor(torch.FloatTensor(x), self.in_type)

        errors = []

        for el in self.space.testing_elements:

            out1 = self(x).transform(el).tensor.detach().numpy()
            out2 = self(x.transform(el)).tensor.detach().numpy()

            b, c, h, w = out2.shape

            center_mask = np.zeros((2, h, w))
            center_mask[1, :, :] = np.arange(0, w) - w / 2
            center_mask[0, :, :] = np.arange(0, h) - h / 2
            center_mask[0, :, :] = center_mask[0, :, :].T
            center_mask = center_mask[0, :, :] ** 2 + center_mask[1, :, :] ** 2 < (h * 0.4) ** 2

            out1 = out1[..., center_mask]
            out2 = out2[..., center_mask]

            out1 = out1.reshape(-1)
            out2 = out2.reshape(-1)

            errs = np.abs(out1 - out2)

            esum = np.maximum(np.abs(out1), np.abs(out2))
            esum[esum == 0.0] = 1

            relerr = errs / esum

            # print(el, relerr.max(), relerr.mean(), relerr.var(), errs.max(), errs.mean(), errs.var())

            # tol = rtol*(np.abs(out1) + np.abs(out2)) + atol
            tol = rtol * esum + atol

            if np.any(errs > tol):
                print(el, relerr.max(), relerr.mean(), relerr.var(), errs.max(), errs.mean(), errs.var())
                # print(errs[errs > tol])
                print(out1[errs > tol])
                print(out2[errs > tol])
                print(tol[errs > tol])

            # assert np.all(np.abs(out1 - out2) < tol), 'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}'.format(el, errs.max(), errs.mean(), errs.var())
            assert np.all(errs < tol), 'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}'.format(el, errs.max(), errs.mean(), errs.var())

            errors.append((el, errs.mean()))

        return errors


class R3Upsampling(_RdUpsampling):

    def __init__(self,
                 in_type: FieldType,
                 scale_factor: Optional[int] = None,
                 size: Optional[Union[int, Tuple[int, int]]] = None,
                 mode: str = "trilinear",
                 align_corners: bool = False
                 ):
        r"""

        Wrapper for :func:`torch.nn.functional.interpolate`. Check its documentation for further details.

        Only ``"trilinear"`` and ``"nearest"`` methods are supported.
        However, ``"nearest"`` is not equivariant; using this method may result in broken equivariance.
        For this reason, we suggest to use ``"trilinear"`` (default value).

        .. warning ::
            The module supports a ``size`` parameter as an alternative to ``scale_factor``.
            However, the use of ``scale_factor`` should be *preferred*, since it guarantees both axes are scaled
            uniformly, which preserves rotation equivariance.
            A misuse of the parameter ``size`` can break the overall equivariance, since it might scale the two axes by
            two different factors.

        .. warning ::
            Even if the input tensor has a `coords` attribute, the output of this module will not have one.


        Args:
            in_type (FieldType): the input field type
            size (optional, int or tuple): output spatial size.
            scale_factor (optional, int): multiplier for spatial size
            mode (str): algorithm used for upsampling: ``nearest`` | ``trilinear``. Default: ``trilinear``
            align_corners (bool): if ``True``, the corner pixels of the input and output tensors are aligned, and thus
                    preserving the values at those pixels. This only has effect when mode is ``trilinear``.
                    Default: ``False``

        """

        assert isinstance(in_type.gspace, GSpace)
        assert in_type.gspace.dimensionality == 3

        if mode not in ["nearest", "trilinear"]:
            raise ValueError(f'Error Upsampling mode {mode} not recognized! Mode should be `nearest` or `trilinear`.')

        super(R3Upsampling, self).__init__(in_type, 3, scale_factor, size, mode, align_corners)

    def check_equivariance(self, atol: float = 0.1, rtol: float = 0.1):

        initial_size = 17

        c = self.in_type.size

        # x = torch.randn(3, c, initial_size, initial_size)

        import matplotlib.image as mpimg
        from skimage.transform import resize

        # x = mpimg.imread('../group/testimage.jpeg').transpose((2, 0, 1))[np.newaxis, 0:c, :, :]
        import scipy
        x = scipy.datasets.face().transpose((2, 0, 1))[np.newaxis, 0:c, :, :]

        x = x / 255.0
        x = resize(
            x,
            (x.shape[0], x.shape[1], initial_size, initial_size, initial_size),
            anti_aliasing=True
        )

        if x.shape[1] < c:
            to_stack = [x for i in range(c // x.shape[1])]
            if c % x.shape[1] > 0:
                to_stack += [x[:, :(c % x.shape[1]), ...]]

            x = np.concatenate(to_stack, axis=1)

        x = GeometricTensor(torch.FloatTensor(x), self.in_type)

        errors = []

        for el in self.space.testing_elements:

            out1 = self(x).transform(el).tensor.detach().numpy()
            out2 = self(x.transform(el)).tensor.detach().numpy()

            b, c, h, w, d = out2.shape

            center_mask = np.zeros((3, h, w, d))
            center_mask[2, ...] = np.arange(0, d) - d / 2
            center_mask[1, ...] = np.arange(0, w) - w / 2
            center_mask[1, ...] = np.swapaxes(center_mask, 3, 2)[1, ...]
            center_mask[0, ...] = np.arange(0, h) - h / 2
            center_mask[0, ...] = np.swapaxes(center_mask, 3, 1)[0, ...]
            center_mask = center_mask[0, :, :] ** 2 + center_mask[1, :, :] ** 2 + center_mask[2, :, :] ** 2 < (h / 4) ** 2
            # center_mask = center_mask[0, ...] ** 2 + center_mask[1, ...] ** 2 < (h * 0.4) ** 2

            out1 = out1[..., center_mask]
            out2 = out2[..., center_mask]

            out1 = out1.reshape(-1)
            out2 = out2.reshape(-1)

            errs = np.abs(out1 - out2)

            esum = np.maximum(np.abs(out1), np.abs(out2))
            esum[esum == 0.0] = 1

            relerr = errs / esum

            # print(el, relerr.max(), relerr.mean(), relerr.var(), errs.max(), errs.mean(), errs.var())

            # tol = rtol*(np.abs(out1) + np.abs(out2)) + atol
            tol = rtol * esum + atol

            if np.any(errs > tol):
                print(el, relerr.max(), relerr.mean(), relerr.var(), errs.max(), errs.mean(), errs.var())
                # print(errs[errs > tol])
                print(out1[errs > tol])
                print(out2[errs > tol])
                print(tol[errs > tol])

            # assert np.all(np.abs(out1 - out2) < tol), 'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}'.format(el, errs.max(), errs.mean(), errs.var())
            assert np.all(
                errs < tol), 'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}'.format(
                el, errs.max(), errs.mean(), errs.var())

            errors.append((el, errs.mean()))

        return errors
