import gc

import torch.nn.functional as F

import escnn.nn
from escnn.nn import FieldType
from escnn.nn import GeometricTensor
from escnn.gspaces import *
from escnn.group import Representation, Group
from escnn.kernels import KernelBasis

from .rd_transposed_convolution import _RdConvTransposed
from .r3convolution import compute_basis_params


from typing import Callable, Union, Tuple, List

import torch
import numpy as np


__all__ = ["R3ConvTransposed"]


class R3ConvTransposed(_RdConvTransposed):
    
    def __init__(self,
                 in_type: FieldType,
                 out_type: FieldType,
                 kernel_size: int,
                 padding: int = 0,
                 output_padding: int = 0,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 bias: bool = True,
                 sigma: Union[List[float], float] = None,
                 frequencies_cutoff: Union[float, Callable[[float], int]] = None,
                 rings: List[float] = None,
                 maximum_offset: int = None,
                 recompute: bool = False,
                 basis_filter: Callable[[dict], bool] = None,
                 initialize: bool = True,
                 ):
        r"""
        
        Transposed G-steerable 3D convolution layer.

        .. warning ::

            This class implements a *discretized* convolution operator over a discrete grid.
            This means that equivariance to continuous symmetries is *not* perfect.
            In practice, by using sufficiently band-limited filters, the equivariance error introduced by the
            discretization of the filters and the features is contained, but some design choices may have a negative
            effect on the overall equivariance of the architecture.

            We provide some :doc:`practical notes <conv_notes>` on using this discretized
            convolution module.

        .. warning ::
            
            Transposed convolution can produce artifacts which can harm the overall equivariance of the model.
            We suggest using :class:`~escnn.nn.R2Upsampling` combined with :class:`~escnn.nn.R3Conv` to perform
            upsampling.
        
        .. seealso ::
            For additional information about the parameters and the methods of this class, see :class:`escnn.nn.R3Conv`.
            The two modules are essentially the same, except for the type of convolution used.
            
        .. warning ::
            Even if the input tensor has a `coords` attribute, the output of this module will not have one.
            
        Args:
            in_type (FieldType): the type of the input field
            out_type (FieldType): the type of the output field
            kernel_size (int): the size of the filter
            padding(int, optional): implicit zero paddings on both sides of the input. Default: ``0``
            output_padding(int, optional): implicit zero paddings on both sides of the input. Default: ``0``
            stride(int, optional): the stride of the convolving kernel. Default: ``1``
            dilation(int, optional): the spacing between kernel elements. Default: ``1``
            groups (int, optional): number of blocked connections from input channels to output channels.
                                    Default: ``1``.
            bias (bool, optional): Whether to add a bias to the output (only to fields which contain a
                    trivial irrep) or not. Default ``True``
            initialize (bool, optional): initialize the weights of the model. Default: ``True``
            
        """
        assert isinstance(in_type.gspace, GSpace3D)
        assert isinstance(out_type.gspace, GSpace3D)

        basis_filter, self._rings, self._sigma, self._maximum_frequency = compute_basis_params(
            kernel_size, frequencies_cutoff, rings, sigma, dilation, basis_filter
        )

        super(R3ConvTransposed, self).__init__(
            in_type,
            out_type,
            3,
            kernel_size,
            padding,
            output_padding,
            stride,
            dilation,
            groups,
            bias,
            basis_filter,
            recompute,
        )

        if initialize:
            # by default, the weights are initialized with a generalized form of He's weight initialization
            escnn.nn.init.generalized_he_init(self.weights.data, self.basisexpansion)

    def _build_kernel_basis(self, in_repr: Representation, out_repr: Representation) -> KernelBasis:
        return self.space.build_kernel_basis(in_repr, out_repr, self._sigma, self._rings,
                                             maximum_frequency=self._maximum_frequency
                                             )

    def forward(self, input: GeometricTensor):
        assert input.type == self.in_type

        if not self.training:
            _filter = self.filter
            _bias = self.expanded_bias
        else:
            # retrieve the filter and the bias
            _filter, _bias = self.expand_parameters()


        # use it for convolution and return the result
        output = F.conv_transpose3d(
                        input.tensor, _filter,
                        padding=self.padding,
                        output_padding=self.output_padding,
                        stride=self.stride,
                        dilation=self.dilation,
                        groups=self.groups,
                        bias=_bias)
        
        return GeometricTensor(output, self.out_type, coords=None)
    
    def check_equivariance_(self, atol: float = 0.1, rtol: float = 0.1, assertion: bool = True, verbose: bool = True, device: str = 'cpu'):
        
        # np.set_printoptions(precision=5, threshold=30 *self.in_type.size**2, suppress=False, linewidth=30 *self.in_type.size**2)

        feature_map_size = 5
        last_downsampling = 5
        first_downsampling = 5

        initial_size = (feature_map_size * last_downsampling + 1 - self.kernel_size) * first_downsampling

        c = self.in_type.size
    
        # x = torch.randn(3, c, 10, 10, 10)

        from tqdm import tqdm
        import matplotlib.image as mpimg
        from skimage.measure import block_reduce
        from skimage.transform import resize


        import scipy
        x = scipy.datasets.face().transpose((2, 0, 1))[np.newaxis, 0:c, :, :]

        x = resize(
            x,
            (x.shape[0], x.shape[1], initial_size, initial_size, initial_size),
            anti_aliasing=True
        )
        
        x = x / 255.0 - 0.5
        
        if x.shape[1] < c:
            to_stack = [x for i in range(c // x.shape[1])]
            if c % x.shape[1] > 0:
                to_stack += [x[:, :(c % x.shape[1]), ...]]
    
            x = np.concatenate(to_stack, axis=1)

        assert x.shape[0] == 1, x.shape

        x = torch.FloatTensor(x)
        x = self.in_type(x)

        # def shrink(t: GeometricTensor, s) -> GeometricTensor:
        #     return GeometricTensor(torch.FloatTensor(block_reduce(t.tensor.detach().numpy(), s, func=np.mean)), t.type)
        shrink1 = escnn.nn.PointwiseAvgPoolAntialiased3D(self.in_type, sigma=first_downsampling / 3., stride=first_downsampling, padding=first_downsampling//2+1)
        shrink2 = escnn.nn.PointwiseAvgPoolAntialiased3D(self.out_type, sigma=last_downsampling / 3., stride=last_downsampling, padding=last_downsampling//2+1)

        errors = []
    
        for el in self.space.testing_elements:
            
            # out1 = self(shrink(x, (1, 1, 5, 5, 5))).transform(el).tensor.detach().numpy()
            # out2 = self(shrink(x.transform(el), (1, 1, 5, 5, 5))).tensor.detach().numpy()
            #
            # out1 = block_reduce(out1, (1, 1, 5, 5, 5), func=np.mean)
            # out2 = block_reduce(out2, (1, 1, 5, 5, 5), func=np.mean)

            out1 = self(shrink1(x)).transform(el)
            out2 = self(shrink1(x.transform(el)))

            out1 = shrink2(out1).tensor.detach().numpy()
            out2 = shrink2(out2).tensor.detach().numpy()

            b, c, w1, w2, w3 = out2.shape

            assert w1 == w2 == w3 == feature_map_size, (w1, w2, w3, feature_map_size)

            # center_mask = np.zeros((3, w1, w2, w3))
            # center_mask[2, ...] = np.arange(0, w3) - w3 // 2
            # center_mask[1, ...] = np.arange(0, w2) - w2 // 2
            # center_mask[0, ...] = np.arange(0, w1) - w1 // 2
            # center_mask[0, ...] = center_mask[0, ...].T
            center_mask = np.stack(np.meshgrid(*[np.arange(0, w) - w//2 for w in [w1, w2, w3]]), axis=0)
            assert center_mask.shape == (3, w1, w2, w3), (center_mask.shape, w1, w2, w3)
            center_mask = (center_mask ** 2).sum(0) < (w1 / 4) ** 2

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
        
            # tol = rtol*(np.abs(out1) + np.abs(out2)) + atol
            tol = rtol * esum + atol
            
            if np.any(errs > tol) and verbose:
                # print(errs[errs > tol])
                print(out1[errs > tol])
                print(out2[errs > tol])
                print(tol[errs > tol])
            
            if assertion:
                assert np.all(errs < tol), 'The error found during equivariance check with element "{}" is too high: max = {}, mean = {} var ={}'.format(el, errs.max(), errs.mean(), errs.var())
        
            errors.append((el, errs.mean()))
    
        return errors
        
    def check_equivariance(self, atol: float = 0.1, rtol: float = 0.1, assertion: bool = True, verbose: bool = True,
                           device: str = 'cpu'):

        # np.set_printoptions(precision=5, threshold=30 *self.in_type.size**2, suppress=False, linewidth=30 *self.in_type.size**2)

        feature_map_size = 5
        last_downsampling = 5
        first_downsampling = 5

        initial_size = (feature_map_size * last_downsampling + 1 - self.kernel_size) * first_downsampling

        c = self.in_type.size

        # x = torch.randn(3, c, 10, 10, 10)

        from tqdm import tqdm
        from skimage.transform import resize

        import scipy
        x = scipy.datasets.face().transpose((2, 0, 1))[np.newaxis, 0:c, :, :]

        x = resize(
            x,
            (x.shape[0], x.shape[1], initial_size, initial_size, initial_size),
            anti_aliasing=True
        )

        x = x / 255.0 - 0.5

        if x.shape[1] < c:
            to_stack = [x for i in range(c // x.shape[1])]
            if c % x.shape[1] > 0:
                to_stack += [x[:, :(c % x.shape[1]), ...]]

            x = np.concatenate(to_stack, axis=1)

        assert x.shape[0] == 1, x.shape

        x = torch.FloatTensor(x)
        x = self.in_type(x)

        shrink1 = escnn.nn.PointwiseAvgPoolAntialiased3D(self.in_type, sigma=first_downsampling / 3.,
                                                         stride=first_downsampling, padding=first_downsampling // 2 + 1)
        shrink2 = escnn.nn.PointwiseAvgPoolAntialiased3D(self.out_type, sigma=last_downsampling / 3.,
                                                         stride=last_downsampling, padding=last_downsampling // 2 + 1)
        shrink1.to(device)
        shrink2.to(device)

        with torch.no_grad():

            gx = self.in_type(torch.cat([x.transform(el).tensor for el in self.space.testing_elements], dim=0))

            gx = gx.to(device)
            gx = shrink1(gx)

            torch.cuda.empty_cache()
            gc.collect()

            self.to(device)
            assert gx.shape[-3:] == (initial_size // first_downsampling,) * 3, (gx.shape, initial_size // first_downsampling)
            outs_2 = self(gx)
            outs_2 = shrink2(outs_2)
            outs_2 = outs_2.tensor.detach().cpu().numpy()
            assert outs_2.shape[-3:] == (feature_map_size, ) * 3, (outs_2.shape, feature_map_size)

            out_1 = shrink1(x.to(device))
            assert out_1.shape[-3:] == (initial_size // first_downsampling,) * 3, (out_1.shape, initial_size // first_downsampling)
            out_1 = self(out_1).to('cpu')
            outs_1 = torch.cat([out_1.transform(el).tensor for el in self.space.testing_elements], dim=0)
            del out_1
            outs_1 = shrink2(self.out_type(outs_1.to(device))).tensor.detach().cpu().numpy()
            assert outs_1.shape[-3:] == (feature_map_size, ) * 3, (outs_1.shape, feature_map_size)

            errors = []

            for i, el in enumerate(tqdm(self.space.testing_elements)):

                out1 = outs_1[i:i+1]
                out2 = outs_2[i:i+1]

                b, c, h, w, d = out2.shape

                center_mask = np.stack(np.meshgrid(*[np.arange(0, _w) - _w // 2 for _w in [h, w, d]]), axis=0)
                assert center_mask.shape == (3, h, w, d), (center_mask.shape, h, w, d)
                center_mask = center_mask[0, :, :] ** 2 + center_mask[1, :, :] ** 2 + center_mask[2, :, :] ** 2 < (h / 4) ** 2

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

    def export(self):
        r"""
        Export this module to a normal PyTorch :class:`torch.nn.ConvTranspose2d` module and set to "eval" mode.

        """
    
        # set to eval mode so the filter and the bias are updated with the current
        # values of the weights
        self.eval()
        _filter = self.filter
        _bias = self.expanded_bias
    
        # build the PyTorch Conv2d module
        has_bias = self.bias is not None
        conv = torch.nn.ConvTranspose3d(self.in_type.size,
                                       self.out_type.size,
                                       self.kernel_size,
                                       padding=self.padding,
                                       stride=self.stride,
                                       dilation=self.dilation,
                                       groups=self.groups,
                                       bias=has_bias)
        
        # set the filter and the bias
        conv.weight.data[:] = _filter.data
        if has_bias:
            conv.bias.data[:] = _bias.data
    
        return conv

