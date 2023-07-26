.. automodule::  escnn.nn
   :no-members:

escnn.nn
========

This subpackage provides implementations of equivariant neural network modules.

In an equivariant network, features are associated with a transformation law under actions of a symmetry group.
The transformation law of a feature field is implemented by its :class:`~escnn.nn.FieldType` which can be interpreted as a data type.
A :class:`~escnn.nn.GeometricTensor` is wrapping a :class:`torch.Tensor` to endow it with a :class:`~escnn.nn.FieldType`.
Geometric tensors are processed by :class:`~escnn.nn.EquivariantModule` s which are :class:`torch.nn.Module` s that guarantee the
specified behavior of their output fields given a transformation of their input fields.


This subpackage depends on :doc:`escnn.group` and :doc:`escnn.gspaces`.


To enable efficient deployment of equivariant networks, many :class:`~escnn.nn.EquivariantModule` s implement a
:meth:`~escnn.nn.EquivariantModule.export` method which converts a *trained* equivariant module into a pure PyTorch
module, with few or no dependencies with **escnn**.
Not all modules support this feature yet, so read each module's documentation to check whether it implements this method
or not.
We provide a simple example::

    # build a simple equivariant model using a SequentialModule

    s = escnn.gspaces.rot2dOnR2(8)
    c_in = escnn.nn.FieldType(s, [s.trivial_repr]*3)
    c_hid = escnn.nn.FieldType(s, [s.regular_repr]*3)
    c_out = escnn.nn.FieldType(s, [s.regular_repr]*1)

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

    # train the model
    # ...

    # export the model

    net.eval()
    net_exported = net.export()

    print(net)
    > SequentialModule(
    >   (0): R2Conv([8-Rotations: {irrep_0, irrep_0, irrep_0}], [8-Rotations: {regular, regular, regular}], kernel_size=5, stride=1, bias=False)
    >   (1): InnerBatchNorm([8-Rotations: {regular, regular, regular}], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    >   (2): ReLU(inplace=True, type=[8-Rotations: {regular, regular, regular}])
    >   (3): PointwiseMaxPool()
    >   (4): R2Conv([8-Rotations: {regular, regular, regular}], [8-Rotations: {regular}], kernel_size=3, stride=1, bias=False)
    >   (5): InnerBatchNorm([8-Rotations: {regular}], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    >   (6): ELU(alpha=1.0, inplace=True, type=[8-Rotations: {regular}])
    >   (7): GroupPooling([8-Rotations: {regular}])
    > )

    print(net_exported)
    > Sequential(
    >   (0): Conv2d(3, 24, kernel_size=(5, 5), stride=(1, 1), bias=False)
    >   (1): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    >   (2): ReLU(inplace=True)
    >   (3): MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(1, 1), ceil_mode=False)
    >   (4): Conv2d(24, 8, kernel_size=(3, 3), stride=(1, 1), bias=False)
    >   (5): BatchNorm2d(8, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    >   (6): ELU(alpha=1.0, inplace=True)
    >   (7): MaxPoolChannels(kernel_size=8)
    > )


    # check that the two models are equivalent

    x = torch.randn(10, c_in.size, 31, 31)
    x = GeometricTensor(x, c_in)

    y1 = net(x).tensor
    y2 = net_exported(x.tensor)

    assert torch.allclose(y1, y2)

|


.. contents:: Contents
    :local:
    :backlinks: top



Field Type
----------

.. autoclass:: escnn.nn.FieldType
    :members:
    :undoc-members:

Geometric Tensor
----------------

.. autoclass:: escnn.nn.GeometricTensor
    :members:
    :undoc-members:
    :exclude-members: __add__,__sub__,__iadd__,__isub__,__mul__,__repr__,__rmul__,__imul__


Equivariant Module
------------------

.. autoclass:: escnn.nn.EquivariantModule
    :members:


Utils
-----

direct sum
~~~~~~~~~~

.. autofunction:: escnn.nn.tensor_directsum

Linear Layers
-------------

Linear
~~~~~~
.. autoclass:: escnn.nn.Linear
    :members:
    :show-inheritance:


.. _steerable-dense-conv:

Steerable Dense Convolution
---------------------------

The following modules implement **discretized convolution** operators over discrete grids.
This means that **equivariance** to continuous symmetries is **not perfect**.
In practice, by using sufficiently band-limited filters, the equivariance error introduced by the
discretization of the filters and the features is contained, but some design choices may have a negative
effect on the overall equivariance of the architecture.

We also provide some :doc:`practical notes <conv_notes>` on using these discretized convolution modules.


.. contents::
    :local:
    :backlinks: top

RdConv
~~~~~~
.. autoclass:: escnn.nn.modules.conv._RdConv
    :members:
    :show-inheritance:

R2Conv
~~~~~~
.. autoclass:: escnn.nn.R2Conv
    :members:
    :show-inheritance:

R3Conv
~~~~~~
.. autoclass:: escnn.nn.R3Conv
    :members:
    :show-inheritance:

R3IcoConv
~~~~~~~~~
.. autoclass:: escnn.nn.R3IcoConv
    :members:
    :show-inheritance:

R2ConvTransposed
~~~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.R2ConvTransposed
    :members:
    :show-inheritance:

R3ConvTransposed
~~~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.R3ConvTransposed
    :members:
    :show-inheritance:

R3IcoConvTransposed
~~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.R3IcoConvTransposed
    :members:
    :show-inheritance:

Steerable Point Convolution
---------------------------

.. contents::
    :local:
    :backlinks: top

RdPointConv
~~~~~~~~~~~
.. autoclass:: escnn.nn.modules.pointconv._RdPointConv
    :members:
    :show-inheritance:

R2PointConv
~~~~~~~~~~~
.. autoclass:: escnn.nn.R2PointConv
    :members:
    :show-inheritance:

R3PointConv
~~~~~~~~~~~
.. autoclass:: escnn.nn.R3PointConv
    :members:
    :show-inheritance:


BasisManager
------------

.. autoclass:: escnn.nn.modules.basismanager.BasisManager
    :members:
    :show-inheritance:

.. autoclass:: escnn.nn.modules.basismanager.BlocksBasisExpansion
    :members:
    :show-inheritance:

.. autoclass:: escnn.nn.modules.basismanager.BlocksBasisSampler
    :members:
    :show-inheritance:


Non Linearities
---------------

.. contents::
    :local:
    :backlinks: top

PointwiseNonLinearity
~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.PointwiseNonLinearity
   :members:
   :show-inheritance:

ReLU
~~~~
.. autoclass:: escnn.nn.ReLU
   :members:
   :show-inheritance:


ELU
~~~
.. autoclass:: escnn.nn.ELU
    :members:
    :show-inheritance:

LeakyReLU
~~~~~~~~~
.. autoclass:: escnn.nn.LeakyReLU
    :members:
    :show-inheritance:

FourierPointwise
~~~~~~~~~~~~~~~~

.. autoclass:: escnn.nn.FourierPointwise
    :members:
    :show-inheritance:

FourierELU
~~~~~~~~~~

.. autoclass:: escnn.nn.FourierELU
    :members:
    :show-inheritance:

QuotientFourierPointwise
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: escnn.nn.QuotientFourierPointwise
    :members:
    :show-inheritance:

QuotientFourierELU
~~~~~~~~~~~~~~~~~~

.. autoclass:: escnn.nn.QuotientFourierELU
    :members:
    :show-inheritance:

TensorProductModule
~~~~~~~~~~~~~~~~~~~

.. autoclass:: escnn.nn.TensorProductModule
    :members:
    :show-inheritance:


GatedNonLinearity1
~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.GatedNonLinearity1
   :members:
   :show-inheritance:

GatedNonLinearity2
~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.GatedNonLinearity2
   :members:
   :show-inheritance:

GatedNonLinearityUniform
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.GatedNonLinearityUniform
   :members:
   :show-inheritance:

InducedGatedNonLinearity1
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.InducedGatedNonLinearity1
   :members:
   :show-inheritance:

ConcatenatedNonLinearity
~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.ConcatenatedNonLinearity
    :members:
    :show-inheritance:

NormNonLinearity
~~~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.NormNonLinearity
   :members:
   :show-inheritance:

InducedNormNonLinearity
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.InducedNormNonLinearity
   :members:
   :show-inheritance:


VectorFieldNonLinearity
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.VectorFieldNonLinearity
   :members:
   :show-inheritance:


Invariant Maps
--------------

.. contents::
    :local:
    :backlinks: top


GroupPooling
~~~~~~~~~~~~
.. autoclass:: escnn.nn.GroupPooling
   :members:
   :show-inheritance:

NormPool
~~~~~~~~
.. autoclass:: escnn.nn.NormPool
   :members:
   :show-inheritance:


InducedNormPool
~~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.InducedNormPool
   :members:
   :show-inheritance:


Pooling
-------

.. contents::
    :local:
    :backlinks: top

NormMaxPool
~~~~~~~~~~~
.. autoclass:: escnn.nn.NormMaxPool
   :members:
   :show-inheritance:

PointwiseMaxPool2D
~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.PointwiseMaxPool2D
   :members:
   :show-inheritance:

PointwiseMaxPoolAntialiased2D
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.PointwiseMaxPoolAntialiased2D
   :members:
   :show-inheritance:

PointwiseMaxPool3D
~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.PointwiseMaxPool3D
   :members:
   :show-inheritance:

PointwiseMaxPoolAntialiased3D
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.PointwiseMaxPoolAntialiased3D
   :members:
   :show-inheritance:

PointwiseAvgPool2D
~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.PointwiseAvgPool2D
   :members:
   :show-inheritance:

PointwiseAvgPoolAntialiased2D
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.PointwiseAvgPoolAntialiased2D
   :members:
   :show-inheritance:

PointwiseAvgPool3D
~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.PointwiseAvgPool3D
   :members:
   :show-inheritance:

PointwiseAvgPoolAntialiased3D
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.PointwiseAvgPoolAntialiased3D
   :members:
   :show-inheritance:

PointwiseAdaptiveAvgPool2D
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.PointwiseAdaptiveAvgPool2D
   :members:
   :show-inheritance:

PointwiseAdaptiveAvgPool3D
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.PointwiseAdaptiveAvgPool3D
   :members:
   :show-inheritance:

PointwiseAdaptiveMaxPool2D
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.PointwiseAdaptiveMaxPool2D
   :members:
   :show-inheritance:

PointwiseAdaptiveMaxPool3D
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.PointwiseAdaptiveMaxPool3D
   :members:
   :show-inheritance:


Normalization
-------------------

.. contents::
    :local:
    :backlinks: top


IIDBatchNorm1d
~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.IIDBatchNorm1d
   :members:
   :show-inheritance:


IIDBatchNorm2d
~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.IIDBatchNorm2d
   :members:
   :show-inheritance:

IIDBatchNorm3d
~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.IIDBatchNorm3d
   :members:
   :show-inheritance:


FieldNorm
~~~~~~~~~
.. autoclass:: escnn.nn.FieldNorm
   :members:
   :show-inheritance:


InnerBatchNorm
~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.InnerBatchNorm
   :members:
   :show-inheritance:

NormBatchNorm
~~~~~~~~~~~~~
.. autoclass:: escnn.nn.NormBatchNorm
   :members:
   :show-inheritance:

InducedNormBatchNorm
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.InducedNormBatchNorm
   :members:
   :show-inheritance:

GNormBatchNorm
~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.GNormBatchNorm
   :members:
   :show-inheritance:


Dropout
-------

.. contents::
    :local:
    :backlinks: top

FieldDropout
~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.FieldDropout
   :members:
   :show-inheritance:

PointwiseDropout
~~~~~~~~~~~~~~~~
.. autoclass:: escnn.nn.PointwiseDropout
   :members:
   :show-inheritance:


Other Modules
-------------

.. contents::
    :local:
    :backlinks: top

Sequential
~~~~~~~~~~

.. autoclass:: escnn.nn.SequentialModule
    :members:
    :show-inheritance:

Restriction
~~~~~~~~~~~

.. autoclass:: escnn.nn.RestrictionModule
    :members:
    :show-inheritance:

Disentangle
~~~~~~~~~~~

.. autoclass:: escnn.nn.DisentangleModule
    :members:
    :show-inheritance:

Upsampling
~~~~~~~~~~

.. autoclass:: escnn.nn.R2Upsampling
    :members:
    :show-inheritance:

.. autoclass:: escnn.nn.R3Upsampling
    :members:
    :show-inheritance:

Multiple
~~~~~~~~

.. autoclass:: escnn.nn.MultipleModule
    :members:
    :show-inheritance:
    
Reshuffle
~~~~~~~~~

.. autoclass:: escnn.nn.ReshuffleModule
    :members:
    :show-inheritance:

Mask
~~~~

.. autoclass:: escnn.nn.MaskModule
    :members:
    :show-inheritance:

Identity
~~~~~~~~

.. autoclass:: escnn.nn.IdentityModule
    :members:
    :show-inheritance:


HarmonicPolynomialR3
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: escnn.nn.HarmonicPolynomialR3
    :members:
    :show-inheritance:


Weight Initialization
---------------------

.. automodule:: escnn.nn.init
   :members:


.. toctree::
   :glob:
   :maxdepth: 1
   :hidden:

   escnn.nn.others




