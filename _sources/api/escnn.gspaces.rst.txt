
.. automodule:: escnn.gspaces
   :no-members:

escnn.gspaces
====================================

This subpackage implements G-spaces as a user interface for defining spaces and their symmetries.
The user typically instantiates a subclass of :class:`~escnn.gspaces.GSpace` to specify the symmetries considered and uses
it to instantiate equivariant neural network modules (see :doc:`escnn.nn`).


This subpackage depends on :doc:`escnn.group` and :doc:`escnn.kernels`.



A point :math:`\bold{v} \in \R^n` is parametrized using an :math:`(X, Y, Z, \dots)` convention,
i.e. :math:`\bold{v} = (x, y, z, \dots)^T`.
The representation :attr:`escnn.gspaces.GSpace.basespace_action` also assumes this convention.

However, when working with voxel data, the :math:`(\cdots, -Z, -Y, X)` convention is used.
That means that, in a feature tensor of shape :math:`(B, C, D_1, D_2, \cdots, D_{n-2}, D_{n-1}, D_n)``,
the last dimension is the X axis but the :math:`i`-th last dimension is the *inverted* :math:`i`-th axis.
Note that this is consistent with 2D images, where a :math:`(-Y, X)` convention is used.

This is especially relevant when transforming a :class:`~escnn.nn.GeometricTensor` or when building
convolutional filters in :class:`~escnn.nn.R3Conv` which should be equivariant to subgroups of :math:`\O3`
(e.g. when choosing the rotation axis for :func:`~escnn.gspaces.rot2dOnR3`).


.. contents:: Contents
    :local:
    :backlinks: top


Factory Methods
---------------


(trivial) action on single point
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: escnn.gspaces.no_base_space

action on plane
~~~~~~~~~~~~~~~

.. contents::
    :local:
    :backlinks: top

.. autofunction:: escnn.gspaces.rot2dOnR2

.. autofunction:: escnn.gspaces.flipRot2dOnR2

.. autofunction:: escnn.gspaces.flip2dOnR2

.. autofunction:: escnn.gspaces.trivialOnR2

action on volume
~~~~~~~~~~~~~~~~

.. contents::
    :local:
    :backlinks: top

.. autofunction:: escnn.gspaces.flipRot3dOnR3

.. autofunction:: escnn.gspaces.rot3dOnR3

.. autofunction:: escnn.gspaces.fullIcoOnR3

.. autofunction:: escnn.gspaces.icoOnR3

.. autofunction:: escnn.gspaces.fullOctaOnR3

.. autofunction:: escnn.gspaces.octaOnR3

.. autofunction:: escnn.gspaces.dihedralOnR3

.. autofunction:: escnn.gspaces.rot2dOnR3

.. autofunction:: escnn.gspaces.conicalOnR3

.. autofunction:: escnn.gspaces.fullCylindricalOnR3

.. autofunction:: escnn.gspaces.cylindricalOnR3

.. autofunction:: escnn.gspaces.mirOnR3

.. autofunction:: escnn.gspaces.invOnR3

.. autofunction:: escnn.gspaces.trivialOnR3


Abstract Group Space
--------------------

.. autoclass:: escnn.gspaces.GSpace
    :members:
    :undoc-members:

Group Action (trivial) on single point
--------------------------------------

.. autoclass:: escnn.gspaces.GSpace0D
    :members:
    :undoc-members:
    :show-inheritance:

Group Actions on the Plane
--------------------------

.. autoclass:: escnn.gspaces.GSpace2D
    :members:
    :undoc-members:
    :show-inheritance:

Group Actions on the 3D Space
-----------------------------

.. autoclass:: escnn.gspaces.GSpace3D
    :members:
    :undoc-members:
    :show-inheritance:


