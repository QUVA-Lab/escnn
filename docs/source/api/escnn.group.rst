
.. automodule:: escnn.group
   :no-members:

escnn.group
===========

This subpackage implements groups and group representations.

To avoid creating multiple redundant instances of the same group, we suggest using the factory functions in :ref:`Factory Functions <factory-functions>`.
These functions only build a single instance of each different group and return that instance on consecutive calls.

When building equivariant networks, it is not necessary to directly instantiate objects from this subpackage.
Instead, we suggest using the interface provided in :doc:`escnn.gspaces`.

This subpackage is not dependent on the others and can be used alone to generate groups and compute their representations.

The main classes are :class:`~escnn.group.Group`, :class:`~escnn.group.GroupElement` and :class:`~escnn.group.Representation`.
We provide a simple tools and interfaces for combining and generating new instances of these classes.
For examples, group elements can be combined using the group operations (see :class:`~escnn.group.GroupElement`),
representations can be restricted to a subgroup (:meth:`~escnn.group.Representation.restrict`), (often) induced to a
larger group (:meth:`~escnn.group.Group.induced_representation`), or combined via the direct-sum
(:func:`~escnn.group.direct_sum` or, simply, using the binary operator `+`) or the tensor-product (:meth:`~escnn.group.Representation.tensor`),
while groups can be combined via the direct product (:func:`~escnn.group.direct_product`, :func:`~escnn.group.double_group` )
or generated via :meth:`~escnn.group.Group.subgroup`.

.. contents:: Contents
    :local:
    :backlinks: top


.. _factory-functions:

Factory Functions
-----------------

.. contents::
    :local:
    :backlinks: top

.. _so2-group-factory:

SO(2)
~~~~~
.. autofunction:: escnn.group.so2_group

.. _o2-group-factory:

O(2)
~~~~
.. autofunction:: escnn.group.o2_group

.. _cyclic-group-factory:

Cyclic Group
~~~~~~~~~~~~
.. autofunction:: escnn.group.cyclic_group

.. _dihedral-group-factory:

Dihedral Group
~~~~~~~~~~~~~~
.. autofunction:: escnn.group.dihedral_group

.. _trivial-group-factory:

Trivial Group
~~~~~~~~~~~~~
.. autofunction:: escnn.group.trivial_group

.. _so3-group-factory:

SO(3)
~~~~~
.. autofunction:: escnn.group.so3_group

.. _o3-group-factory:

O(3)
~~~~
.. autofunction:: escnn.group.o3_group

.. _klein4-group-factory:

Klein 4 Group
~~~~~~~~~~~~~
.. autofunction:: escnn.group.klein4_group

.. _octa-group-factory:

Octahedral Group
~~~~~~~~~~~~~~~~~
.. autofunction:: escnn.group.octa_group

.. _full-octa-group-factory:

Full Octahedral Group
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: escnn.group.full_octa_group


.. _ico-group-factory:

Icosahedral Group
~~~~~~~~~~~~~~~~~
.. autofunction:: escnn.group.ico_group


.. _full-ico-group-factory:

Full Icosahedral Group
~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: escnn.group.full_ico_group

.. _cylinder-group-factory:

Cylinder Group
~~~~~~~~~~~~~~
.. autofunction:: escnn.group.cylinder_group

.. _full-cylinder-group-factory:

Full Cylinder Group
~~~~~~~~~~~~~~~~~~~
.. autofunction:: escnn.group.full_cylinder_group

.. _cylinder-discrete-group-factory:

Discrete Cylinder Group
~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: escnn.group.cylinder_discrete_group

.. _full-cylinder-discrete-group-factory:

Discrete Full Cylinder Group
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: escnn.group.full_cylinder_discrete_group

.. _direct-product-factory:

Direct Product
~~~~~~~~~~~~~~
.. autofunction:: escnn.group.direct_product

.. _double-group-factory:

Double Group
~~~~~~~~~~~~
.. autofunction:: escnn.group.double_group


Group
-----
.. autoclass:: escnn.group.Group
    :members:
    :undoc-members:


GroupElement
------------
.. autoclass:: escnn.group.GroupElement
    :members:
    :undoc-members:


Groups
------

.. contents::
    :local:
    :backlinks: top

SO(2)
~~~~~
.. autoclass:: escnn.group.SO2
    :members:
    :show-inheritance:

O(2)
~~~~
.. autoclass:: escnn.group.O2
    :members:
    :show-inheritance:


Cyclic Group
~~~~~~~~~~~~
.. autoclass:: escnn.group.CyclicGroup
    :members:
    :show-inheritance:

Dihedral Group
~~~~~~~~~~~~~~
.. autoclass:: escnn.group.DihedralGroup
    :members:
    :show-inheritance:

SO(3)
~~~~~
.. autoclass:: escnn.group.SO3
    :members:
    :show-inheritance:


O(3)
~~~~
.. autoclass:: escnn.group.O3
    :members:
    :show-inheritance:

Icosahedral Group
~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.group.Icosahedral
    :members:
    :show-inheritance:

Octahedral Group
~~~~~~~~~~~~~~~~
.. autoclass:: escnn.group.Octahedral
    :members:
    :show-inheritance:

Direct Product Group
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.group.DirectProductGroup
    :members:
    :show-inheritance:

Double Group
~~~~~~~~~~~~~~~~~~~~
.. autoclass:: escnn.group.DoubleGroup
    :members:
    :show-inheritance:


Homogeneous Space
-----------------

.. autoclass:: escnn.group.HomSpace
    :members:
    :show-inheritance:

Representations
---------------

Representation
~~~~~~~~~~~~~~~

.. autoclass:: escnn.group.Representation
    :members:
    :undoc-members:

Irreducible Representation
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: escnn.group.IrreducibleRepresentation
    :members:
    :show-inheritance:

Utility Functions
-----------------

.. autofunction:: escnn.group.change_basis

.. autofunction:: escnn.group.directsum

.. autofunction:: escnn.group.disentangle

.. autofunction:: escnn.group.homomorphism_space

Subpackages
-----------

.. toctree::
   :maxdepth: 1
   
   escnn.group.utils
   
   
