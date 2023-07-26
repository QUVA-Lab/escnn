
## Getting Started

To get started, we provide some examples and tutorials:
- The [introductory tutorial](https://github.com/QUVA-Lab/escnn/blob/master/examples/introduction.ipynb) introduces the basic functionality of the library.
- A second [tutorial](https://github.com/QUVA-Lab/escnn/blob/master/examples/model.ipynb) goes through building and training
  an equivariant model on the rotated MNIST dataset.
- Note that *escnn* also supports equivariant MLPs; see [these examples](https://github.com/QUVA-Lab/escnn/blob/master/examples/mlp.ipynb).
- Check also the [tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Geometric_deep_learning/tutorial2_steerable_cnns.html) on Steerable CNNs using our library in the *Deep Learning 2* course at the University of Amsterdam.

More complex equivariant *Wide Resnet* models are implemented in [e2wrn.py](https://github.com/QUVA-Lab/escnn/blob/master/examples/e2wrn.py).
To try a model which is equivariant under reflections call:
```
cd examples
python e2wrn.py
```
A version of the same model which is simultaneously equivariant under reflections and rotations of angles multiple of 90 degrees can be run via:
```
python e2wrn.py --rot90
```
For a 3D CNN equivariant to rotations and translations in 3D, check
[se3_3Dcnn.py](https://github.com/QUVA-Lab/escnn/blob/master/examples/se3_3Dcnn.py):
```
cd examples
python se3_3Dcnn.py
```
Moreover, [mlp.ipynb](https://github.com/QUVA-Lab/escnn/blob/master/examples/mlp.ipynb) implements some equivariant MLPs.


## Useful References to learn about Equivariance and Steerable CNNs

If you want to better understand the theory behind equivariant and steerable neural networks, you can check these references:
- Erik Bekkers' [lectures](https://uvagedl.github.io/) on *Geometric Deep Learning* at in the Deep Learning 2 course at the University of Amsterdam
- The course material also includes a [tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Geometric_deep_learning/tutorial1_regular_group_convolutions.html) on *group convolution* and [another](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Geometric_deep_learning/tutorial2_steerable_cnns.html) about Steerable CNNs, using *this library*.
- My [thesis](https://gabri95.github.io/Thesis/thesis.pdf) provides a brief overview of the essential mathematical ingredients needed to understand Steerable CNNs.
