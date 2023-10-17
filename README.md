
E(n)-equivariant Steerable CNNs (*escnn*)
--------------------------------------------------------------------------------

*escnn* is a [PyTorch](https://pytorch.org/) extension for equivariant deep learning.
*escnn* is the successor of the [e2cnn](<https://github.com/QUVA-Lab/e2cnn>) library, which only supported planar isometries.
Instead, *escnn* supports steerable CNNs equivariant to both 2D and 3D isometries, as well as equivariant MLPs.

<div align="center">
<table>
  <tbody>
    <tr><td style="border: none">
        <b><a href='https://quva-lab.github.io/escnn/' target='_blank'>&nbsp;
            Documentation</a>&nbsp;</b>
    </td><td style="border: none">
        <b><a href='https://openreview.net/forum?id=WE4qe9xlnQw' target='_blank'>&nbsp;
            Paper ICLR 22</a>&nbsp;</b>
    </td><td style="border: none">
        <b><a href='https://gabri95.github.io/Thesis/thesis.pdf' target='_blank'>&nbsp;
            MSc Thesis Gabriele</a>&nbsp;</b> 
    </td><td style="border: none">
        <b><a href='https://github.com/QUVA-Lab/e2cnn' target='_blank'>&nbsp;
            e2cnn library</a>&nbsp;</b>
    </td></tr>
    <tr></tr>
    <tr><td style="border: none">
    </td><td style="border: none">
        <b><a href='https://arxiv.org/abs/1911.08251' target='_blank'>&nbsp;
            Paper NeurIPS 19</a>&nbsp;</b>
    </td><td style="border: none">
        <b><a href='https://maurice-weiler.gitlab.io/#cnn_book' target='_blank'>&nbsp;
            PhD Thesis Maurice</a>&nbsp;</b>
    </td><td style="border: none">
        <b><a href='https://github.com/QUVA-Lab/e2cnn_experiments' target='_blank'>&nbsp;
            e2cnn experiments</a>&nbsp;</b>
    </td></tr>
  </tbody>
</table>
</div>

If you prefer using Jax, check our this fork [escnn_jax](https://github.com/emilemathieu/escnn_jax) of our library!

--------------------------------------------------------------------------------

*Equivariant neural networks* guarantee a specified transformation behavior of their feature spaces under transformations of their input.
For instance, classical convolutional neural networks (*CNN*s) are by design equivariant to translations of their input.
This means that a translation of an image leads to a corresponding translation of the network's feature maps.
This package provides implementations of neural network modules which are equivariant under all *isometries* $\mathrm{E}(2)$ of the image plane $\mathbb{R}^2$ and all *isometries* $\mathrm{E}(3)$ of the 3D space $\mathbb{R}^3$, that is, under *translations*, *rotations* and *reflections* (and can, potentially, be extended to all isometries $\mathrm{E}(n)$ of $\mathbb{R}^n$).
In contrast to conventional CNNs, $\mathrm{E}(n)$-equivariant models are guaranteed to generalize over such transformations, and are therefore more data efficient.

The feature spaces of $\mathrm{E}(n)$-equivariant Steerable CNNs are defined as spaces of *feature fields*, being characterized by their transformation law under rotations and reflections.
Typical examples are scalar fields (e.g. gray-scale images or temperature fields) or vector fields (e.g. optical flow or electromagnetic fields).

![feature field examples](https://github.com/QUVA-Lab/escnn/raw/master/visualizations/feature_fields.png)

Instead of a number of channels, the user has to specify the field *types* and their *multiplicities* in order to define a feature space.
Given a specified input- and output feature space, our ``R2conv`` and ``R3conv`` modules instantiate the *most general* convolutional mapping between them.
Our library provides many other equivariant operations to process feature fields, including nonlinearities, mappings to produce invariant features, batch normalization and dropout.

In theory, feature fields are defined on continuous space $\mathbb{R}^n$.
In practice, they are either sampled on a *pixel grid* or given as a *point cloud*.
escnn represents feature fields by ``GeometricTensor`` objects, which wrap a ``torch.Tensor`` with the corresponding transformation law.
All equivariant operations perform a dynamic type-checking in order to guarantee a geometrically sound processing of the feature fields.


To parameterize steerable kernel spaces, equivariant to an arbitrary compact group $G$,
in our [paper](https://openreview.net/forum?id=WE4qe9xlnQw), we generalize the Wigner-Eckart theorem in
[A Wigner-Eckart Theorem for Group Equivariant Convolution Kernels](https://arxiv.org/abs/2010.10952)
from $G$-homogeneous spaces to more general spaces $X$ carrying a $G$-action. 
In short, our method leverages a $G$-steerable basis for unconstrained scalar filters over the whole Euclidean space $\mathbb{R}^n$ to generate steerable kernel spaces with arbitrary input and output field *types*.
For example, the left side of the next image shows two elements of a $\mathrm{SO}(2)$-steerable basis for functions on $\mathbb{R}^2$ which are used to generate two basis elements for $\mathrm{SO}(2)$-equivariant steerable kernels on the right.
In particular, the steerable kernels considered map a frequency $l=1$ vector field (2 channels) to a frequency $J=2$ 
vector field (2 channels).

![we_theorem_example](https://github.com/QUVA-Lab/escnn/raw/master/visualizations/wigner_eckart_theorem_2.png)


$\mathrm{E}(n)$-Equivariant Steerable CNNs unify and generalize a wide range of isometry equivariant CNNs in one single framework.
Examples include:
- [Group Equivariant Convolutional Networks](https://arxiv.org/abs/1602.07576)
- [Harmonic Networks: Deep Translation and Rotation Equivariance](https://arxiv.org/abs/1612.04642)
- [Steerable CNNs](https://arxiv.org/abs/1612.08498)
- [Rotation equivariant vector field networks](https://arxiv.org/abs/1612.09346)
- [Learning Steerable Filters for Rotation Equivariant CNNs](https://arxiv.org/abs/1711.07289)
- [HexaConv](https://arxiv.org/abs/1803.02108)
- [Roto-Translation Covariant Convolutional Networks for Medical Image Analysis](https://arxiv.org/abs/1804.03393)
- [3D Steerable CNNs](https://arxiv.org/abs/1807.02547)
- [Tensor Field Networks](https://arxiv.org/abs/1802.08219)
- [Cormorant: Covariant Molecular Neural Networks](https://arxiv.org/abs/1906.04015)
- [3D GCNNs for Pulmonary Nodule Detection](https://arxiv.org/abs/1804.04656)


For more details, we refer to our ICLR 2022 paper [A Program to Build E(N)-Equivariant Steerable CNNs](https://openreview.net/forum?id=WE4qe9xlnQw)
and our NeurIPS 2019 paper [General E(2)-Equivariant Steerable CNNs](https://arxiv.org/abs/1911.08251).

--------------------------------------------------------------------------------

The library is structured into four subpackages with different high-level features:

| Component                                                                         | Description                                                      |
|-----------------------------------------------------------------------------------|------------------------------------------------------------------|
| [**escnn.group**](https://github.com/QUVA-Lab/escnn/tree/master/escnn/group/)     | implements basic concepts of *group* and *representation* theory |
| [**escnn.kernels**](https://github.com/QUVA-Lab/escnn/tree/master/escnn/kernels/) | solves for spaces of equivariant convolution kernels             |
| [**escnn.gspaces**](https://github.com/QUVA-Lab/escnn/tree/master/escnn/gspaces/) | defines the Euclidean spaces and their symmetries                |
| [**escnn.nn**](https://github.com/QUVA-Lab/escnn/tree/master/escnn/nn/)           | contains equivariant modules to build deep neural networks       |
--------------------------------------------------------------------------------------------------------------------------------------------------

> **WARNING**:
> **escnn.kernels** received major refactoring in version 1.0.0 and it is not compatible with previous versions of the library. These changes do not affect the interface provided in the rest of the library but, sometimes, the weights of a network trained with a previous version might not load correctly in a newly instantiated model.
> We recommend using version [v0.1.9](https://github.com/QUVA-Lab/escnn/tree/v0.1.9) for backward compatibility.



## Demo

Since $\mathrm{E}(2)$-steerable CNNs are equivariant under rotations and reflections, their inference is independent from the choice of image orientation.
The visualization below demonstrates this claim by feeding rotated images into a randomly initialized $\mathrm{E}(2)$-steerable CNN (left).
The middle plot shows the equivariant transformation of a feature space, consisting of one scalar field (color-coded) and one vector field (arrows), after a few layers.
In the right plot we transform the feature space into a comoving reference frame by rotating the response fields back (stabilized view).

![Equivariant CNN output](https://github.com/QUVA-Lab/escnn/raw/master/visualizations/vectorfield.gif)

The invariance of the features in the comoving frame validates the rotational equivariance of $\mathrm{E}(2)$-steerable CNNs empirically.
Note that the fluctuations of responses are discretization artifacts due to the sampling of the image on a pixel grid, which does not allow for exact continuous rotations.
<!-- Note that the fluctuations of responses are due to discretization artifacts coming from the  -->

For comparison, we show a feature map response of a conventional CNN for different image orientations below.

![Conventional CNN output](https://github.com/QUVA-Lab/escnn/raw/master/visualizations/conventional_cnn.gif)

Since conventional CNNs are not equivariant under rotations, the response varies randomly with the image orientation.
This prevents CNNs from automatically generalizing learned patterns between different reference frames.


## Experimental results

$\mathrm{E}(n)$-steerable convolutions can be used as a drop in replacement for the conventional convolutions used in CNNs.
While using the same base architecture (with similar memory and computational cost), 
this leads to significant performance boosts compared to CNN baselines (values are test accuracies in percent).

| model        | Rotated ModelNet10 |
|--------------|--------------------|
| CNN baseline | 82.5       ± 1.4   |
| SO(2)-CNN    | 86.9       ± 1.9   |
| Octa-CNN     | 89.7       ± 0.6   |
| Ico-CNN      | 90.0       ± 0.6   |
| SO(3)-CNN    | 89.5       ± 1.0   |

All models share approximately the same architecture and width.
For more details we refer to our [paper](https://openreview.net/forum?id=WE4qe9xlnQw).

This library supports $\mathrm{E}(2)$-steerable CNNs implemented in our previous [e2cnn](<https://github.com/QUVA-Lab/e2cnn>) library as a special case; 
we include some representative results in the 2D setting from there:

| model        | CIFAR-10                | CIFAR-100                | STL-10             |
|--------------|-------------------------|--------------------------|--------------------|
| CNN baseline | 2.6 &nbsp; ± 0.1 &nbsp; | 17.1 &nbsp; ± 0.3 &nbsp; | 12.74 ± 0.23       |
| E(2)-CNN *   | 2.39       ± 0.11       | 15.55       ± 0.13       | 10.57 ± 0.70       |
| E(2)-CNN     | 2.05       ± 0.03       | 14.30       ± 0.09       | &nbsp; 9.80 ± 0.40 |

While using the same training setup (*no further hyperparameter tuning*) used for the CNN baselines, the equivariant models achieve significantly better results (values are test errors in percent).
For a fair comparison, the models without * are designed such that the number of parameters of the baseline is approximately preserved while models with * preserve the number of channels, and hence compute.
For more details we refer to our previous *e2cnn* [paper](https://arxiv.org/abs/1911.08251).


## Getting Started

*escnn* is easy to use since it provides a high level user interface which abstracts most intricacies of group and representation theory away.
The following code snippet shows how to perform an equivariant convolution from an RGB-image to 10 *regular* feature fields (corresponding to a
[group convolution](https://arxiv.org/abs/1602.07576)).

```python3
from escnn import gspaces                                          #  1
from escnn import nn                                               #  2
import torch                                                       #  3
                                                                   #  4
r2_act = gspaces.rot2dOnR2(N=8)                                    #  5
feat_type_in  = nn.FieldType(r2_act,  3*[r2_act.trivial_repr])     #  6
feat_type_out = nn.FieldType(r2_act, 10*[r2_act.regular_repr])     #  7
                                                                   #  8
conv = nn.R2Conv(feat_type_in, feat_type_out, kernel_size=5)       #  9
relu = nn.ReLU(feat_type_out)                                      # 10
                                                                   # 11
x = torch.randn(16, 3, 32, 32)                                     # 12
x = feat_type_in(x)                                                # 13
                                                                   # 14
y = relu(conv(x))                                                  # 15
```

Line 5 specifies the symmetry group action on the image plane $\mathbb{R}^2$ under which the network should be equivariant.
We choose the 
[*cyclic group*](https://en.wikipedia.org/wiki/Cyclic_group)
 $\mathrm{C}_8$, which describes discrete rotations by multiples of $2\pi/8$.
Line 6 specifies the input feature field types.
The three color channels of an RGB image are thereby to be identified as three independent scalar fields, which transform under the
[*trivial representation*](https://en.wikipedia.org/wiki/Trivial_representation)
 of $\mathrm{C}_8$ (when the input image is rotated, the RGB values do not change; compare the scalar and vector fields in the first image above).
Similarly, the output feature space in line 7 is specified to consist of 10 feature fields which transform under the
[*regular representation*](https://en.wikipedia.org/wiki/Regular_representation)
of $\mathrm{C}_8$.
The $\mathrm{C}_8$-equivariant convolution is then instantiated by passing the input and output type as well as the kernel size to the constructor (line 9).
Line 10 instantiates an equivariant ReLU nonlinearity which will operate on the output field and is therefore passed the output field type.

Lines 12 and 13 generate a random minibatch of RGB images and wrap them into a `nn.GeometricTensor` to associate them
with their correct field type `feat_type_in`.
The equivariant modules process the geometric tensor in line 15.
Each module is thereby checking whether the geometric tensor passed to them satisfies the expected transformation law.

Because the parameters do not need to be updated anymore at test time, after training, any equivariant network can be 
converted into a pure PyTorch model with no additional computational overhead in comparison to conventional CNNs.
The code currently supports the automatic conversion of a few commonly used modules through the `.export()` method; 
check the [documentation](https://quva-lab.github.io/escnn/api/escnn.nn.html) for more details.

To get started, we provide some examples and tutorials:
- The [introductory tutorial](https://github.com/QUVA-Lab/escnn/blob/master/examples/introduction.ipynb) introduces the basic functionality of the library.
- A second [tutorial](https://github.com/QUVA-Lab/escnn/blob/master/examples/model.ipynb) goes through building and training
an equivariant model on the rotated MNIST dataset.
- Note that *escnn* also supports equivariant MLPs; see [these examples](https://github.com/QUVA-Lab/escnn/blob/master/examples/mlp.ipynb).
- Check also the [tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Geometric_deep_learning/tutorial2_steerable_cnns.html) on Steerable CNNs using our library in the *Deep Learning 2* course at the University of Amsterdam.

More complex 2D equivariant *Wide Resnet* models are implemented in [e2wrn.py](https://github.com/QUVA-Lab/escnn/blob/master/examples/e2wrn.py).
To try a model which is equivariant under reflections call:
```
cd examples
python e2wrn.py
```
A version of the same model which is simultaneously equivariant under reflections and rotations of angles multiple of 90 degrees can be run via:
```
python e2wrn.py --rot90
```
You can find more examples in the [example](https://github.com/QUVA-Lab/escnn/tree/master/examples) folder.
For instance, [se3_3Dcnn.py](https://github.com/QUVA-Lab/escnn/blob/master/examples/se3_3Dcnn.py) implements a 3D CNN equivariant to
rotations and translations in 3D. You can try it with
```
cd examples
python se3_3Dcnn.py
```

## Useful material to learn about Equivariance and Steerable CNNs

If you want to better understand the theory behind equivariant and steerable neural networks, you can check these references:
- Erik Bekkers' [lectures](https://uvagedl.github.io/) on *Geometric Deep Learning* at in the Deep Learning 2 course at the University of Amsterdam
- The course material also includes a [tutorial](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Geometric_deep_learning/tutorial1_regular_group_convolutions.html) on *group convolution* and [another](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/DL2/Geometric_deep_learning/tutorial2_steerable_cnns.html) about Steerable CNNs, using *this library*.
- Gabriele's [MSc thesis](https://gabri95.github.io/Thesis/thesis.pdf) provides a brief overview of the essential mathematical ingredients needed to understand Steerable CNNs.
- Maurice's [PhD thesis](https://maurice-weiler.gitlab.io/#cnn_book) develops the representation theory of steerable CNNs, deriving the most prominent layers and explaining the gauge theoretic viewpoint. 

## Dependencies

The library is based on Python3.7

```
torch>=1.3
numpy
scipy
lie_learn
joblib
py3nj
```
Optional:
```
torch-geometric
pymanopt>=1.0.0
autograd
```

> **WARNING**: `py3nj` enables a fast computation of Clebsh Gordan coefficients.
If this package is not installed, our library relies on a numerical method to estimate them.
This numerical method is not guaranteed to return the same coefficients computed by `py3nj` (they can differ by a sign).
For this reason, models built with and without `py3nj` might not be compatible.

> To successfully install `py3nj` you may need a Fortran compiler installed in you environment.

## Installation

You can install the latest [release](https://github.com/QUVA-Lab/escnn/releases) as

```
pip install escnn
```

or you can clone this repository and manually install it with
```
pip install git+https://github.com/QUVA-Lab/escnn
```


## Contributing

Would you like to contribute to **escnn**? That's great!

Then, check the instructions in [CONTRIBUTING.md](https://github.com/QUVA-Lab/escnn/blob/master/CONTRIBUTING.md) and help us to
improve the library!


Do you have any doubts? Do you have some idea you would like to discuss? 
Feel free to open a new thread under in [Discussions](https://github.com/QUVA-Lab/escnn/discussions)!

## Cite

The development of this library was part of the work done for our papers
[A Program to Build E(N)-Equivariant Steerable CNNs](https://openreview.net/forum?id=WE4qe9xlnQw)
and [General E(2)-Equivariant Steerable CNNs](https://arxiv.org/abs/1911.08251).
Please cite these works if you use our code:

```

   @inproceedings{cesa2022a,
        title={A Program to Build {E(N)}-Equivariant Steerable {CNN}s },
        author={Gabriele Cesa and Leon Lang and Maurice Weiler},
        booktitle={International Conference on Learning Representations},
        year={2022},
        url={https://openreview.net/forum?id=WE4qe9xlnQw}
    }
    
   @inproceedings{e2cnn,
       title={{General E(2)-Equivariant Steerable CNNs}},
       author={Weiler, Maurice and Cesa, Gabriele},
       booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
       year={2019},
       url={https://arxiv.org/abs/1911.08251}
   }
```

Feel free to [contact us](mailto:cesa.gabriele@gmail.com,m.weiler.ml@gmail.com).

## License

*escnn* is distributed under BSD Clear license. See LICENSE file.
