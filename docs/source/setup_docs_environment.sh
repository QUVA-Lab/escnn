
conda create --name docsenv python=3.7
source activate docsenv

conda install -y pytorch=1.8 torchvision cpuonly -c pytorch
conda install scipy
pip install pymanopt
conda install autograd
pip install lie_learn
pip install joblib
pip install torch-geometric

pip install sphinx
pip install sphinx-rtd-theme
pip install sphinx-autodoc-typehints
