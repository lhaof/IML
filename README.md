# Instance-aware Multi-task Learning for Nuclei Segmentation (IML)
This is the official PyTorch implementation of IML, a instance-aware multi-task learning framework that strengthens a pixel-wise prediction branch with an instance-wise prediction branch. The whole framework includes four key components: 1) a pixel-wise prediction branch, 2) an instance-wise prediction branch with an instance-disentangling feature learning module, 3) a dualbranch synchronizing training scheme, and 4) a dual-branch unified post-processing algorithm.

Part of the codes are from the implementation of [DINO](https://github.com/IDEA-Research/DINO).

> **If you intend to use anything from this repo, citation of the original publication given above is necessary**

![](diagram/fig1.png)

## Set Up Environment
```
conda install -c pytorch pytorch torchvision
pip install -r requirements.txt
cd models/dino/ops
python setup.py build install
```
