# NDA_SNN

Pytorch implementation of Neuromorphic Data Augmentation for SNN, Accepted to ECCV 2022.
Paper link: [Neuromorphic Data Augmentation for Training Spiking Neural Networks](https://arxiv.org/pdf/2203.06145.pdf).


## Dataset Preparation

For CIFAR10-DVS dataset, please refer the Google Drive link below:

+ [Training set](https://drive.google.com/file/d/1pzYnhoUvtcQtxk_Qmy4d2VrhWhy5R-t9/view?usp=sharing)
+ [Test set](https://drive.google.com/file/d/1q1k6JJgVH3ZkHWMg2zPtrZak9jRP6ggG/view?usp=sharing)

For N-Caltech 101, we suggest using [SpikingJelly](https://github.com/fangwei123456/spikingjelly) package to pre-process the data. 
Specifically, initialize the `NCaltech101` in SpikingJelly as:

```python
from spikingjelly.datasets.n_caltech101 import NCaltech101
dataset = NCaltech101(root='data', data_type='frame', frames_number=10, split_by='time')
```
If you can initialize this class, then you will be able to use our provided dataloader in `functions/data_loaders.py`


## Run Experiments

To run a VGG-11 without NDA on CIFAR10-DVS:

`python main.py --dset dc10 --amp`

Here, `--amp` use FP16 training which can accelerate the training stage.
Use `--dset nc101` to change the dataset to NCaltech 101. 

To enable NDA training:

`python main.py --dset dc10 --amp --nda`

### Reference

If you find our work is interesting, please consider cite us:

```bibtex
@article{li2022neuromorphic,
  title={Neuromorphic Data Augmentation for Training Spiking Neural Networks},
  author={Li, Yuhang and Kim, Youngeun and Park, Hyoungseob and Geller, Tamar and Panda, Priyadarshini},
  journal={arXiv preprint arXiv:2203.06145},
  year={2022}
}
```


