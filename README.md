<p align="center">
	<img src="https://github.com/daiquocnguyen/U2GNN/blob/master/logo.png">
</p>

# Transformer for Graph Classification<a href="https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2Fdaiquocnguyen%2FU2GNN%2Fblob%2Fmaster%2FREADME.md"><img alt="Twitter" src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fdaiquocng"></a>

<img alt="GitHub top language" src="https://img.shields.io/github/languages/top/daiquocnguyen/U2GNN"><a href="https://github.com/daiquocnguyen/U2GNN/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/daiquocnguyen/U2GNN"></a>
<img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/daiquocnguyen/U2GNN">
<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/daiquocnguyen/U2GNN">
<a href="https://github.com/daiquocnguyen/U2GNN/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/daiquocnguyen/U2GNN"></a>
<a href="https://github.com/daiquocnguyen/U2GNN/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/daiquocnguyen/U2GNN"></a>
<img alt="GitHub" src="https://img.shields.io/github/license/daiquocnguyen/U2GNN">

This program provides the implementation of our U2GNN as described in our paper [Universal Self-Attention Network for Graph Classification](https://arxiv.org/pdf/1909.11855.pdf) where we leverage the transformer self-attention network to construct an advanced aggregation function to learn graph representations.

<p align="center">
	<img src="https://github.com/daiquocnguyen/U2GNN/blob/master/U2GNN.png">
</p>

## Usage

### News

- 17-05-2020: Update Pytorch (1.5.0) implementation. 

### Requirements
- Python 	3.x
- Tensorflow 	1.14
- Tensor2tensor 1.13
- Networkx 	2.3
- Scikit-learn	0.21.2

### Training

Regarding our supervised U2GNN:

	U2GNN$ python train_U2GNN_Sup.py --dataset IMDBBINARY --batch_size 4 --ff_hidden_size 1024 --fold_idx 1 --num_neighbors 8 --num_sampled 512 --num_epochs 50 --num_timesteps 4 --learning_rate 0.0005 --model_name IMDBBINARY_bs4_fold1_dro05_1024_8_idx0_4_1
	
	U2GNN$ python train_U2GNN_Sup.py --dataset PTC --batch_size 4 --ff_hidden_size 1024 --fold_idx 1 --num_neighbors 16 --num_sampled 512 --num_epochs 50 --num_timesteps 3 --learning_rate 0.0005 --model_name PTC_bs4_fold1_dro05_1024_16_idx0_3_1

Regarding our unsupervised U2GNN:

	U2GNN$ python train_U2GNN_Unsup.py --dataset IMDBBINARY --batch_size 2 --ff_hidden_size 1024 --num_neighbors 8 --num_sampled 512 --num_epochs 50 --num_timesteps 1 --learning_rate 0.0001 --model_name IMDBBINARY_bs2_dro05_1024_8_idx0_1_2
	
	U2GNN$ python train_U2GNN_Unsup.py --dataset PTC --batch_size 2 --degree_as_tag --ff_hidden_size 1024 --num_neighbors 4 --num_sampled 512 --num_epochs 50 --num_timesteps 2 --learning_rate 0.0001 --model_name PTC_bs2_dro05_1024_4_idx0_2_2

- Note that we follow some unsupervised approaches (such as DGK, Graph2Vec, and AWE) to use all nodes from the entire dataset (i.e., including all nodes in the test set) during training our "unsupervised" GCN baseline and U2GNN models. You should change to the `log_uniform` directory to perform `make` to build `SampledSoftmax`, and then add the `log_uniform` directory to your PYTHONPATH.

## Cite  
Please cite the paper whenever U2GNN is used to produce published results or incorporated into other software:

	@article{Nguyen2019U2GNN,
		author={Dai Quoc Nguyen and Tu Dinh Nguyen and Dinh Phung},
		title={{Universal Self-Attention Network for Graph Classification}},
		journal={arXiv preprint arXiv:1909.11855},
		year={2019}
	}

## License
As a free open-source implementation, U2GNN is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. All other warranties including, but not limited to, merchantability and fitness for purpose, whether express, implied, or arising by operation of law, course of dealing, or trade usage are hereby disclaimed. I believe that the programs compute what I claim they compute, but I do not guarantee this. The programs may be poorly and inconsistently documented and may contain undocumented components, features or modifications. I make no guarantee that these programs will be suitable for any application.

U2GNN is licensed under the Apache License 2.0.
