<p align="center">
	<img src="https://github.com/daiquocnguyen/U2GNN/blob/master/u2gnn_logo.png">
</p>

# Transformer for Graph Classification<a href="https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2Fdaiquocnguyen%2FU2GNN%2Fblob%2Fmaster%2FREADME.md"><img alt="Twitter" src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fdaiquocng"></a>

<img alt="GitHub top language" src="https://img.shields.io/github/languages/top/daiquocnguyen/U2GNN"><a href="https://github.com/daiquocnguyen/U2GNN/issues"><img alt="GitHub issues" src="https://img.shields.io/github/issues/daiquocnguyen/U2GNN"></a>
<img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/daiquocnguyen/U2GNN">
<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/daiquocnguyen/U2GNN">
<a href="https://github.com/daiquocnguyen/U2GNN/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/daiquocnguyen/U2GNN"></a>
<a href="https://github.com/daiquocnguyen/U2GNN/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/daiquocnguyen/U2GNN"></a>
<img alt="GitHub" src="https://img.shields.io/github/license/daiquocnguyen/U2GNN">

- This program provides the implementation of our U2GNN as described in our paper, titled [Universal Self-Attention Network for Graph Classification](https://arxiv.org/abs/1909.11855), where we induce an advanced aggregation function using a transformer self-attention network to produce plausible node and graph embeddings. In general, our supervised and unsupervised U2GNN models produce state-of-the-art accuracies on most of the benchmark datasets. 

- Regarding our unsupervised learning, we view the aggregation function as an encoder to encode the substructure around a given node into a vector. Then we make the similarity between this encoded vector and the embedding of the given node higher than that between the encoded vector and the embeddings of the other nodes. As a result, our unsupervised learning helps to recognize and distinguish substructures within each graph, leading to effectively identify structural differences among graphs.

<p align="center">
	<img src="https://github.com/daiquocnguyen/U2GNN/blob/master/U2GNN.png">
</p>

## Usage

### News

- 17-05-2020: Update Pytorch (1.5.0) implementation. You should change to the `log_uniform` directory to perform `make` to build `SampledSoftmax`, and then add the `log_uniform` directory to your PYTHONPATH.

### Requirements
- Python 	3.x
- Tensorflow 	1.14
- Tensor2tensor 1.13
- Networkx 	2.3
- Scikit-learn	0.21.2

### Training

Regarding our supervised U2GNN:

	U2GNN$ python train_U2GNN_Sup.py --dataset IMDBBINARY --batch_size 4 --ff_hidden_size 1024 --fold_idx 1 --num_neighbors 8 --num_sampled 512 --num_epochs 50 --num_hidden_layers 4 --learning_rate 0.0005 --model_name IMDBBINARY_bs4_fold1_dro05_1024_8_idx0_4_1
	
	U2GNN$ python train_U2GNN_Sup.py --dataset PTC --batch_size 4 --ff_hidden_size 1024 --fold_idx 1 --num_neighbors 16 --num_sampled 512 --num_epochs 50 --num_hidden_layers 3 --learning_rate 0.0005 --model_name PTC_bs4_fold1_dro05_1024_16_idx0_3_1

Regarding our unsupervised U2GNN:

	U2GNN$ python train_U2GNN_Unsup.py --dataset IMDBBINARY --batch_size 2 --ff_hidden_size 1024 --num_neighbors 8 --num_sampled 512 --num_epochs 50 --num_hidden_layers 1 --learning_rate 0.0001 --model_name IMDBBINARY_bs2_dro05_1024_8_idx0_1_2
	
	U2GNN$ python train_U2GNN_Unsup.py --dataset PTC --batch_size 2 --degree_as_tag --ff_hidden_size 1024 --num_neighbors 4 --num_sampled 512 --num_epochs 50 --num_hidden_layers 2 --learning_rate 0.0001 --model_name PTC_bs2_dro05_1024_4_idx0_2_2

NOTE THAT we follow some unsupervised approaches (such as DGK, Graph2Vec, and AWE) to use all nodes from the entire dataset (i.e., the union of training and test sets) during training our "unsupervised" U2GNN, as most of the recent supervised works cite the accuracy results of the unsupervised ones and accept the difference in the training data.

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
