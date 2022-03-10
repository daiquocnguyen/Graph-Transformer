<p align="center">
	<img src="https://github.com/daiquocnguyen/U2GNN/blob/master/logo.png">
</p>

# Universal Graph Transformer Self-Attention Networks<a href="https://twitter.com/intent/tweet?text=Wow:&url=https%3A%2F%2Fgithub.com%2Fdaiquocnguyen%2FU2GNN%2Fblob%2Fmaster%2FREADME.md"><img alt="Twitter" src="https://img.shields.io/twitter/url?style=social&url=https%3A%2F%2Ftwitter.com%2Fdaiquocng"></a>

<img alt="GitHub top language" src="https://img.shields.io/github/languages/top/daiquocnguyen/U2GNN"><img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/daiquocnguyen/U2GNN">
<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/daiquocnguyen/U2GNN">
<a href="https://github.com/daiquocnguyen/U2GNN/network"><img alt="GitHub forks" src="https://img.shields.io/github/forks/daiquocnguyen/U2GNN"></a>
<a href="https://github.com/daiquocnguyen/U2GNN/stargazers"><img alt="GitHub stars" src="https://img.shields.io/github/stars/daiquocnguyen/U2GNN"></a>
<img alt="GitHub" src="https://img.shields.io/github/license/daiquocnguyen/U2GNN">

This program provides the implementation of our graph transformer, named UGformer, as described in [our paper](https://arxiv.org/pdf/1909.11855.pdf), where we leverage the transformer self-attention network to learn graph representations in both supervised inductive setting and unsupervised transductive setting.

Variant 1            |  Variant 2
:-------------------------:|:-------------------------:
![](https://github.com/daiquocnguyen/U2GNN/blob/master/UGformer_v1.png)  |  ![](https://github.com/daiquocnguyen/U2GNN/blob/master/UGformer_v2.png)


## Usage

### News
- 05-03-2022: Our Graph Transformer paper has been accepted to the Poster and Demo Track at The ACM Web Conference 2022.

- 20-08-2021: Release a Pytorch implementation to apply the Variant 2 for inductive text classification.

- 04-05-2021: Release a Pytorch 1.5.0 implementation (i.e., Variant 2) to leverage the transformer on all input nodes.

- 17-05-2020: Release a Pytorch 1.5.0 implementation. 

- 11-12-2019: Release a Tensorflow 1.14 implementation for both supervised inductive setting and unsupervised transductive setting.

### Training
		
- Variant 1: Leveraging the transformer on sampled neighbors of each node:

		$ python train_UGformerV1_Sup.py --dataset IMDBBINARY --batch_size 4 --ff_hidden_size 1024 --fold_idx 1 --num_neighbors 8 --num_epochs 50 --num_timesteps 4 --learning_rate 0.0005 --model_name IMDBBINARY_bs4_fold1_1024_8_idx0_4_1
	
- Variant 2: Leveraging the transformer directly on all nodes of the input graph:
		 
		$ python train_UGformerV2.py --dataset PTC --ff_hidden_size 1024 --fold_idx 1 --num_epochs 50 --num_timesteps 3 --learning_rate 0.0005 --model_name PTC_fold1_1024_idx0_1
		
- Applying the Variant 2 for inductive text classification:

		$ python train_TextGNN.py --dataset mr --learning_rate 0.0001 --batch_size 4096 --num_epochs 150 --num_GNN_layers 2 --hidden_size 384 --model GatedGT

- Applying an unsupervised transductive setting for graph classification: 

		$ python train_UGformerV1_UnSup.py --dataset PTC --batch_size 2 --degree_as_tag --ff_hidden_size 1024 --num_neighbors 4 --num_sampled 512 --num_epochs 50 --num_timesteps 2 --learning_rate 0.0001 --model_name PTC_bs2_dro05_1024_4_idx0_2_2


#### Requirements
- Python 	3.x
- Tensorflow 	1.14 & Tensor2tensor 1.13
- Pytorch >= 1.5.0
- Networkx 	2.3
- Scikit-learn	0.21.2

## Cite  
Please cite the paper whenever our graph transformer is used to produce published results or incorporated into other software:

	@inproceedings{NguyenUGformer,
		author={Dai Quoc Nguyen and Tu Dinh Nguyen and Dinh Phung},
		title={Universal Graph Transformer Self-Attention Networks},
		booktitle={Companion Proceedings of the Web Conference 2022 (WWW '22 Companion)},
		year={2022}
	}

## License
As a free open-source implementation, Graph-Transformer is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. All other warranties including, but not limited to, merchantability and fitness for purpose, whether express, implied, or arising by operation of law, course of dealing, or trade usage are hereby disclaimed. I believe that the programs compute what I claim they compute, but I do not guarantee this. The programs may be poorly and inconsistently documented and may contain undocumented components, features or modifications. I make no guarantee that these programs will be suitable for any application.

Graph-Transformer is licensed under the Apache License 2.0.
