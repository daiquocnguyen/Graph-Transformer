### Usage: Graph Transformer (Variant 2) for Inductive Text Classification

- Download and unzip `glove.6B.300d.txt` from [http://nlp.stanford.edu/data/glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip).

- Construct a text graph from each textual dataset:

      $ build_graph.py <dataset> <word_embedding_dim> <path_to_glove_directory>

- Apply the Variant 2 for inductive text classification:

      $ python train_TextGNN.py --dataset mr --learning_rate 0.0001 --batch_size 4096 --num_epochs 150 --num_GNN_layers 2 --hidden_size 384 --model GatedGT
		  
      $ python train_TextGNN.py --dataset R8 --learning_rate 0.0001 --batch_size 4096 --num_epochs 150 --num_GNN_layers 2 --hidden_size 384 --model GatedGT
		
      $ python train_TextGNN.py --dataset R52 --learning_rate 0.0005 --batch_size 4096 --num_epochs 150 --num_GNN_layers 2 --hidden_size 384 --model GatedGT
		
      $ python train_TextGNN.py --dataset ohsumed --learning_rate 0.001 --batch_size 4096 --num_epochs 150 --num_GNN_layers 2 --hidden_size 384 --model GatedGT
