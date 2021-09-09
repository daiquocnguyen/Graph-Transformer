### Graph Transformer (UGformer v2) for Inductive Text Classification

- Download `glove.6B.300d.txt` from [http://nlp.stanford.edu/data/glove.6B.zip](http://nlp.stanford.edu/data/glove.6B.zip) and unzip to `TextGNN`.

- Step 1: Build a text graph from each textual document:

      $ build_graph.py mr 300 glove
      
      $ build_graph.py R8 300 glove
      
      $ build_graph.py R52 300 glove
      
      $ build_graph.py ohsumed 300 glove

- Step 2: Apply the Variant 2 for inductive text classification:

      $ python train_TextGNN.py --dataset mr --learning_rate 0.0001 --batch_size 4096 --num_epochs 150 --num_GNN_layers 2 --hidden_size 384 --model GatedGT
		  
      $ python train_TextGNN.py --dataset R8 --learning_rate 0.0001 --batch_size 4096 --num_epochs 150 --num_GNN_layers 2 --hidden_size 384 --model GatedGT
		
      $ python train_TextGNN.py --dataset R52 --learning_rate 0.0005 --batch_size 4096 --num_epochs 150 --num_GNN_layers 2 --hidden_size 384 --model GatedGT
		
      $ python train_TextGNN.py --dataset ohsumed --learning_rate 0.001 --batch_size 4096 --num_epochs 150 --num_GNN_layers 2 --hidden_size 384 --model GatedGT
      
For a fair comparison, we run `10 times`, wherein for each time, we run Step 1 and then Step 2.
