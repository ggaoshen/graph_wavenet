## CS224W Final Project - Graph WaveNet

Graph Neural Networks have proven to be a highly effective modeling technique for a multitude of real-world datasets.  GNNs can learn over a rich, relational structure of heterogeneous entities, modeling the complexity seen in many disparate domains where traditional deep learning techniques would otherwise fall short.  Within this field, one of the cutting-edge areas of research is how to model on top of graphs that are changing over time.  This differs from more traditional GNNs, where a snapshot of a static graph and its features are used to train a model that can be used to make predictions on new input data.  With temporal graphs, the training process must support a time dimension, and often the task is to predict the state of the system at some point in the future.  This area of research greatly interested us since real-world datasets often have multiple temporal properties.  Indeed, spatial-temporal graph modeling has received increased attention due to its applicability in diverse problem spaces, encompassing areas like traffic speed forecasting, taxi demand prediction, human action recognition, and more.  Theoretically, capturing this temporal dimension during training should learn a richer internal representation than simply training over a snapshot of the graph at some moment in time.

We can further analyze the temporal properties of graphs and see that it comes in different flavors.  The figure below from [2] shows several different types of temporal graphs.  Dynamic graphs are ones where the edges between nodes are changing at each time period.  Both (a) and (b) are dynamic graphs, but (a) has the added complexity where the signals, or features assigned to each node, are also changing over time.  The graph in (c) uses a static graph, but with these same temporal node features.  There is also a fourth type, not depicted here, where the graph and node features are static, but edge features, such as weights, are what change over time.


## Setup
    sh setup.sh

## Training Graph WaveNet using the Chickenpox Cases in Hungary dataset
    python3 src/train_temporal_val.py

## Resources
- Original Graph WaveNet [Paper](https://arxiv.org/pdf/1906.00121.pdf) | [Code](https://github.com/nnzhan/Graph-WaveNet) 
