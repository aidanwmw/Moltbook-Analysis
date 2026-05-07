# ECE 408 Final Project Code

Aidan Meador-Woodruff, Michel Malda
Department of Physics and Astronomy, University of Rochester
Spring Semester 2026

In this directory is the code for our final machine learning project on moltbook and reddit datasets. The files are as follows:

- FigureHelpers.ipynb: This file is figures for our presentation.

- GraphStats.ipynb: This file is an overview of graph-level statistics for our own research out of class (for example, fitting power-law distributions and looking at the change in time of the topology). More network theoretic stuff than ML and was not used for classification, but was useful for our paper's introduction so we included it. 

- Random_Forest.ipynb : This file is the random forest classifier for the moltbook vs. reddit ego graph classification task.

- GNN.ipynb : This file is the GNN (GCN) classifier for the moltbook vs. reddit ego graph classification task.

- MLP.ipynb : This file is the top K% MLP classifier for the populartiy classification task.

- utils.py : This file are just some general utilities.

- README.md : Just kidding...

If you have a HF API Key, you should pass it. It is significantly slower without, but I didn't want to hardcode one.

Of note, we wrote some of these codes seperately, so there is often redundant code. For example, the GNN code is completely self contained while the RFC uses data import/graph building functions in utils.py which are different than the ones in the GNN but largely have the same functionality. A future direction would be making this more efficient.

