> 📋Markovian Random Geometric Graph

# Markov Random Geometric Graph (MRGG):

A Growth Model for Temporal Dynamic Networks


| **Paper**                   | 
|:------------------------- |
| [![][paper-img]][paper-url] | 
| [![][arXiv-img]][arXiv-url] | 


[arXiv-img]: https://img.shields.io/badge/arXiv-2107.11000-blue.svg
[arXiv-url]: https://arxiv.org/abs/2006.07001

[paper-img]: https://img.shields.io/badge/doi-10.1002/mrm.29071-blue.svg
[paper-url]: https://projecteuclid.org/journals/electronic-journal-of-statistics/volume-16/issue-1/Markov-random-geometric-graph-MRGG--A-growth-model-for/10.1214/21-EJS1969.full


#### Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


#### How to cite this work?
If you use this package for your own work, please consider citing it with the following piece of Bibtex:


```bibtex
@article{10.1214/21-EJS1969,
author = {Quentin Duchemin and Yohann De Castro},
title = {{Markov random geometric graph, MRGG: A growth model for temporal dynamic networks}},
volume = {16},
journal = {Electronic Journal of Statistics},
number = {1},
publisher = {Institute of Mathematical Statistics and Bernoulli Society},
pages = {671 -- 699},
keywords = {link prediction, Markov chains, Non-parametric estimation, Random geometric graph, spectral methods},
year = {2022},
doi = {10.1214/21-EJS1969},
URL = {https://doi.org/10.1214/21-EJS1969}
}
```

## Brief description on the way to run an basic simulation

#### In the Notebook 'Experiments.ipynb', we can find the code to reproduce all our results provided in the paper.

#### How to build a graph from simulated data

```train
G = Graph(n, d, sampling_type = 'markov', envelope = 'heaviside', latitude = 'mixture')
```

> 📋The previous command allows you to create a graph of size n with a Markovian dynamic on the latent points on the Sphere of dimension d. The envelope function used in that case if the Heaviside and the latitude function is a mixture of beta distribution. You can define your own envelope and latitude functions by modyfying the files `Latitude.py` and `Envelope.py`.



#### How to build a graph from real data

```train
G = Graph(n, d, adjacency_matrix = A, sparsity = sparsity)
```

> 📋The previous command allows you to define a Graph instance with your own adjacency matrix. The `sparsity` parameter should define the average degree of a node of the graph divided by the size of the graph.



#### Launch the algorithm SCCHEi

The following command launches the algorithm SCCHEi with a search of the best resolution level R. 

```train
G.SCCHEi_with_R_search()
```


#### Visualize results

Visualize the true and the estimated envelope function (resp. latitude function). 

```train
G.plot_estimation_envelope()
G.plot_densities_latitude()
```

