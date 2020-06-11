> 📋Markovian Random Geometric Graph

# Markov Random Geometric Graph (MRGG):

A Growth Model for Temporal Dynamic Networks


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## In the Notebook 'Experiments.ipynb', we can find the code to reproduce all our results provided in the paper.

## How to build a graph from simulated data

```train
G = Graph(n, d, sampling_type = 'markov', envelope = 'heaviside', latitude = 'mixture')
```

> 📋The previous command allows you to create a graph of size n with a Markovian dynamic on the latent points on the Sphere of dimension d. The envelope function used in that case if the Heaviside and the latitude function is a mixture of beta distribution. You can define your own envelope and latitude functions by modyfying the files `Latitude.py` and `Envelope.py`.



## How to build a graph from real data

```train
G = Graph(n, d, adjacency_matrix = A, sparsity = sparsity)
```

> 📋The previous command allows you to define a Graph instance with your own adjacency matrix. The `sparsity` parameter should define the average degree of a node of the graph divided by the size of the graph.



## Launch the algorithm SCCHEi

The following command launches the algorithm SCCHEi with a search of the best resolution level R. 

```train
G.SCCHEi_with_R_search()
```


## Visualize results

Visualize the true and the estimated envelope function (resp. latitude function). 

```train
G.plot_estimation_envelope()
G.plot_densities_latitude()
```

