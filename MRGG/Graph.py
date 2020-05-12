import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

from .Sampling import Sampling
from .Envelope import Envelope
from .Results import Results


class Graph(Sampling, Envelope, Results):
    """ Main class building the graph """
    def __init__(self, nb_nodes, dimension, sampling_type = 'iid', latitude = 'default', enveloppe = 'default', sparsity = 1, adjacency_matrix = None, nbeigvals = None):
        Sampling.__init__(self, nb_nodes, dimension, sampling_type, latitude = latitude)
        Envelope.__init__(self, nb_nodes, dimension, sampling_type, enveloppe = enveloppe)
        Results.__init__(self, nb_nodes, dimension, sampling_type)
        self.sparsity = sparsity
        if adjacency_matrix is None:
            self.adjacency()
        else:
            self.A = adjacency_matrix
            self.latent_distance_estimation()
        if nbeigvals is None:
            eig, vec = np.linalg.eig(self.A / (self.n * self.sparsity))
        else:
            eig, vec = sc.linalg.eigh(self.A / (self.n * self.sparsity), eigvals=(eigvals,self.n-1))

        self.latent_distance_estimation(eig, vec)
        eig = np.real(eig)
        dec_order = np.argsort(np.abs(eig))[::-1]
        dec_eigs = eig[dec_order]
        self.dec_eigs = dec_eigs 
    
    def adjacency(self):
        V = np.zeros((self.d,self.n))
        V[:,0] = self.uniform()
        for i in range(1,self.n):
            V[:,i] = self.sample(V[:,i-1])
        self.V = V
        self.Theta = np.array(list(map(lambda x: self.compute_enveloppe(x),np.dot(V.T,V).reshape(-1))))
        self.Theta = self.sparsity * self.Theta.reshape(self.n,self.n)
        for i in range(self.n):
            self.Theta[i,i] = 0
        uniform = np.random.rand(self.n, self.n)
        self.A = 1*(uniform<self.Theta)

    def show_graph_with_labels(self, adjacency_matrix):
        rows, cols = np.where(adjacency_matrix == 1)
        import networkx as nx
        edges = zip(rows.tolist(), cols.tolist())
        gr = nx.Graph()
        gr.add_edges_from(edges)
        nx.draw(gr, node_size=500, with_labels=True)
        plt.show()