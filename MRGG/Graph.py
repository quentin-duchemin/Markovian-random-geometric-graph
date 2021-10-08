import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

from .Sampling import Sampling
from .Envelope import Envelope
from .Results import Results


class Graph(Sampling, Envelope, Results):
    """ Main class building the graph """
    def __init__(self, nb_nodes, dimension, sampling_type = 'uniform', latitude = 'default', envelope = 'heaviside', sparsity = 1, epsilon=1e-2, adjacency_matrix = None, nbeigvals = None, eig = None, vec = None, rlim = 0.9, dic_params_functions={}):
        Sampling.__init__(self, nb_nodes, dimension, sampling_type, latitude = latitude, epsilon=epsilon, rlim=rlim)
        Envelope.__init__(self, nb_nodes, dimension, sampling_type, envelope = envelope)
        Results.__init__(self, nb_nodes, dimension, sampling_type)
        self.sparsity = sparsity
        self.dicparas = dic_params_functions
        if eig is None:
            if adjacency_matrix is None:
                self.adjacency()
            else:
                self.A = adjacency_matrix

            if nbeigvals is None:
                eig, vec = np.linalg.eigh(self.A / (self.n * self.sparsity))
            else:
                row = []
                col = []
                [row,col] = np.where(self.A!=0)
                Asparse = sc.sparse.csr_matrix((np.ones(len(row)) / (self.n * self.sparsity), (row, col)), shape=(self.n,self.n))
                eig, vec = sc.sparse.linalg.eigsh(Asparse, k=nbeigvals)
        self.latent_distance_estimation(self.sparsity * eig, vec)
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
        argument = np.dot(V.T,V).reshape(-1)
        self.Theta = np.array(list(map(lambda x: self.compute_envelope(x),argument)))
        self.Theta = self.sparsity * self.Theta.reshape(self.n,self.n)
        for i in range(self.n):
            self.Theta[i,i] = 0
        uniform = np.random.rand(self.n, self.n)
        uniform = np.triu(uniform)
        uniform = (uniform + uniform.T)
        self.A = 1*(uniform<self.Theta)

    def show_graph_with_labels(self, adjacency_matrix):
        rows, cols = np.where(adjacency_matrix == 1)
        import networkx as nx
        edges = zip(rows.tolist(), cols.tolist())
        gr = nx.Graph()
        gr.add_edges_from(edges)
        nx.draw(gr, node_size=500, with_labels=True)
        plt.show()