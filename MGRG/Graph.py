import numpy as np
import matplotlib.pyplot as plt

from .Sampling import Sampling
from .Envelope import Envelope
from .Results import Results


class Graph(Sampling, Envelope, Results):
    """ Main class building the graph """
    def __init__(self, nb_nodes, dimension, sampling_type = 'iid', latitude = 'default', enveloppe = 'default', adjacency_matrix = None):
        Sampling.__init__(self, nb_nodes, dimension, sampling_type, latitude = latitude)
        Envelope.__init__(self, nb_nodes, dimension, sampling_type, enveloppe = enveloppe)
        Results.__init__(self, nb_nodes, dimension, sampling_type)
        if adjacency_matrix is None:
            self.adjacency()
        else:
            self.A = adjacency_matrix
        self.latent_distance_estimation()

    def adjacency(self):
        V = np.zeros((self.d,self.n))
        V[:,0] = self.uniform()
        for i in range(1,self.n):
            V[:,i] = self.sample(V[:,i-1])
        self.V = V
        self.Theta = np.array([[self.compute_enveloppe(np.dot(V[:,i],V[:,j])) for i in range(self.n)] for j in range(self.n)])
        for i in range(self.n):
            self.Theta[i,i] = 0
        uniform = np.random.rand(self.n, self.n)
        self.A = np.array([[uniform[i,j]<self.Theta[i,j] for i in range(self.n)] for j in range(self.n)])
    
    def show_graph_with_labels(self, adjacency_matrix):
        rows, cols = np.where(adjacency_matrix == 1)
        import networkx as nx
        edges = zip(rows.tolist(), cols.tolist())
        gr = nx.Graph()
        gr.add_edges_from(edges)
        nx.draw(gr, node_size=500, with_labels=True)
        plt.show()

    def eigen_values(self):
        eig, v = np.linalg.eig(self.A / self.n)
        eig = np.sort(eig)[::-1]
        plt.scatter([i for i in range(self.n)],eig)