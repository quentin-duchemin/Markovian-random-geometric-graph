import numpy as np

from .MarkovKernel import Latitude

class Sampling(Latitude):
    """ Class aiming at sampling the next point on the Sphere """
    def __init__(self, nb_nodes, dimension, sampling_type, latitude = 'default', epsilon = 1e-2, rlim = None):
        Latitude.__init__(self, nb_nodes, dimension, sampling_type, latitude = latitude, rlim = rlim)
        self.epsilon = epsilon

    def uniform(self):
        """ Uniform sampling on the sphere """
        x = np.random.normal(0,1,self.d)
        return x / np.linalg.norm(x)

    def markov(self, prev_state):
        """ Markovian sampling on the sphere """ 
        z = np.random.normal(0,1,self.d)
        prev_ortho = z - np.dot(z,prev_state)*prev_state
        cos = self.compute_latitude()
        sin = np.sqrt(1-cos**2) 
        new_state = (cos*prev_state + sin*prev_ortho / np.linalg.norm(prev_ortho))
        return new_state
    
    def mixture(self, prev_state):
        """ Markovian sampling on the sphere """ 
        if np.random.rand() <= self.epsilon:
            return self.uniform()
        else:
            return self.markov(prev_state)
    
    def sample(self, prev_state, epsilon=1e-2):
        """ Sampling of the latent point on the sphere """
        if self.sampling_type == 'uniform':
            return self.uniform()
        elif self.sampling_type == 'markov':
            return self.markov(prev_state)
        elif self.sampling_type == 'mixture':
            return self.mixture(prev_state)
