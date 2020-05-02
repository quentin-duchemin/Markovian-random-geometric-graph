import numpy as np

from .MarkovKernel import Latitude

class Sampling(Latitude):
    """ Class aiming at sampling the next point of the Sphere """
    def __init__(self, nb_nodes, dimension, sampling_type, latitude = 'default'):
        Latitude.__init__(self, nb_nodes, dimension, sampling_type, latitude = latitude)

    def uniform(self):
        x = np.random.normal(0,1,self.d)
        return x / np.linalg.norm(x)

    def markov(self, prev_state):
        z = np.random.normal(0,1,self.d)
        prev_ortho = z - np.dot(z,prev_state)*prev_state
        cos = self.compute_latitude()
        sin = np.sqrt(1-cos**2) 
        new_state = (cos*prev_state + sin*prev_ortho / np.linalg.norm(prev_ortho))
        return new_state
    
    def sample(self, prev_state):
        if self.sampling_type == 'uniform':
            return self.uniform()
        elif self.sampling_type == 'markov':
            return self.markov(prev_state)
