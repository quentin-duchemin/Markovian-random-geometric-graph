import numpy as np
from scipy import stats

from .EstimatorLatitude import EstimatorLatitude

class Latitude(EstimatorLatitude):
    """ Class containing all the work related to the latitude function """
    def __init__(self, nb_nodes, dimension, sampling_type, latitude = 'default'):
        EstimatorLatitude.__init__(self, nb_nodes, dimension, sampling_type)
        self.latitude = latitude
        if self.latitude == 'beta':
          self.alpha = 5
          self.beta = 1

    def defaut_latitude(self):
        return np.random.uniform(0,np.pi)

    def beta_latitude(self):
        return np.pi * np.random.beta(self.alpha, self.beta)

    def log_normal(self):
        return np.pi * np.random.lognormal()

    def compute_arccos_latitude(self):
        if self.latitude == 'uniform':
          return self.defaut_latitude()
        elif self.latitude == 'beta':
          return self.beta_latitude()
        elif self.latitude == 'log-normal':
          return self.log_normal()

    def compute_latitude(self):
        if self.latitude == 'default':
            return np.random.uniform(-1,1)
        elif self.latitude == 'linear':
            h = 2/3
            u = np.random.uniform(0,1)
            return -3 + np.sqrt(9*h*h-h*(5*h-8*u))/h
        else:
            return np.cos(self.compute_arccos_latitude())

    def density_latitude(self, cosangle):
        if self.latitude == 'uniform':
          return (1 /(np.pi* np.sqrt(1-cosangle**2)))
        elif self.latitude == 'beta':
          return (stats.beta.pdf(np.arccos(cosangle)/np.pi, self.alpha, self.beta)*(1/(np.pi*np.sqrt(1-cosangle**2))))
        elif self.latitude == 'log-normal':
          return (stats.lognorm.pdf(np.arccos(cosangle)/np.pi, 1)*(1/(np.pi*np.sqrt(1-cosangle**2))))
        elif self.latitude == 'default':
          return (1/2)
        elif self.latitude == 'linear':
          h = 2/3
          return (h/4)*(cosangle+1)+h/2