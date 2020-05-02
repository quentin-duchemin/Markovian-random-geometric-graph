import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.integrate as integrate

from .Parameters import Parameters

class Results(Parameters):
    """ Class estimating the enveloppe function using the strategy described in  arXiv:1708.02107 """
    def __init__(self, nb_nodes, dimension, sampling_type):
        Parameters.__init__(self, nb_nodes, dimension, sampling_type)
  
    def compute_eigenvalues(self, mode='enveloppe'):
        self.compute_dimensions_sphere(R=40)
        eigenvalues = []
        beta = (self.d-2)/2
        bd = math.gamma(self.d/2) / (math.gamma(1/2) * math.gamma(self.d/2 - 1/2))
        for l in range(min(40,len(self.dimensions))):
          Gegen = gegenbauer(l, beta)
          if mode == 'enveloppe':
            integral = integrate.quad(lambda x:  self.compute_enveloppe(x)*Gegen(x)*(1-x**2)**(beta-1/2), -1, 1)
          elif mode == 'latitude':
            integral = integrate.quad(lambda x:  self.density_latitude(x)*Gegen(x)*(1-x**2)**(beta-1/2), -1, 1)
          else:
            integral = integrate.quad(lambda x:  self.latitude_estimator(x)*Gegen(x)*(1-x**2)**(beta-1/2), -1, 1)            
          cl = (2*l+self.d-2)/(self.d-2)
          eigenvalues.append(cl*bd*integral[0]/self.dimensions[l])
        return eigenvalues

    def exact_delta2_metric(self, esti_spec, true_spec):
        import itertools
        R = len(esti_spec)
        permus = list(itertools.permutations([i for i in range(R)]))
        best_error = np.float('inf')
        for permu in permus:
          error = 0
          for k in range(R):
            error += self.dimensions[k]*(esti_spec[k]-true_spec[permu[k]])**2
          if error < best_error:
            best_error = error
        return np.sqrt(best_error)

    def delta2_metric(self, esti_spec, true_spec):
        print(len(esti_spec),len(true_spec))
        import itertools
        r = len(esti_spec)
        R = len(true_spec)
        for i in range(r,R):
          esti_spec.append(0)
        error = 0
        for i in range(R):
          error += self.dimensions[i]*(esti_spec[i]-true_spec[i])**2
        return error

    def error_estimation_enveloppe(self):
        eigenvalues = self.compute_eigenvalues()
        size = min(len(eigenvalues),len(self.spectrumenv))
        return self.delta2_metric(self.spectrumenv[:size],eigenvalues)

    def error_estimation_latitude(self):
        true_eigenvalues = self.compute_eigenvalues(mode='latitude')
        esti_eigenvalues = self.compute_eigenvalues(mode='estimation_latitude')
        return self.delta2_metric(esti_eigenvalues, true_eigenvalues)


    def plot_comparison_eig_envelope(self):
        plt.scatter(eigenvalues[:size],[0 for i in range(size)],label='True Envelope')
        plt.scatter(self.spectrumenv[:size],[1 for i in range(size)],label='Estimated Envelope')
        plt.title('Eigenvalues Envelope')
        plt.legend()
        plt.show()