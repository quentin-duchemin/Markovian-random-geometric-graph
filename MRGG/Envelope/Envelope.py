import numpy as np
from scipy.special import gegenbauer
from .EstimatorEnvelope import EstimatorEnvelope

class Envelope(EstimatorEnvelope):
    """ Class containing all the work related to the envelope function """
    def __init__(self, nb_nodes, dimension, sampling_type, envelope = 'default'):
        EstimatorEnvelope.__init__(self, nb_nodes, dimension, sampling_type)
        self.envelope = envelope

    def function_gegenbauer(self, t, spectrum):
        """ Returns the value at t of the function with a decomposition 'spectrum' in the gegenbauer basis """
        res = 0
        for i in range(len(spectrum)):
            coeffi = (2*i+self.d-2)/(self.d-2)
            Gegen = gegenbauer(i, (self.d-2)/2)
            res += spectrum[i] * coeffi * Gegen(t)
        return res  

    def indicator_envelope(self, t):
        if t>=0.3:
          return 1
        else:
          return 0

    def linear_envelope(self, t):
        return np.abs((1+t)/2)

    def heaviside(self, t):
        if t>=0:
          return 1
        else:
          return 0

    def prim1_heaviside(self, t):
        if t>=0:
          return t
        else:
          return 0

    def prim2_heaviside(self, t):
        if t>=0:
          return t**2/2
        else:
          return 0

    def prim3_heaviside(self, t):
        if t>=0:
          return t**3/6
        else:
          return 0

    def sinus_envelope(self, t):
        return 0.5 + 0.5 *np.sin(np.pi*t/2)

    def compute_envelope(self, t):
        if self.envelope == 'indicator':
            return self.indicator_envelope(t)
        elif self.envelope == 'linear':
            return self.linear_envelope(t)
        elif self.envelope == 'heaviside':
            return self.heaviside(t)
        elif self.envelope == 'prim1_heaviside':
            return self.prim1_heaviside(t)
        elif self.envelope == 'prim2_heaviside':
            return self.prim2_heaviside(t)
        elif self.envelope == 'prim3_heaviside':
            return self.prim3_heaviside(t)
        elif self.envelope == 'sinus':
            return self.sinus_envelope(t)
        elif self.envelope == 'polynomial':
            return (1/3 + (35*t**4-30*t**2+ 3) / 12)