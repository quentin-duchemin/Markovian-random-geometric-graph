import numpy as np
from scipy.special import gegenbauer
from .EstimatorEnvelope import EstimatorEnvelope

class Envelope(EstimatorEnvelope):
    """ Class containing all the work related to the enveloppe function """
    def __init__(self, nb_nodes, dimension, sampling_type, enveloppe = 'default'):
        EstimatorEnvelope.__init__(self, nb_nodes, dimension, sampling_type)
        self.enveloppe = enveloppe

    def function_gegenbauer(self, t, spectrum):
        """ Returns the value at t of the function with a decomposition 'spectrum' in the gegenbauer basis """
        res = 0
        if self.d>2:
            for i in range(len(spectrum)):
                coeffi = (2*i+self.d-2)/(self.d-2)
                Gegen = gegenbauer(i, (self.d-2)/2)
                res += spectrum[i] * coeffi * Gegen(t)
        elif self.d==2:
            for i in range(len(spectrum)):
                res += spectrum[i] * np.cos(i*np.arccos(t))
        return res  

    def default_enveloppe(self, t):
        spectrum =  [1,5,11,4,-7,20] #  [4,2,1,0.5] 
        return self.function_gegenbauer(t, spectrum)

    def indicator_enveloppe(self, t):
        if t>=0.3:
          return 1
        else:
          return 0

    def linear_enveloppe(self, t):
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

    def sinus_enveloppe(self, t):
        return 0.5 + 0.5 *np.sin(np.pi*t/2)

    def compute_enveloppe(self, t):
        if self.enveloppe == 'default':
            return self.default_enveloppe(t)
        elif self.enveloppe == 'indicator':
            return self.indicator_enveloppe(t)
        elif self.enveloppe == 'linear':
            return self.linear_enveloppe(t)
        elif self.enveloppe == 'heaviside':
            return self.heaviside(t)
        elif self.enveloppe == 'prim1_heaviside':
            return self.prim1_heaviside(t)
        elif self.enveloppe == 'prim2_heaviside':
            return self.prim2_heaviside(t)
        elif self.enveloppe == 'prim3_heaviside':
            return self.prim3_heaviside(t)
        elif self.enveloppe == 'sinus':
            return self.sinus_enveloppe(t)
        elif self.enveloppe == 'polynomial':
            return (1/3 + (35*t**4-30*t**2+ 3) / 12)