
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import gegenbauer
import math
import pandas as pd

from MGRG.Graph import Graph


def study_error(latitude,enveloppe):
  d = 3
  results  = {'error_enveloppe':[],'error_latitude':[],'error_gram':[],'size':[]}
  listen = list(map(int,np.logspace(np.log10(20),np.log10(3000),30)))

  for n in listen:
    for _ in range(15):
      G = Graph(n, d, 'markov', latitude=latitude, enveloppe=enveloppe)
      G.HAC_R_study()
      results['error_enveloppe'].append(G.error_estimation_enveloppe())
      results['error_latitude'].append(G.error_estimation_latitude())
      results['error_gram'].append(G.error_gram_matrix())
      results['size'].append(n)
  df = pd.DataFrame(results)
  df = df.apply(np.log10)
  return df

LATITUDES = ['linear','default','beta','mixture']
ENVS = ['heaviside','prim1_heaviside','prim2_heaviside','sinus','polynomial']


for lat in LATITUDES:
  for env in ENVS:
    df = study_error(lat,env)
    df.to_pickle("./df-"+lat+"-"+env+".pkl")
    