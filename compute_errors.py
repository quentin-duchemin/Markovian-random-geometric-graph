import os
#os.chdir('MRGG')
import MRGG
from MRGG.Graph import Graph
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc
import time
import math

def compute_errors(dimension, seed, list_n=None, envelope='', latitude='', path='final_data/', params={}):
	if list_n is None:
		list_n = [200,500,1000,1500]
	nbite = 40
	L2errors_env = np.zeros((nbite,len(list_n)))
	L2errors_lat = np.zeros((nbite,len(list_n)))
	delta2errors_env = np.zeros((nbite,len(list_n)))
	delta2errors_lat = np.zeros((nbite,len(list_n)))
	for ite in range(nbite):
		for i, n in enumerate(list_n):
			np.random.seed(seed+100*ite)
			G = Graph(n, dimension, sampling_type='markov',envelope=envelope,latitude=latitude, sparsity = 1, dic_params_functions=params)
			G.SCCHEi_with_R_search(listeR=[i for i in range(1,20)], figure=False)
			L2errors_env[ite,i] = (G.L2_error_estimation_envelope())
			L2errors_lat[ite,i] = (G.L2_error_estimation_latitude())
			delta2errors_env[ite,i] = G.error_estimation_envelope()
			delta2errors_lat[ite,i] = G.error_estimation_latitude()
			if ite == 0:
				G.plot_estimation_envelope(savename=path+'figenv_n'+str(n)+'_'+str(dimension)+'_'+envelope+'_'+latitude+'_'+str(seed)+'.png', display=False)
				G.plot_densities_latitude(savename=path+'figlat_n'+str(n)+'_'+str(dimension)+'_'+envelope+'_'+latitude+'_'+str(seed)+'.png', display=False)
				G.plot_adjacency_eigs_vs_SCCHEi_clusters(G.adaptiveR, savename=path+'figeigs_n'+str(n)+'_'+str(dimension)+'_'+envelope+'_'+latitude+'_'+str(seed)+'.png', display=False)

	np.save(path+'L2errors_env_dim_'+str(dimension)+'_'+envelope+'_'+latitude+'_'+str(seed)+'.npy', np.array(L2errors_env))
	np.save(path+'L2errors_lat_dim_'+str(dimension)+'_'+envelope+'_'+latitude+'_'+str(seed)+'.npy', np.array(L2errors_lat))
	np.save(path+'delta2errors_env_dim_'+str(dimension)+'_'+envelope+'_'+latitude+'_'+str(seed)+'.npy', np.array(delta2errors_env))
	np.save(path+'delta2errors_lat_dim_'+str(dimension)+'_'+envelope+'_'+latitude+'_'+str(seed)+'.npy', np.array(delta2errors_lat))