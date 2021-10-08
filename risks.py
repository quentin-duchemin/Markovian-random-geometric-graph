import os
import MRGG
from MRGG.Graph import Graph
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc
import time
import math

def risks(dimension, seed, list_n=None, envelope='', latitude='', path='final_data/', params={}):
	risks = np.zeros((len(list_n),100,3))
	import math
	if list_n is None:
		list_n = [100*i for i in range(10,16)]
	NBRISK = 70
	for indi,n in enumerate(list_n):
		print(n)
		nsamples = 40
		np.random.seed(seed)
		G = Graph(n, dimension, sparsity = 1 ,sampling_type='markov',envelope=envelope,latitude=latitude, dic_params_functions=params)
		G.SCCHEi_with_R_search(listeR=[i for i in range(1,30)],listekappa=np.logspace(-4,0,1000), figure=False)
		eig, vec = np.linalg.eigh(G.A / (G.n * G.sparsity))
		G.latent_distance_estimation(G.sparsity * eig, vec, GRAM=True)
		link_probas = np.zeros(n)
		link_probas_esti = np.zeros(n)
		for i in range(NBRISK):
			integral = 0
			discretize = np.linspace(-1,1,nsamples)
			true = np.array(list(map(G.density_latitude, discretize)))
			dense_lat = true/sum(true)
			densityY = np.array([(1-r_in**2)**((G.d-4)/2) for r_in in discretize])
			densityY /= np.sum(densityY)
			for k,r_nnp in enumerate(discretize):
				for l,r_in in enumerate(discretize):
					density = dense_lat[k] * densityY[l]
					t = np.clip(G.gram[i,G.n-1],-1,1) * r_nnp + np.sqrt(1-r_nnp**2)*np.sqrt(1-np.clip(G.gram[i,G.n-1],-1,1)**2)*r_in
					m = density * G.compute_envelope(t) 
					if not(math.isnan(m)):
						integral += m
					else:
						print('aie', np.sqrt(1-r_nnp**2)*np.sqrt(1-G.gram[i,G.n-1]**2))
			link_probas[i] = integral


			def find_closest(t,discretize):
				argmin = np.argmin(np.abs(discretize-t))
				return argmin

			integral = 0
			nsamples = 150
			discretize = np.linspace(-1,1,nsamples)
			esti = [G.estimation_envelope(xi) for xi in discretize]
			esti = list(map(lambda x:max(0,x),esti))
			esti = list(map(lambda x:min(1,x),esti))
			lat = np.array(list(map(G.latitude_estimator, discretize)))
			dense_lat = lat/sum(lat)
			densityY = np.array([(1-r_in**2)**((G.d-4)/2) for r_in in discretize])
			densityY /= np.sum(densityY)
			for k,r_nnp in enumerate(discretize):
				for l,r_in in enumerate(discretize):
					density = dense_lat[k] * densityY[l]
					t = np.clip(G.gram[i,G.n-1],-1,1) * r_nnp + np.sqrt(1-r_nnp**2)*np.sqrt(1-np.clip(G.gram[i,G.n-1],-1,1)**2)*r_in
					m = density * esti[find_closest(t,discretize)]
					if not(math.isnan(m)):
						integral += m
					else:
						print('aie', np.sqrt(1-r_nnp**2)*np.sqrt(1-G.gram[i,G.n-1]**2))
			link_probas_esti[i] = integral

		p = np.sum(G.A) / n**2
		link_probas = link_probas[:NBRISK]
		link_probas_esti = link_probas_esti[:NBRISK]
		for k in range(100):
			Xnext = G.sample(G.V[:,G.n-1])
			envs = np.array(list(map(lambda x: G.compute_envelope(x), G.V.T @ Xnext)))
			envs = envs.reshape(-1)
			uniform = np.random.rand(G.n)
			A = 1*(uniform<envs)

			A=A[:NBRISK]

			risks[indi,k,0] = np.sum((np.ones(NBRISK)-link_probas) * A + link_probas * (np.ones(NBRISK)-A))
			risks[indi,k,1] = np.sum((1-p) * A + p * (np.ones(NBRISK)-A))
			risks[indi,k,2] = np.sum((np.ones(NBRISK)-link_probas_esti) * A + link_probas_esti * (np.ones(NBRISK)-A))

	np.save(path+'RISKS_'+str(dimension)+'_'+envelope+'_'+latitude+'_'+str(seed)+'.npy',risks)