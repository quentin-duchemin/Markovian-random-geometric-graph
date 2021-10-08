import os
import MRGG
from MRGG.Graph import Graph
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc
import time
import math

def link_probas(dimension, seed, list_n=None, nbnodes=10, envelope='', latitude='', path='final_data/', params={}):

	for n in list_n:
		np.random.seed(seed)
		G = Graph(n, dimension, sparsity = 1 ,sampling_type='markov',envelope=envelope,latitude=latitude, dic_params_functions=params)
		G.SCCHEi_with_R_search(listeR=[i for i in range(1,30)],listekappa=np.logspace(-4,0,1000), figure=False)
		eig, vec = np.linalg.eigh(G.A / (G.n * G.sparsity))
		G.latent_distance_estimation(G.sparsity * eig, vec, GRAM=True)

		# TRUE INTEGRAL
		nsamples = 40
		true_link_probas = np.zeros(nbnodes)
		GRAM = np.dot(G.V.T,G.V)
		for i in range(nbnodes):
			integral = 0
			discretize = np.linspace(-1,1,nsamples)
			densityY = np.array([(1-r_in**2)**((G.d-4)/2) for r_in in discretize])
			densityY /= np.sum(densityY)
			true = np.array(list(map(G.density_latitude, discretize)))
			dense_lat = true/sum(true)
			for k,r_nnp in enumerate(discretize):
				for l,r_in in enumerate(discretize):
					density = dense_lat[k] * densityY[l]
					t = GRAM[i,G.n-1] * r_nnp + np.sqrt(1-r_nnp**2)*np.sqrt(1-GRAM[i,G.n-1]**2)*r_in
					integral += density * G.compute_envelope(t)
			print(i)
			true_link_probas[i] = integral



		def find_closest(t,discretize):
				i = np.argmin(np.abs(discretize-t))
				return i
		
		# Link prediction   
		nsamples = 150
		link_probas_pred = np.zeros(nbnodes)
		discretize = np.linspace(-1,1,nsamples)
		esti = [G.estimation_envelope(xi) for xi in discretize]
		esti = list(map(lambda x:max(0,x),esti))
		esti = list(map(lambda x:min(1,x),esti))
		densityY = np.array([(1-r_in**2)**((G.d-4)/2) for r_in in discretize])
		densityY /= np.sum(densityY)
		for i in range(nbnodes):
			integral = 0
			true = np.array(list(map(G.latitude_estimator, discretize)))
			dense_lat = true/sum(true)
			for k,r_nnp in enumerate(discretize):
				for l,r_in in enumerate(discretize):
					density = dense_lat[k] * densityY[l]
					t = np.clip(G.gram[i,G.n-1],-1,1) * r_nnp + np.sqrt(1-r_nnp**2)*np.sqrt(1-np.clip(G.gram[i,G.n-1],-1,1)**2)*r_in
					integral += density * esti[find_closest(t,discretize)] 
			print(i)
			link_probas_pred[i] = integral

		# IID
		nsamples = 40
		link_probas_iid = np.zeros(nbnodes)
		for i in range(nbnodes):
			integral = 0
			discretize = np.linspace(-1,1,nsamples)
			dense_lat = np.array([(1-r_in**2)**((G.d-3)/2) for r_in in discretize])
			dense_lat /= np.sum(dense_lat)
			densityY = np.array([(1-r_in**2)**((G.d-4)/2) for r_in in discretize])
			densityY /= np.sum(densityY)
			for k,r_nnp in enumerate(discretize):
				for l,r_in in enumerate(discretize):
					density = densityY[l] * dense_lat[k]
					t = np.clip(G.gram[i,G.n-1],-1,1) * r_nnp + np.sqrt(1-r_nnp**2)*np.sqrt(1-np.clip(G.gram[i,G.n-1],-1,1)**2)*r_in
					integral += density * G.compute_envelope(t)
			print(i)
			link_probas_iid[i] = integral
	np.save(path+'link_probas_iid_n'+str(n)+'_'+str(dimension)+'_'+envelope+'_'+latitude+'_'+str(seed)+'.npy',link_probas_iid)
	np.save(path+'link_probas_pred_n'+str(n)+'_'+str(dimension)+'_'+envelope+'_'+latitude+'_'+str(seed)+'.npy',link_probas_pred)
	np.save(path+'true_link_probas_n'+str(n)+'_'+str(dimension)+'_'+envelope+'_'+latitude+'_'+str(seed)+'.npy',true_link_probas)