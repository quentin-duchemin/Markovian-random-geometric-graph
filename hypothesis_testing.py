import os
import MRGG
from MRGG.Graph import Graph
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc
import time
import math


def hypothesis_testing(dimension, seed, list_n, envelope='', latitude='', path='final_data/', params={}):

	Nclasses = 70
	if list_n is None:
		list_n = [200,500,1000,1500]
	nbite = 40
	pvalue_lat = np.zeros((nbite,len(list_n)))
	pvalue_iid = np.zeros((nbite,len(list_n)))

	discretize = np.linspace(-1,1,1000)

	nbsamples = 1000
	for ite in range(nbite):
		for ind,n in enumerate(list_n):
			true_freq = nbsamples * np.ones(Nclasses) / Nclasses
			np.random.seed(seed+100*ite)
			Giid = Graph(n, dimension, sparsity = 1 ,sampling_type='uniform',envelope=envelope,latitude=latitude, dic_params_functions=params)
			probabilities = list(map(lambda x:Giid.latitude_estimator(x,(1/Giid.n)**(4/10)), discretize))
			probabilities /= np.sum(probabilities)
			IIDs = np.random.choice(discretize, nbsamples, p=probabilities)
			IIDshist, IIDsbins = np.histogram(IIDs, bins=Nclasses, range=(-1,1))
			fexp = true_freq[3:-3]
			nbexp = np.sum(fexp)
			iidh = IIDshist[3:-3]
			iidh = nbexp * iidh  / np.sum(iidh)
			chi2iid, piid = sc.stats.chisquare(iidh, f_exp=fexp)
			pvalue_iid[ite,ind] = piid
			np.random.seed(seed+100*ite)
			G = Graph(n, dimension, sparsity = 1 ,sampling_type='markov',envelope=envelope,latitude=latitude, dic_params_functions=params)
			probabilities = list(map(lambda x:G.latitude_estimator(x,(1/G.n)**(4/10)), discretize))
			probabilities /= np.sum(probabilities)
			latitudels = np.random.choice(discretize, nbsamples, p=probabilities)
			latitudehist, latitudebins = np.histogram(latitudels, bins=Nclasses, range=(-1,1))
			fexp = true_freq[3:-3]
			nbexp = np.sum(fexp)
			lat = latitudehist[3:-3]
			lat = nbexp * lat  / np.sum(lat)
			chi2lat, plat = sc.stats.chisquare(lat, f_exp=fexp)
			Tchi2 = sc.stats.chi2.ppf(0.95, df=Nclasses-1)
			pvalue_lat[ite,ind] = plat
			np.save(path+'pvalue_lat_dim_'+str(dimension)+'_'+envelope+'_'+latitude+'_'+str(seed)+'.npy',pvalue_lat)
			np.save(path+'pvalue_iid_dim_'+str(dimension)+'_'+envelope+'_'+latitude+'_'+str(seed)+'.npy',pvalue_iid)