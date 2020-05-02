import scipy as sc
import numpy as np
import matplotlib.pyplot as plt

from ..Parameters import Parameters

class EstimatorEnvelope(Parameters):
    """ Class estimating the enveloppe function using the strategy described in  arXiv:1708.02107 """
    def __init__(self, nb_nodes, dimension, sampling_type):
        Parameters.__init__(self, nb_nodes, dimension, sampling_type)

    def compute_dimensions_sphere(self, R=None):
        """ Computes the dimensions of the spaces of real spherical harmonics for degrees up to R """
        self.dimensions = []
        l = 0
        sum_dim = 0
        lim_sum = np.float('inf')
        if R is None:
          lim_sum = self.n
          R = self.n
        while (sum_dim < lim_sum and l<R+1):
          if l==0:
            self.dimensions.append(1)
          elif l==1:
            self.dimensions.append(self.d)
          else:
            next_d = sc.special.binom(l+self.d-1,l) - sc.special.binom(l+self.d-3,l-2)
            sum_dim += next_d
            if sum_dim  < lim_sum:
              self.dimensions.append( int(next_d) )
          l += 1


    def tree_HAC(self, eigens, indices):
        from sklearn.cluster import AgglomerativeClustering
        clustering = AgglomerativeClustering(linkage='complete').fit(eigens.reshape(-1,1))
        children = clustering.children_
        children = np.array(children)
        tree = [ {} for i in range(children.shape[0]+1)]
        tree[0] = {ind:[indices[ind]] for ind in range(len(eigens))}
        treesizes = [ {} for i in range(children.shape[0]+1)]
        treesizes[0] = {ind: 1 for ind in range(len(eigens))}

        nsamples = len(eigens)
        Max = np.max(indices)
        saveeig2name = [[i for i in range(Max+1)] for j in range(children.shape[0]+1)]
        for i,ind in enumerate(indices):
          saveeig2name[0][ind] = i

        for i in range(len(children)):
          name1 = children[i,0]
          name2 = children[i,1]
          tree[i+1] = {name:clust.copy() for name,clust in tree[i].items()}
          tree[i+1][nsamples+i] = tree[i][name1].copy() + tree[i][name2].copy()
          saveeig2name[i+1] =  np.copy(saveeig2name[i])
          for ei in tree[i+1][nsamples+i]:
            saveeig2name[i+1][ei] = nsamples+i
          del tree[i+1][name2]
          del tree[i+1][name1]
          
          treesizes[i+1] = {name:len(group) for name,group in tree[i+1].items()}

        return tree, treesizes, saveeig2name


    def roam_tree(self, final_clusters, dims, tree, treesizes, saveeig2name):
        missing_dims = []
        L = len(tree)
        for d in dims:
          depth = len(tree)-1
          found = False
          while (depth >= 0 and not(found)):
            if d in treesizes[depth].values():
              dic = tree[depth].copy()
              items = list(dic.items())
              N = len(items)
              indname = 0
              while indname<N and not(found):
                  name,clust = items[indname][0],items[indname][1].copy()
                  indname += 1
                  if len(clust)==d:
                    final_clusters[1].append(clust)
                    found = True
                    for i in range(L):
                      for ei in clust:
                        if treesizes[i][saveeig2name[i][ei]]==1:
                          del tree[i][saveeig2name[i][ei]]
                          del treesizes[i][saveeig2name[i][ei]]
                        else: 
                          tree[i][saveeig2name[i][ei]].remove(ei)
                          treesizes[i][saveeig2name[i][ei]] -= 1
                    
            depth -= 1
          if not(found):
            missing_dims.append(d)
        return final_clusters, missing_dims, tree, treesizes, saveeig2name

    def hierarchical_clustering(self, eigens):
        """ Performs a hierarchical agglomerative clustering with predefined sizes for the clusters """
        if np.sum(self.dimensions)>self.n:
          cumsum = np.cumsum(self.dimensions)
          R = 0
          while cumsum[R+1]<self.n:
            R += 1
          self.dimensions = self.dimensions[:R+1]
        tree, treesizes, saveeig2name = self.tree_HAC(eigens,[i for i in range(len(eigens))])
        
        dims = self.dimensions
        missing_dims = []
        final_clusters = {0:[],1:[]}
        nsamples = self.n

        while dims != []:
          final_clusters, missing_dims, tree, treesizes, saveeig2name = self.roam_tree(final_clusters, dims, tree, treesizes, saveeig2name)
          dims = []
          while missing_dims != []:
            d = missing_dims.pop(0)
            depth = len(tree)-1
            optdepth = depth
            optname = -1
            gap = np.float('inf')
            while (depth >= 0):
              for name,size in treesizes[depth].items():
                if abs(size-d)<gap and d<=size:
                  gap = size-d
                  optname = name
                  optdepth = depth
              depth -= 1
            if optname != -1 and gap==0:
              cluster = tree[optdepth][optname].copy()
              final_clusters[1].append(cluster)
              for i in range(len(tree)):
                for ei in cluster:
                  if treesizes[i][saveeig2name[i][ei]]==1:
                    del tree[i][saveeig2name[i][ei]]
                    del treesizes[i][saveeig2name[i][ei]]
                  else:
                    tree[i][saveeig2name[i][ei]].remove(ei)
                    treesizes[i][saveeig2name[i][ei]] -= 1

            elif optname != -1:
              cluster = tree[optdepth][optname].copy()
              clust, remain = self.split(cluster,d)
              final_clusters[1].append(clust)
              for i in range(len(tree)):
                for ei in clust:
                  if treesizes[i][saveeig2name[i][ei]]==1:
                    del tree[i][saveeig2name[i][ei]]
                    del treesizes[i][saveeig2name[i][ei]]
                  else:
                    tree[i][saveeig2name[i][ei]].remove(ei)
                    treesizes[i][saveeig2name[i][ei]] -= 1
 
            else:
                remaining_eigs_indices = [tree[0][name][0] for name in tree[0].keys()]
                tree, treesizes, saveeig2name = self.tree_HAC(eigens[remaining_eigs_indices],remaining_eigs_indices)
                dims = [d]+missing_dims
                missing_dims = []
                nsamples = len(remaining_eigs_indices)            

        for ls in tree[0].values():
          final_clusters[0].append(ls[0])

        return final_clusters

    def split(self, clust, d):
        clust = np.array(clust)
        order = (np.argsort(clust)[::-1])
        return list(clust[order[:d]]), list(clust[order[d:]])

    def HAC_basic(self):
        eig, v = np.linalg.eig(self.A / self.n)
        eig = np.real(eig)
        self.compute_dimensions_sphere()  
        bestMSE = np.float('inf')
        bestR = 0
        self.spectrumenv = {}
        clustering = self.hierarchical_clustering(eig)
        size2clust = {len(clust):clust for clust in clustering[1]}
        self.spectrumenv = [ np.real(np.mean(eig[size2clust[int(d)]])) for d in self.dimensions ]

    def HAC_R_study(self, figure=False):
        eig, v = np.linalg.eig(self.A / self.n)
        eig = np.real(eig)
        idx = eig.argsort()[::-1]   
        eig = eig[idx]
        self.compute_dimensions_sphere()
        L = len(self.dimensions)
        bestMSE = np.float('inf')
        bestR = 0
        listeR = np.linspace(0,L,6)[1:]
        listeMSE = []
        for R in listeR:
          spectrumenv = {}
          self.compute_dimensions_sphere(R=R)
          clustering = self.hierarchical_clustering(eig[:sum(self.dimensions)])
          size2clust = {len(clust):clust for clust in clustering[1]}
          spectrumenv = [ np.real(np.mean(eig[size2clust[int(d)]])) for d in self.dimensions ]
          MSE = 0
          for i,clust in size2clust.items():
            if i<=R:
              MSE += np.sum((eig[clust]-np.mean(eig[clust]))**2)
            else:
              MSE += np.sum((eig[clust])**2)
          MSE += np.sum((eig[clustering[0]])**2)
          MSE += np.sum((eig[sum(self.dimensions):])**2)
          listeMSE.append(MSE)
          if MSE < bestMSE:
            bestMSE = MSE
            bestR = R
            self.spectrumenv = spectrumenv
        if figure:
          plt.scatter(listeR, listeMSE)
        
            
    def estimation_enveloppe(self, t):
        return self.function_gegenbauer(t, self.spectrumenv)
    
    # Test
    def plot_estimation_enveloppe(self):
        x = np.linspace(-1,1,100)
        self.esti = [self.estimation_enveloppe(xi) for xi in x]
        self.true = [self.compute_enveloppe(xi) for xi in x]
        plt.plot(x, self.esti, label='Estimation')
        plt.plot(x, self.true, label='True enveloppe')
        plt.legend()
        plt.show()

    def check_HAC(self, Rmax):
        eig, v = np.linalg.eig(self.A / self.n)
        eig = np.real(eig)
        self.compute_dimensions_sphere()  
        clusters = self.hierarchical_clustering(eig, Rmax)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        count = 0
        for i in range(len(clusters[1])):
          ls = list(np.real(eig[clusters[1][i]]))
          ax1.scatter(clusters[1][i], ls)
          count += len(ls)
        ls = list(np.real(eig[clusters[0]]))
        ax1.scatter(clusters[0], ls, label='eigenvalues set to 0')
        plt.legend()
        plt.show()

    def check_HAC_bis(self, Rmax, nbmax_eigs=100):
        eig, v = np.linalg.eig(self.A / self.n)
        eig = np.real(eig)
        self.compute_dimensions_sphere(R=Rmax)  
        clusters = self.hierarchical_clustering(eig)
        fig = plt.figure(figsize=(10,4))
        ax1 = fig.add_subplot(111)
        ax1.scatter([i for i in range(nbmax_eigs)], eig[:nbmax_eigs],  c='blue', marker = 'o',label='True eigenvalues')
        indices = []
        means = []
        for i in range(len(clusters[1])):
          mean = np.mean(eig[clusters[1][i]])
          for ei in clusters[1][i]:
            if ei<nbmax_eigs:
              means.append(mean)
              indices.append(ei)
        ax1.scatter(indices, means, c='red', marker = '+', label='Clusters')
        inds = list(filter(lambda x: x<nbmax_eigs,clusters[0]))
        ls = list(np.real(eig[inds]))
        ax1.scatter(inds, ls, c='black',marker='*', label='Eigenvalues set to 0')
        plt.legend()
        plt.show()

    def check_HAC_line(self, Rmax, thresholdmin=0, thresholdmax = 100, see_eigs_set_to_0 = False):
        eig, v = np.linalg.eig(self.A / self.n)
        eig = np.real(eig)
        self.compute_dimensions_sphere(R=Rmax)  
        idx = eig.argsort()[::-1]   
        eig = eig[idx]
        # maxindUSVT = 0
        # while maxindUSVT<self.n and eig[maxindUSVT] > 1/np.sqrt(n):
        #   maxindUSVT +=1
        # print(maxindUSVT,sum(self.dimensions))
        # maxindUSVT = max(maxindUSVT,sum(self.dimensions))
        maxindUSVT = min(sum(self.dimensions),self.n-1)
        clusters = self.hierarchical_clustering(eig[:maxindUSVT])
        fig = plt.figure(figsize=(10,4))
        ax1 = fig.add_subplot(111)
        for i in range(len(clusters[1])):
          eigens = list(filter(lambda x: (abs(x) < thresholdmax and abs(x)>=thresholdmin) , eig[clusters[1][i]]))
          size = len(eigens)
          ax1.scatter(np.abs(eigens) ,np.zeros(size), marker = '+')
        if see_eigs_set_to_0:
          eigens = list(filter(lambda x: (abs(x) < thresholdmax and abs(x)>=thresholdmin), eig[clusters[0]]))
          size = len(eigens)
          ax1.scatter(np.abs(eigens), np.ones(size),marker='*', label='Eigenvalues set to 0')
          plt.legend()
        ax1.set_xscale('log')
        ax1.set_xlabel('$\log(|eigenvalue|)$')
        ax1.minorticks_on()
        plt.show()