import scipy as sc
import numpy as np
import matplotlib.pyplot as plt

from ..Parameters import Parameters

class EstimatorEnvelope(Parameters):
    """ Class estimating the envelope function using the strategy described in  arXiv:1708.02107 """
    def __init__(self, nb_nodes, dimension, sampling_type):
        Parameters.__init__(self, nb_nodes, dimension, sampling_type)

    def compute_dimensions_sphere(self, R=None):
        """ Computes the dimensions of the spaces of real spherical harmonics for degrees up to R """
        self.dimensions = []
        l = 0
        sum_dim = 0
        lim_sum = np.float('inf')
        if R is None:
          lim_sum = len(self.dec_eigs)
          R = len(self.dec_eigs)
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
        """ Saves the tree built with a Hierarchical Clustering of the eiganvalues in "eigens" """
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
        """ Searches for clusters with sizes equal to dimension of spherical harmonic spaces 
        as close as possible to the root of the tree built by tree_HAC """
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

    def SCCHEi(self, R):
        """ SCCHEi algorithm """
        self.compute_dimensions_sphere(R=R)
        clustering = self.hierarchical_clustering(self.dec_eigs[:sum(self.dimensions)])
        size2clust = {len(clust):clust for clust in clustering[1]}
        spectrumenv = [ np.real(np.mean(self.dec_eigs[size2clust[int(d)]])) for d in self.dimensions ]
        intraclass_var = 0
        for i,clust in size2clust.items():
          if i<=R:
            intraclass_var += np.sum((self.dec_eigs[clust]-np.mean(self.dec_eigs[clust]))**2)
          else:
            intraclass_var+= np.sum((self.dec_eigs[clust])**2)
        intraclass_var += np.sum((self.dec_eigs[clustering[0]])**2)
        intraclass_var += np.sum((self.dec_eigs[sum(self.dimensions):])**2)
        intraclass_var /=  len(self.dec_eigs)
        return intraclass_var, spectrumenv

    def SCCHEi_with_R_search(self, listeR=None, listekappa=None, figure=False):
        """ Searches for the resolution level R that minizes the thresholded intra class variance
        for the clustering returns by the SCCHEi algorithm """
        self.compute_dimensions_sphere()
        L = len(self.dimensions)
        if listeR is None or np.max(listeR)>L-1:
            listeR = [i for i in range(1,L-1)]
        if listekappa is None:
            listekappa = np.logspace(-4,0,1000)
        listeI = []
        listeSpectra = []
        for R in listeR:
            I, spectrumenv = self.SCCHEi(R)
            listeI.append(I)
            listeSpectra.append(spectrumenv)

        listeI = np.array(listeI)
        M = np.tile(listekappa.reshape(1,-1),(len(listeR),1))
        M /= self.n
        dims = np.array(self.dimensions)[listeR]
        M *= np.tile(dims.reshape(-1,1),(1,len(listekappa)))
        M += np.tile(listeI.reshape(-1,1),(1,len(listekappa)))
        R_kappa = np.argmin(M, axis=0)
        D_kappa = dims[R_kappa]
        gap = -np.float('inf')
        kappa0 = 0
        for i in range(len(listekappa)-1):
            newgap = D_kappa[i] - D_kappa[i+1]
            if newgap > gap:
                gap = newgap
                kappa0 = listekappa[i]

        crit = np.float('inf')
        for i,R in enumerate(listeR):
            nextcrit = listeI[i] + 2*kappa0*dims[i]/self.n
            if crit > nextcrit:
                self.spectrumenv = listeSpectra[i]
                crit = nextcrit
                self.adaptiveR = R
        if figure:
          plt.plot(np.log10(listekappa), D_kappa)
          plt.xlabel('$\log_{10}$ $\kappa$', fontsize=16)
          plt.ylabel('tilde{$R$}$(\kappa)$', fontsize=16)
        
    def estimation_envelope(self, t):
        """ Returns the estimated envelope evaluated at t """
        return self.function_gegenbauer(t, self.spectrumenv)

    def SCCHEi_Rmax(self):
        """ Performs the SCCHEi algorithm with the largest resolution possible given the size of the graph """
        self.compute_dimensions_sphere()  
        bestMSE = np.float('inf')
        bestR = 0
        self.spectrumenv = {}
        clustering = self.hierarchical_clustering(self.dec_eigs)
        size2clust = {len(clust):clust for clust in clustering[1]}
        self.spectrumenv = [ np.real(np.mean(self.dec_eigs[size2clust[int(d)]])) for d in self.dimensions ]

    def plot_estimation_envelope(self,True_envelope=True, savename=None, display=True):
        """ Plots the true and the estimated envelope functions """
        x = np.linspace(-1,1,100)
        fig=plt.figure()
        self.esti = [self.estimation_envelope(xi) for xi in x]
        self.esti = list(map(lambda x:max(0,x),self.esti))
        self.esti = list(map(lambda x:min(1,x),self.esti))
        plt.plot(x, self.esti, label='Estimated envelope')
        if True_envelope:
          self.true = [self.compute_envelope(xi) for xi in x]
          plt.plot(x, self.true, label='True envelope', linestyle='--')
        plt.legend(fontsize=13)
        if not(savename is None):
            plt.savefig(savename, dpi=250)
        if display:
            plt.show()
        else:
            plt.close(fig)

    def plot_estimation_envelope_real_data(self, savename=None):
        x = np.linspace(-1,1,100)
        self.esti = [self.estimation_envelope(xi) for xi in x]
        self.esti = [self.estimation_envelope(xi)  for xi in x]
        self.esti[1] = self.esti[0]
        M = np.max(self.esti)
        self.esti /= M
        self.esti = list(map(lambda x:max(0,x), self.esti))
        plt.plot(x, self.esti, label='Estimated envelope')
        if not(savename is None):
            plt.savefig(savename)
        plt.show()



    def check_HAC(self, Rmax):
        self.compute_dimensions_sphere()  
        clusters = self.hierarchical_clustering(self.dec_eigs, Rmax)
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        count = 0
        for i in range(len(clusters[1])):
          ls = list(np.real(self.dec_eigs[clusters[1][i]]))
          ax1.scatter(clusters[1][i], ls)
          count += len(ls)
        ls = list(np.real(self.dec_eigs[clusters[0]]))
        ax1.scatter(clusters[0], ls, label='eigenvalues set to 0')
        plt.legend()
        plt.show()


    def plot_adjacency_eigs_vs_SCCHEi_clusters(self, Rmax, savename=None, display=True):
        """ Represents the eiganvalues of the adjacency matrix and the estimated eigenvalues of the envelope function with multiplicity """
        self.compute_dimensions_sphere(R=Rmax)  
        fig=plt.figure()
        maxindUSVT = min(sum(self.dimensions),self.n-1)
        clusters = self.hierarchical_clustering(self.dec_eigs[:maxindUSVT])
        fig = plt.figure()
        dim2mean = {}
        means = np.zeros(sum([len(clusters[1][i]) for i in range(len(clusters[1]))]))
        for i in range(len(clusters[1])):
          mean = np.mean(self.dec_eigs[clusters[1][i]])
          for k in clusters[1][i]:
            means[k] = mean
        ax1 = fig.add_subplot(111)
        ax1.scatter([i for i in range(len(means))], self.dec_eigs[:len(means)],  s=20, c='blue', marker = 'x',label='Eigenvalues adjacency matrix')
        ax1.scatter([i for i in range(len(means))], means, s=20, c='red', marker = '+', label='Clusters built by SCCHEi')
        ax1.set_xlabel('Indexes eigenvalues envelope',fontsize=14)
        plt.legend(fontsize=14)
        if savename is not None:
            plt.savefig(savename, dpi=250)
        if display:
            plt.show()
        else:
            plt.close(fig)
            
    def plot_eigenvalues_clusters_labeled(self, R, thresholdmin=0, thresholdmax = 100, see_eigs_set_to_0 = False):
        """ Plots the eigenvalues of the adjacency matrix with colors corresponding to the clusters built by SCCHEi with resolution R """
        self.compute_dimensions_sphere(R=R)  
        maxindUSVT = min(sum(self.dimensions),self.n-1)
        clusters = self.hierarchical_clustering(self.dec_eigs[:maxindUSVT])
        fig = plt.figure(figsize=(10,4))
        ax1 = fig.add_subplot(111)
        for i in range(len(clusters[1])):
          eigens = list(filter(lambda x: (abs(x) < thresholdmax and abs(x)>=thresholdmin) , self.dec_eigs[clusters[1][i]]))
          size = len(eigens)
          ax1.scatter(np.abs(eigens) ,np.zeros(size), marker = '+')
        if see_eigs_set_to_0:
          eigens = list(filter(lambda x: (abs(x) < thresholdmax and abs(x)>=thresholdmin), self.dec_eigs[clusters[0]]))
          size = len(eigens)
          ax1.scatter(np.abs(eigens), np.ones(size),marker='*', label='Eigenvalues set to 0')
          plt.legend()
        ax1.set_xscale('log')
        ax1.set_xlabel('$\log(|eigenvalue|)$')
        ax1.minorticks_on()
        plt.show()