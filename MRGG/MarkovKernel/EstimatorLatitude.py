import numpy as np
import matplotlib.pyplot as plt

from ..Parameters import Parameters

class Kernels():
    """ Class that contains the definition of different kernels for kernel density estimation of the latitude function """
    def __init__(self):
        pass

    def gaussian(self, x):
        return (2*np.pi)**(-1/2) * np.exp(-0.5 * x**2)

class EstimatorLatitude(Parameters, Kernels):
    """ Class estimating the latitude function using the strategy described in  arXiv:1909.06841 """
    def __init__(self, nb_nodes, dimension, sampling_type):
        Kernels.__init__(self)
        Parameters.__init__(self, nb_nodes, dimension, sampling_type)

    def compute_gap(self, dec_eigs, i):
        """ Computes the spectral gap for the algorith HEiC """
        liste = [j for j in range(i)] + [j for j in range(i+self.d+1,len(dec_eigs)-1)]
        mat_eigs = np.tile(dec_eigs[liste].reshape(-1,1),(1,self.d+1))
        mat_eigs -= np.tile(dec_eigs[[j for j in range(i,i+self.d+1)]].reshape(1,-1),(len(liste),1))
        mat_eigs = np.abs(mat_eigs)
        max_eigs = np.max(mat_eigs, axis=1)
        gap = np.min(max_eigs)
        return gap

    def latent_distance_estimation(self, eigenvalues, vec, GRAM=False):
        """ Algorithm HEiC with estimation of the Gram matrix of the latent points """
        eig = np.copy(eigenvalues)
        eig = np.real(eig)

        dec_order = np.argsort(eig)[::-1]
        dec_eigs = eig[dec_order]
        gap = self.compute_gap(dec_eigs, 0)
        best_ind_eig = 1
        for i in range(1,len(eig)-self.d-1):
            next_gap = self.compute_gap(dec_eigs, i)
            if next_gap > gap:
              next_gap = gap
              best_ind_eig = i
        self.best_ind_eig = best_ind_eig
        V = vec[:,dec_order[best_ind_eig:best_ind_eig+self.d]]
        if GRAM:
            self.gram = np.real((self.n/self.d) * np.dot(V,V.T))
        # self.updiag_gram = self.n*np.array([self.gram[i,i+1] for i in range(self.gram.shape[0]-1)])
        self.updiag_gram = (self.n /self.d) * np.array([np.dot(V[i,:],V[i+1,:]) for i in range(V.shape[0]-1)])

        try:
            self.gram_true = np.array([[np.dot(self.V[:,i],self.V[:,j]) for i in range(self.n)] for j in range(self.n)])
            self.updiag_gram_true = np.array([self.gram_true[i,i+1] for i in range(self.gram_true.shape[0]-1)])
        except:
            pass


    def latitude_estimator(self, x, h=None):
        """ Performs a kernel density estimation of the latitude function """
        if h is None:
            h = (1/self.n)**(1/3)
        return sum(list(map(lambda u:self.gaussian(u)/(self.n*h), (self.updiag_gram - x)/h)))

    def check_gram_estimation(self):
        """ Plot side by side the true gram matrix and the estimated one """
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(self.n * self.gram)
        true_gram = np.array([[np.dot(self.V[:,i].T,self.V[:,j]) for i in range(self.n)] for j in range(self.n)])
        axs[1].imshow(true_gram)
        plt.show()
        sc = plt.imshow(img)
        plt.colorbar(sc)

    def plot_densities_latitude(self, h=None, savename=None, display=True):
        """ Plot the true and the estimated density of the cosinus of the angle between two consecutive states of the Markov chain on the Sphere """
        x = np.linspace(-0.9,0.9,100)
        fig=plt.figure()
        esti = list(map(lambda a: self.latitude_estimator(a, h=h), x))
        true = np.array(list(map(self.density_latitude, x)))
        plt.plot(x, esti/sum(esti), label='Estimated latitude')
        plt.plot(x, true/sum(true), label='True latitude',linestyle='--')
        plt.legend(fontsize=13)
        #plt.title('Latitude Density Estimation')
        if not(savename is None):
            plt.savefig(savename, dpi=250)
        if display:
            plt.show()
        else:
            plt.close(fig)


    def plot_density_latitude_real_data(self, percent=0.1, bandwidth=0.2, savename=None):
        """ Plot the true and the estimated density of the cosinus of the angle between two consecutive states of the Markov chain on the Sphere """
        L = np.real(self.updiag_gram) 
        Labs = np.sort(np.abs(self.updiag_gram))[::-1]
        size = percent * len(self.updiag_gram)
        normalize = Labs[int(size)]
        L = L[int(size):] / normalize
        L = np.array(list(map(lambda x : min(max(-1,x),x),L))).reshape(-1,1)
        from sklearn.neighbors import KernelDensity
        kde = KernelDensity(kernel='linear', bandwidth=bandwidth).fit(L.reshape(-1,1))
        x = np.linspace(-1,1,100)
        log_dens = kde.score_samples(x.reshape(-1,1))
        plt.plot(x, np.exp(log_dens)/sum(np.exp(log_dens)))
        if not(savename is None):
            plt.savefig(savename)
        plt.show()