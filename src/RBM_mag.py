# -*- coding: utf-8 -*-
from ctypes import sizeof
from matplotlib import pyplot as plt
import math
import numpy as np


class RBM_mag(object):
    def __init__(self, nV, nH, w=None, b=None, c=None):
        self.nV = nV
        self.nH = nH

        # Initialization at 0 of the weight parameters.
        self.w = np.zeros((self.nV, self.nH)) if w is None else w
        self.b = np.zeros(self.nH) if b is None else b
        self.c = np.zeros(self.nV) if c is None else c

    def getParams(self):
        return self.w.copy(), self.b.copy(), self.c.copy()

    def setParams(self, w, b, c):
        self.w = w.copy()
        self.b = b.copy()
        self.c = c.copy()

    # Computes the sigmoid function for all the neurons of the layer.
    def sigmoid(self, X):
        l = len(X)
        out = np.zeros(l)
        for i in range(l):
            x = X[i]
            if x >= 0:
                z = math.exp(-x)
                sig = 1 / (1 + z)
                out[i] = sig
            else:
                z = math.exp(x)
                sig = z / (1 + z)
                out[i] = sig
        return out

    # Computes the activation function for all the neurons of the layer.
    def activ(self, X):
        l = len(X)
        out = np.zeros(l)
        for i in range(l):
            x = X[i]
            r = np.random.uniform()
            if x > r:
                out[i] = 1
            else:
                out[i] = 0
        return out

    # Runs the network forward. Given a configuration of the visible layer calculates
    # the hidden layer one.
    def forward(self, v):
        p_hv = self.b + np.dot(v, self.w)
        s = self.sigmoid(p_hv)
        h = self.activ(s)
        return h

    def backward(self, h):
        """Runs the network bakcward. Given a configuration of the hidden layer calculates the visible layer one.
        Args:
            h (numpy.ndarray): the state of a given hidden layer
        Returns:
            v (numpy.ndarray): stochastic value of the rebuilded visible layer
            s (numpy.ndarray): analytic value of the rebuilded visible layer
        """
        p_vh = self.c + np.dot(self.w, h)
        s = self.sigmoid(p_vh)
        v = self.activ(s)
        return v, s

    # Obtains the k sample of the visible and hidden layer.
    def gibbsSampling(self, v0, K):
        v = v0.copy()
        for k in range(K):
            h = self.forward(v)
            [v, s] = self.backward(h)
            if k == 0:
                h0 = h.copy()
        vk = v.copy()
        vs = s.copy()
        hk = self.forward(vk)

        return h0, vk, hk, vs

    def calculateGradients(self, v0, h0, vk, hk):
        grad_w = np.outer(v0, h0) - np.outer(vk, hk)  # positive gradient - negative gradient
        grad_b = h0 - hk
        grad_c = v0 - vk
        return grad_w, grad_b, grad_c

    # Updates the weights and the bias.
    def updateParams(self, cumGrad_w, cumGrad_b, cumGrad_c, sBatch, learnRate):
        sLearnRate = learnRate / sBatch  # Adjusts the learning rate to the size of the batch.
        self.w = self.w + sLearnRate * cumGrad_w
        self.b = self.b + sLearnRate * cumGrad_b
        self.c = self.c + sLearnRate * cumGrad_c
        return

    # Implements the CD_K training algorithm with a division of the
    # dataset in mini-batches.
    def CD_K(self, dataset, epochs, batch_size, K, learnRate):
        print('Training starts.')
        error_plot = []
        for epoch in range(epochs):
            error = 0
            for iBatch in range(0, len(dataset), batch_size):
                mini_batch = dataset[iBatch:iBatch + batch_size]
                cumGrad_w = np.zeros((self.nV, self.nH))
                cumGrad_b = np.zeros(self.nH)
                cumGrad_c = np.zeros(self.nV)
                for data in mini_batch:
                    [h0, vk, hk, vs] = self.gibbsSampling(data, K)
                    [grad_w, grad_b, grad_c] = self.calculateGradients(data, h0, vk, hk)
                    cumGrad_w += grad_w
                    cumGrad_b += grad_b
                    cumGrad_c += grad_c
                    error += np.sum((data - vs) ** 2)
                self.updateParams(cumGrad_w, cumGrad_b, cumGrad_c, len(mini_batch), learnRate)
            error = error / dataset.size
            print("Epoch %s: error is %s" % (epoch, error))
            error_plot.append(error)
            pc = (((epoch + 1) / epochs) * 100)
            if pc % 10 == 0:
                print(str(int(pc)) + '% trained.')
        plt.plot(error_plot)
        plt.savefig('error.png')
        return self.nV, self.nH, self.w, self.b, self.c

    # Obtains a reconstruction of the input by running the network forward and backward.
    def run(self, v, K):
        for k in range(K):
            h = self.forward(v)
            [v, s] = self.backward(h)
        return v
    
    
    
    # Obtains the 2^n possible basis.
    def Basis(self, n):
        m = 2**n
        basis = []
        for i in range(m):
            vec = np.zeros(n)
            strBin = format(i,"b")
            length = len(strBin)
            for j in range(length):
                vec[n-1-j] = int(strBin[length-1-j])
            basis.append(vec)
        return basis
    
    # Calculates the amplitude represented by the network for a given vector. (Formula 7 of Torlai's paper.)
    def rho(self, vec):
        sumBias_v = np.dot(self.c,vec)
        sum_i = 0
        for i in range(self.nH):
            sum_j = 0
            for j in range(self.nV):
                sum_j += self.w[j,i]*vec[j]
            sum_i += math.log(1+math.exp(self.b[i]+sum_j))
        return math.exp(sumBias_v + sum_i)
    
    # Calculates the amplitude normalized for a given vector.
    def waveF(self, sigma):
        basis = self.Basis(self.nV)
        Z = 0
        for vec in basis:
            Z += self.rho(vec)
        return math.sqrt(self.rho(sigma)/Z)
    
    
    # Calculates the squared overlap of the state represented by the network
    # and the target state.
    def Squared_Overlap(self, basisT, n, k_gs):
        N = len(basisT)
        basis = self.Basis(self.nV)

        sum_n = 0
        for j in range(n):
            indRand = np.random.randint(2**self.nV)
            sigma_j = self.run(basis[indRand].copy(),k_gs)
            sum_N = 0
            flag = False
            k = 0
            while k<N and flag==False:
                if np.array_equal(sigma_j,basisT[k]):
                    flag = True
                    sum_N += 1/math.sqrt(N)
                k += 1
            if sum_N != 0:
                sum_n += sum_N/math.sqrt(self.rho(sigma_j.copy()))
        overlap = sum_n/n    
        sum_N = 0
        for k in range(N):
            sum_N += math.sqrt(self.rho(basisT[k].copy())/N)
        overlap = overlap*sum_N
        return overlap