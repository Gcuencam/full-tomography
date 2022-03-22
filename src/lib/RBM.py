# -*- coding: utf-8 -*-
from ctypes import sizeof
import math
import numpy as np


class RBM(object):
    def __init__(self, nV, nH):
        self._nV = nV
        self._nH = nH

        # Random initialization of the weight parameters.
        self._w = np.random.rand(self._nV, self._nH)
        self._b = np.random.rand(self._nH)
        self._c = np.random.rand(self._nV)

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
        p_hv = self._b + np.dot(v, self._w)
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
        p_vh = self._c + np.dot(self._w, h)
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
        self._w = self._w + sLearnRate * cumGrad_w
        self._b = self._b + sLearnRate * cumGrad_b
        self._c = self._c + sLearnRate * cumGrad_c
        return

    # Implements the CD_K training algorithm with a division of the
    # dataset in mini-batches.

    def CD_K(self, dataset, epochs, batch_size, K, learnRate):
        for epoch in range(epochs):
            error = 0
            for iBatch in range(0, len(dataset), batch_size):
                mini_batch = dataset[iBatch:iBatch + batch_size]
                cumGrad_w = np.zeros((self._nV, self._nH))
                cumGrad_b = np.zeros(self._nH)
                cumGrad_c = np.zeros(self._nV)
                for data in mini_batch:
                    [h0, vk, hk, vs] = self.gibbsSampling(data, K)
                    [grad_w, grad_b, grad_c] = self.calculateGradients(data, h0, vk, hk)
                    cumGrad_w += grad_w
                    cumGrad_b += grad_b
                    cumGrad_c += grad_c
                    error += np.sum((data - vs) ** 2) / dataset.size
                self.updateParams(cumGrad_w, cumGrad_b, cumGrad_c, len(mini_batch), learnRate)
            print("Epoch %s: error is %s" % (epoch, error))
        return

    # Obtains a reconstruction of the input by running the network forward and backward.
    def run(self, v, K):
        for k in range(K):
            h = self.forward(v)
            [v, s] = self.backward(h)
        return v
