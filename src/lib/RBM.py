# -*- coding: utf-8 -*-
import math
import numpy as np



class RBM(object):
    def __init__(self,nV,nH,learningRate):
        self._nV = nV
        self._nH = nH
        self._learningRate = learningRate
        self._w = np.random.rand(self._nV,self._nH)
        self._b = np.random.rand(self._nH)
        self._c = np.random.rand(self._nV)
        
        
        
    def sigmoid(self,X):
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


    def activ(self,X):
        l = len(X)
        out = np.zeros(l)
        for i in range(l):
            x = X[i]
            r = np.random.uniform()
            if x>r:
                out[i] = 1
            else:
                out[i] = 0
        return out
        
    
    def forward(self,v):
        p_hv = self._b + np.dot(v,self._w)
        s = self.sigmoid(p_hv)
        h = self.activ(s)
        return h
    
    def backward(self,h):
        p_vh = self._c + np.dot(self._w,h)
        s = self.sigmoid(p_vh)
        v = self.activ(s)
        return v
    
      
    def gibbsSampling(self,v0,K):
        v = v0.copy()
        for k in range(K):
            h = self.forward(v)
            v = self.backward(h)
            if k==0:
                h0 = h.copy()
        vk = v.copy()
        hk = self.forward(vk)
        return h0,vk,hk
             
    def updateParams(self,v0,h0,vk,hk):
        posGrad = np.outer(v0,h0)
        negGrad = np.outer(vk,hk)
        
        self._w = self._w + self._learningRate * (posGrad-negGrad)
        self._b = self._b + self._learningRate * (h0-hk)
        self._c = self._c + self._learningRate * (v0-vk)
        
        return               
    
    
    
    def CD_K(self,dataset,epochs,K):
        for i in range(epochs):
            for sample in dataset:
                [h0,vk,hk] = self.gibbsSampling(sample,K)
                self.updateParams(sample,h0,vk,hk)
        return

    def run(self,v,K):        
        for k in range(K):
            h = self.forward(v)
            v = self.backward(h)
        return v



