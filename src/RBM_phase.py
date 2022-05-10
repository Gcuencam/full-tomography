# -*- coding: utf-8 -*-
from ctypes import sizeof
import numpy as np
import scipy
from enum import Enum

class TypeP(Enum):
    b = 0
    c = 1
    w = 2

class Param_idx(object):
    def __init__(self, typeP, idx, idx_abs):
        self.typeP = typeP
        self.idx = idx
        self.idx_abs = idx_abs

class RBM_phase(object):
    def __init__(self, rbm_mag, nH, b=None, c=None, w=None):
        self.rbm_mag = rbm_mag
        self.nV = self.rbm_mag.nV
        self.nH = nH

        # Initialization at 0 of the weight parameters.
        self.b = np.zeros(self.nV) if b is None else b
        self.c = np.zeros(self.nH) if c is None else c
        self.w = np.zeros((self.nH, self.nV)) if w is None else w
        
        self.par_vec = self.Create_par_vec()
        self.len_par_vec = self.nV + self.nH + self.nH*self.nV
        self.I = np.identity(2)
        self.H = (1/np.sqrt(2))*np.array([[1,1],[1,-1]])
        self.K = (1/np.sqrt(2))*np.array([[1,-1j],[1,1j]])


    def Create_par_vec(self):
        par_vec = []
        i = 0
        for i_bias in range(self.nV):
            par_vec.append(Param_idx(TypeP.b,i_bias,i))
            i+=1
        for i_c in range(self.nH):
            par_vec.append(Param_idx(TypeP.c,i_c,i))
            i+=1
        for i_w in range(self.nH):
            for j_w in range(self.nV):
                par_vec.append(Param_idx(TypeP.w,[i_w,j_w],i))
                i+=1
        return par_vec

    def fi(self, vec):
        sumBias_v = np.dot(self.b,vec)
        sum_i = 0
        for i in range(self.nH):
            sum_j = 0
            for j in range(self.nV):
                sum_j += self.w[i,j]*vec[j]
            sum_i += np.log(1+np.exp(self.c[i]+sum_j))
        return sumBias_v + sum_i
    
    def waveF_Qb(self, vec):
        return np.sqrt(self.rbm_mag.rho(vec))*np.exp(complex(0,self.fi(vec)))

    def waveF(self,vec):
        basis = self.rbm_mag.Basis(self.nV)
        Z = 0
        for vec_b in basis:
            Z += self.rbm_mag.rho(vec_b)
            
        mag = np.sqrt(self.rbm_mag.rho(vec)/Z)
        fi = self.fi(vec)
        
        wf = mag*np.exp(complex(0,fi))
        return wf
    
    def createUb(self, basis, i_b):
        n = self.nV
        idx = np.zeros(n)
        if basis == 0:  #XX
            idx[i_b] = 1
            idx[i_b+1] = 1
        else:   #XY
            idx[i_b] = 1
            idx[i_b+1] = 2
        
        if idx[0] == 0 and idx[1] == 0:
            Ub = np.kron(self.I,self.I)
        elif idx[0] == 0 and idx[1] == 1:
            Ub = np.kron(self.I,self.H)
        elif idx[0] == 1 and idx[1] == 1:
            Ub = np.kron(self.H,self.H)
        elif idx[0] == 1 and idx[1] == 2:
            Ub = np.kron(self.H,self.K)
            
        for i in range(2,n):
            if idx[i] == 0: 
                Ub = np.kron(Ub,self.I)
            elif idx[i] == 1:
                Ub = np.kron(Ub,self.H)
            elif idx[i] == 2:
                Ub = np.kron(Ub,self.K)
        self.Ub = Ub.copy()
         
    def create_g(self, mini_batch, sBatch):
        g = np.zeros(self.len_par_vec)
        for k in self.par_vec:
            sum_B = 0
            for data in mini_batch:
                sum_B += self.D_mu_Qb(data,k).imag
            g[k.idx_abs] = sum_B/sBatch
        return g
        
        
    def generateSigma(self, i):
        n = self.nV
        sigma = np.zeros(n)
        strBin = format(i,"b")
        length = len(strBin)
        for j in range(length):
            sigma[n-1-j] = int(strBin[length-1-j])
            
        return np.array(sigma)

    def D_mu_Qb(self, i_sigma_b, k):
        N = 2**self.nV
        sum_num = 0
        sum_den = 0
        for i in range(N):
            if (self.Ub[i_sigma_b,i] != 0):
                sigma = self.generateSigma(i)
                Qb = self.Ub[i_sigma_b,i]*self.waveF_Qb(sigma)
                sum_num += Qb*self.D_mu(sigma,k)
                sum_den += Qb
        return sum_num/sum_den
        
    def D_mu(self, vec, param_idx):
        if param_idx.typeP == TypeP.b:
            k = param_idx.idx
            return vec[k]
        
        elif param_idx.typeP == TypeP.c:
            k = param_idx.idx
            sum_j = 0
            for j in range(self.nV):
                sum_j += self.w[k,j]*vec[j]
            exp = np.exp(self.c[k] + sum_j)
            return (1/(1+exp))*exp
        
        elif param_idx.typeP == TypeP.w:
            k_i = param_idx.idx[0]
            k_j = param_idx.idx[1]
            sum_j = 0
            for j in range(self.nV):
                sum_j += self.w[k_i,j]*vec[j]
            exp = np.exp(self.c[k_i] + sum_j)
            return (1/(1+exp))*exp*vec[k_j]
    
    def updateParams(self, learnRate, g):
        for k in self.par_vec:
            upd = learnRate*g[k.idx_abs]
            if k.typeP == TypeP.b:
                self.b[k.idx] = self.b[k.idx]-upd
            elif k.typeP == TypeP.c:
                self.c[k.idx] = self.c[k.idx]-upd
            elif k.typeP == TypeP.w:
                self.w[k.idx[0],k.idx[1]] = self.w[k.idx[0],k.idx[1]]-upd
    
    
    def train(self, epochs, data_path, batch_size, learnRate):
        print('Training starts.')
        basis_meas = ['XX','XY']
        lb = len(basis_meas)
        for epoch in range(epochs):
            for b in range(lb):
                b_path = data_path + basis_meas[b] + '_'
                for i_b in range(self.nV-1):
                    self.createUb(b,i_b)
                    dataset = np.int_(np.loadtxt(b_path + str(i_b) + '.txt'))
                    for iBatch in range(0, len(dataset), batch_size):
                        mini_batch = dataset[iBatch:iBatch + batch_size]
                        g = self.create_g(mini_batch, len(mini_batch))
                        self.updateParams(learnRate, g)
            pc = (((epoch + 1) / epochs) * 100)
            if pc % 10 == 0:
                print(str(int(pc)) + '% trained.')

        
    def overlap(self, target_wf):
        wf = []
        for vec in self.rbm_mag.Basis(self.nV):
            wf.append(self.waveF(vec))
        wf = np.array(wf)
        return np.abs(np.dot(np.conj(target_wf),wf))