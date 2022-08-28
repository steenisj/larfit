'''
This script will store the models we will use to fit the data.
'''

import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class DefaultModels:
    def __init__(self):
        pass

    def Gauss(x,A,B): #Easy starting fit function
        y = A*np.exp(-1*B*x**2)
        return y

    def NRElectronYields(X, gamma, delta, epsilon, zeta, eta): #Couldn't get to work
        energy, Efield = X
        return (1/(gamma*(Efield**delta)))*(1/np.sqrt(energy+epsilon))*(1-(1/(1+(energy/zeta)**eta)))

    def NRTotalYields(energy, alpha, beta):
        return alpha*energy**beta

    def NRPhotonYields(X, alpha, beta, gamma, delta, epsilon, zeta, eta):
        energy, Efield = X
        return (alpha*energy**beta) - ((1/(gamma*(Efield**delta)))*(1/np.sqrt(energy+epsilon)))

    def ERPhotonYields(X): #Which one do I choose?
        pass

    def ERElectronYields(X): #Density?
        pass

    def alphaPhotonYields(Efield, A, B, C, D, E, F, G, H, I, J, K, L, M):
        quench = 1.0/(A*Efield**B)
        fieldTerm = H*(I+(Efield/J)**K)**(L)
        return quench*C*(D*E+(D/F)*(1-(G*np.log(1+(D/F)*(fieldTerm/M))/((D/F)*fieldTerm))))

    def alphaElectronYields(Efield, A, B, C, D, E, F, G, H, I, J):
        fieldTerm = F*(G+Efield**H)**(I)
        return A*(B - (B*C + (B/D)*(1-(E*np.log(1+(B/D)*(fieldTerm/J))/((B/D)*fieldTerm)))))



class ModelSelector:
    def __init__(self):
        pass

    def selector(self, func_index):
        if func_index == 0: #0
            dimension = 2
            init_params = [0,0]
            return DefaultModels.Gauss, init_params, dimension
        if func_index == 1: #1 
            dimension = 3
            init_params = [0.1, -0.0932, 2.998, 0.3, 2.94]
            return DefaultModels.NRElectronYields, init_params, dimension
        if func_index == 2: #2  
            dimension = 3
            init_params = [11.1, 0.087, 0.1, -0.0932, 2.998, 0.3, 2.94]
            return DefaultModels.NRPhotonYields, init_params, dimension
        if func_index == 3: #3 
            dimension = 2
            init_params = [11.1, 0.087]
            return DefaultModels.NRTotalYields, init_params, dimension
        if func_index == 4: #4  
            dimension = 2
            init_params = []
            return DefaultModels.ERPhotonYields, init_params, dimension
        if func_index == 5: #5 
            dimension = 2
            init_params = []
            return DefaultModels.ERElectronYields, init_params, dimension
        if func_index == 6: #6  
            dimension = 2
            init_params = [1.5, -0.012, 1/6500, 278037, 0.17, 1.21, 2, 0.6535, 4.985, 10.1, 1.21, -0.9798, 3]
            return DefaultModels.alphaPhotonYields, init_params, dimension
        if func_index == 7: #7 
            dimension = 2
            init_params = [1/6200, 64478399, 0.174, 1.21, 0.0285, 0.01, 4.716, 7.72, -0.11, 3]
            return DefaultModels.alphaElectronYields, init_params, dimension
