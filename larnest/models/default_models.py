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


class ModelSelector:
    def __init__(self):
        pass

    def selector(self, func_index):
        if func_index == 0: #0
            dimension = 2
            init_params = [0,0]
            return DefaultModels.Gauss, init_params, dimension
        if func_index == 1: #1 Failed
            dimension = 3
            init_params = [0.1, -0.0932, 2.998, 0.3, 2.94]
            return DefaultModels.NRElectronYields, init_params, dimension
        if func_index == 2: #2 Recreated 
            dimension = 3
            init_params = [11.1, 0.087, 0.1, -0.0932, 2.998, 0.3, 2.94]
            return DefaultModels.NRPhotonYields, init_params, dimension
        if func_index == 3: #3 Recreated
            dimension = 2
            init_params = [11.1, 0.087]
            return DefaultModels.NRTotalYields, init_params, dimension
