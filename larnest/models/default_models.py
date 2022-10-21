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

    def NRElectronYields(X, *args):
        energy, Efield = X
        gamma, delta, epsilon, zeta, eta = args
        return (1/(gamma*(Efield**delta)))*(1/np.sqrt(energy+epsilon))*(1-(1/(1+(energy/zeta)**eta)))

    def NRTotalYields(energy, *args):
        alpha, beta = args
        return alpha*energy**beta

    def NRPhotonYields(X, *args):
        energy, Efield = X
        alpha, beta, gamma, delta, epsilon = args
        return (alpha*energy**beta) - ((1/(gamma*(Efield**delta)))*(1/np.sqrt(energy+epsilon)))

    def alphaPhotonYields(Efield, *args):
        A, B, C, D, E, F, G, H, I, J, K, L, M = args
        quench = 1.0/(A*Efield**B)
        fieldTerm = H*(I+(Efield/J)**K)**(L)
        return quench*C*(D*E+(D/F)*(1-(G*np.log(1+(D/F)*(fieldTerm/M))/((D/F)*fieldTerm))))

    def alphaElectronYields(Efield, *args):
        A, B, C, D, E, F, G, H, I, J = args
        fieldTerm = F*(G+Efield**H)**(I)
        return A*(B - (B*C + (B/D)*(1-(E*np.log(1+(B/D)*(fieldTerm/J))/((B/D)*fieldTerm)))))

    def GetERElectronYieldsAlpha(Efield, *args):
        A, B, C, D, E, F, G = args
        density = 1.396 #Density of LAr in g/L
        return A + B/(C+(Efield/(D+E*np.exp(density/F)))**G)

    def GetERElectronYieldsBeta(Efield, *args):
        A, B, C, D, E, F = args
        return A + B*(C+(Efield/D)**E)**F

    def GetERElectronYieldsGamma(Efield, *args): #fWorkQuantaFunction?
        A, B, C, D, E, F, G = args
        WorkQuantaFunction = 19.5
        return A*((B/WorkQuantaFunction)+C*(D+(E/((Efield/F)**G))))

    def GetERElectronYieldsDokeBirks(Efield, *args):
        A, B, C, D, E = args
        return A+B/(C+(Efield/D)**E)

    def GetERElectronYields(X, *args):
        p1, p2, p3, p4, p5, delta, let = args
        energy, Efield = X
        return DefaultModels.alpha*DefaultModels.beta + (DefaultModels.gamma-DefaultModels.alpha*DefaultModels.beta)/(p1+(p2*(energy+0.5)**p3)) + delta/(p5 + DefaultModels.doke_birks*energy**let)

    def GetERTotalYields(energy, WorkQuantaFunction):
        #WorkQuantaFunction = 19.5
        #I'm going to make this a parameter and fit it to the data to see if it matches.
        return (energy*1000)/WorkQuantaFunction

    def ERPhotonYields(X):
        pass

class LeastSquaresModels:
    def __init__(self):
        pass

class ModelSelector:
    def __init__(self):
        pass

    def selector(self, func_index):
        retrieved_models = []
        dimension = []
        init_params = []
        param_list = []
        off_parameters = []

        if func_index == 0: #0
            dimension.append(2)
            init_params.append([1,1])
            param_list.append(['A','B'])
            off_parameters.append({'B': 0})
            retrieved_models.append(DefaultModels.Gauss)
        elif func_index == 1: #1 
            dimension.append(3)
            init_params.append([0.1, -0.0932, 2.998, 0.3, 2.94])
            param_list.append(['gamma', 'delta', 'epsilon', 'zeta', 'eta'])
            off_parameters.append({'delta': 0, 'epsilon': 0, 'zeta': 1, 'eta': 0})
            retrieved_models.append(DefaultModels.NRElectronYields)
        elif func_index == 2: #2  
            dimension.append(3)
            init_params.append([11.1, 0.087, 0.1, -0.0932, 2.998])
            param_list.append(['alpha', 'beta', 'gamma', 'delta', 'epsilon'])
            off_parameters.append({'beta':0, 'gamma':1, 'delta':0, 'epsilon':0})
            retrieved_models.append(DefaultModels.NRPhotonYields)
        elif func_index == 3: #3 
            dimension.append(2)
            init_params.append([11.1, 0.087])
            param_list.append(['alpha', 'beta'])
            off_parameters.append({'beta': 0})
            retrieved_models.append(DefaultModels.NRTotalYields)
        elif func_index == 4: #4  
            dimension.append(2)
            init_params.append([1.5, -0.012, 1/6500, 278037, 0.17, 1.21, 2, 0.6535, 4.985, 10.1, 1.21, -0.9798, 3])
            param_list.append(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M'])
            off_parameters.append({'A': 0, 'B': 0, 'D': 0, 'E': 0, 'F': 1, 'G': 0, 'H': 0, 'I': 0, 'J': 0, 'K': 0, 'L': 0, 'M': 1})
            retrieved_models.append(DefaultModels.alphaPhotonYields)
        elif func_index == 5: #5
            dimension.append(2)
            init_params.append([1/6200, 64478399, 0.174, 1.21, 0.0285, 0.01, 4.716, 7.72, -0.11, 3])
            param_list.append(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
            off_parameters.append({'B': 1, 'C': 0, 'D': 1, 'E': 0, 'J': 1})#,'F': 1, 'G': 0, 'H': 0, 'I': 0}
            retrieved_models.append(DefaultModels.alphaElectronYields)
        elif func_index == 6: #6 
            ####################################
            dimension.append(2)
            init_params.append([32.998, -552.988, 17.23, -4.7, 0.025, 0.27, 0.242])
            param_list.append(['A', 'B', 'C', 'D', 'E', 'F', 'G'])
            off_parameters.append({'B': 0, 'C': 0, 'D': 1, 'E': 0, 'F': 1, 'G': 0})
            retrieved_models.append(DefaultModels.GetERElectronYieldsAlpha)
            ####################################
            dimension.append(2)
            init_params.append([0.778, 25.9, 1.105, 0.4, 4.55, -7.5])
            param_list.append(['A', 'B', 'C', 'D', 'E', 'F'])
            off_parameters.append({'B': 0, 'C': 0, 'D': 1, 'E': 0, 'F': 0})
            retrieved_models.append(DefaultModels.GetERElectronYieldsBeta)
            ####################################
            dimension.append(2)
            init_params.append([0.6595, 1000, 6.5, 5, -0.5, 1047.4, 0.0185])
            param_list.append(['A', 'B', 'C', 'D', 'E', 'F', 'G'])
            off_parameters.append({'B': 0, 'C': 1, 'D': 1, 'E': 0, 'F': 1, 'G': 0})
            retrieved_models.append(DefaultModels.GetERElectronYieldsGamma)
            ####################################
            dimension.append(2)
            init_params.append([1052.3, 14159350000-1652.3, -5, 0.158, 1.84])
            param_list.append(['A', 'B', 'C', 'D', 'E'])
            off_parameters.append({'B': 0, 'C': 1, 'D': 1, 'E': 0})
            retrieved_models.append(DefaultModels.GetERElectronYieldsDokeBirks)
            ##################################### 
            dimension.append(3) #Put the final model at the end; once the others have been done
            init_params.append([1, 10.3, 13.1, 0.1, 0.7, 15.7, -2.1])
            param_list.append(['p1', 'p2', 'p3', 'p4', 'p5', 'delta', 'let'])
            off_parameters.append({'p2': 0, 'p3': 0, 'p4': 0, 'p5': 1, 'delta': 0, 'let': 0})
            retrieved_models.append(DefaultModels.GetERElectronYields)    
            ####################################  
        elif func_index == 7: #7 
            dimension.append(2)
            init_params.append([None])
            param_list.append([None])
            off_parameters.append({})
            retrieved_models.append(DefaultModels.ERPhotonYields)
            
        elif func_index == 8: #8
            dimension.append(2)
            init_params.append([None])
            param_list.append([None])
            off_parameters.append({})
            retrieved_models.append(DefaultModels.GetERTotalYields)
        else:
            print("Error with Model Selector. Take a peak at default_models.py!")
            
        return retrieved_models, init_params, dimension, param_list, off_parameters

        '''elif func_index == 8: #8 
            dimension = 2
            init_params = [32.998, -552.988, 17.23, -4.7, 0.025, 0.27, 0.242]
            param_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
            off_parameters = {'B': 0, 'C': 0, 'D': 1, 'E': 0, 'F': 1, 'G': 0}
            return DefaultModels.GetERElectronYieldsAlpha, init_params, dimension, param_list, off_parameters
        elif func_index == 9: #9 
            dimension = 2
            init_params = [0.778, 25.9, 1.105, 0.4, 4.55, -7.5]
            param_list = ['A', 'B', 'C', 'D', 'E', 'F']
            off_parameters = {'B': 0, 'C': 0, 'D': 1, 'E': 0, 'F': 0}
            return DefaultModels.GetERElectronYieldsBeta, init_params, dimension, param_list, off_parameters
        elif func_index == 10: #10
            dimension = 2
            init_params = [0.6595, 1000, 6.5, 5, -0.5, 1047.4, 0.0185]
            param_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
            off_parameters = {'B': 0, 'C': 1, 'D': 1, 'E': 0, 'F': 1, 'G': 0}
            return DefaultModels.GetERElectronYieldsGamma, init_params, dimension, param_list, off_parameters
        elif func_index == 11: #11
            dimension = 2
            init_params = [1052.3, 14159350000-1652.3, -5, 0.158, 1.84]
            param_list = ['A', 'B', 'C', 'D', 'E']
            off_parameters = {'B': 0, 'C': 1, 'D': 1, 'E': 0}
            return DefaultModels.GetERElectronYieldsDokeBirks, init_params, dimension, param_list, off_parameters'''