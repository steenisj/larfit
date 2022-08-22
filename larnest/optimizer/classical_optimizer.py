'''
This script will allow us to perform least squares fitting on data.
'''

#In the Z axis, we should be plotting yield/energy which we ALSO call yield (nick says 'yield density') <N> = Y*E

from dataclasses import dataclass
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.insert(0,'models') #Change to the /models folder
from default_models import DefaultModels as dm
from default_models import ModelSelector as ms

class LeastSquares:
    '''
    To perform least squares
    '''
    def __init__(self):
        pass

    def least_squares(self, dataset_label, x_index, y_index, z_index, func_index):
        model, init_params, dimension = ms.selector(self, func_index)
        energy = pd.to_numeric(self.data[dataset_label].iloc[0:,x_index]).to_numpy() 
        Efield = pd.to_numeric(self.data[dataset_label].iloc[0:,y_index]).to_numpy() 
        init_n_yield = pd.to_numeric(self.data[dataset_label].iloc[0:,z_index]).to_numpy() #Before being divided by energy
        #n_yield = np.divide(init_n_yield,energy)

        #X and Y range for plotting the fit
        x_min = np.min(energy)
        x_max = np.max(energy)
        x_range = np.arange(x_min, x_max, 0.1)
        num_x = (x_max-x_min)/0.1

        y_min = np.min(Efield)
        y_max = np.max(Efield)
        y_interval = (y_max-y_min)/num_x
        y_range = np.arange(y_min, y_max, y_interval)

        if dimension==2:
            parameters, covariance = curve_fit(model, energy, n_yield, p0=init_params) #Storing errors and params from fit
            print(parameters)

            fit_y = model(x_range, *parameters)
            print(fit_y)
            plt.plot(energy, init_n_yield, 'o', label='data')
            plt.plot(x_range, fit_y, '-', label='fit')
            plt.xlabel('Energy')
            plt.ylabel('Yield Density')
            plt.title(dataset_label)
            plt.legend()
            plt.show()

            return parameters, x_range, fit_y

        if dimension==3:
            parameters, covariance = curve_fit(model, (energy, Efield), init_n_yield, p0=init_params) #Storing errors and params from fit
            print(parameters)
            #print(n_yield)

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.set_title(dataset_label)
            ax.scatter(energy, Efield, init_n_yield, label='Data')
            ax.set_xlabel(self.data[dataset_label].columns[x_index])
            ax.set_ylabel(self.data[dataset_label].columns[y_index])
            ax.set_zlabel('Yield Density')#self.data[dataset_label].columns[z_index]) 
            ax.set_xscale('log')

            X, Y = np.meshgrid(x_range, y_range)
            z_range = model((X, Y), *parameters)
            Z = z_range
            ax.plot_surface(X, Y, Z, color='orange')
            ax.legend()
            fig.tight_layout()
            plt.show()

            return parameters, x_range, y_range

    def light_yield_plots(self, func_index, parameters, x_range, y_range):
        #x->energy y->Efield z->yield
        model, init_params, dimension = ms.selector(self, func_index)
        Efield_min = min(y_range)
        Efield_max = max(y_range)
        E_interval = (Efield_max - Efield_min)/10
        E_arr = np.arange(Efield_min, Efield_max, E_interval)

        for i in E_arr: #Iterating through Efield values
            yields = model((x_range, i), *parameters)
            plt.plot(x_range, yields, label=round(i, 0))
        plt.xlabel('Energy')
        plt.ylabel('Yield Density')
        legend = plt.legend()
        legend.set_title("E-Field [V/cm]", prop = {'size':10})
        plt.title('E v. Yield Density')
        plt.xscale('log')
        plt.show()

