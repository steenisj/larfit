'''
This script will allow us to perform least squares fitting on data.
'''

#In the Z axis, we should be plotting yield/energy which we ALSO call yield (nick says 'yield density') <N> = Y*E

from dataclasses import dataclass
from turtle import color
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from iminuit import Minuit

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

    def curve_fit_least_squares(self, dataset_label, x_index, y_index, z_index, func_index):
        model, init_params, dimension = ms.selector(self, func_index)
        x_arr = pd.to_numeric(self.data[dataset_label].iloc[0:,x_index]).to_numpy() 
        y_arr = pd.to_numeric(self.data[dataset_label].iloc[0:,y_index]).to_numpy() 
        n_yield = pd.to_numeric(self.data[dataset_label].iloc[0:,z_index]).to_numpy() #Before being divided by energy
        labels = self.data[dataset_label].iloc[0:,0].astype(str).values.tolist()
        #SOMEWHAT HARD CODED TO PICK ERRORS and only chooses one of the two error columns
        yield_errors = pd.to_numeric(self.data[dataset_label].iloc[0:,z_index+1]).to_numpy() 

        #X and Y range for plotting the fit
        x_min = np.min(x_arr)
        x_max = np.max(x_arr)
        x_range = np.arange(x_min, x_max, 0.1)
        num_x = (x_max-x_min)/0.1

        y_min = np.min(y_arr)
        y_max = np.max(y_arr)
        y_interval = (y_max-y_min)/num_x
        y_range = np.arange(y_min, y_max, y_interval)

        if dimension==2:
            #print(yield_errors)
            if dataset_label == 'alpha_charge':
                parameters, covariance = curve_fit(model, x_arr, n_yield, p0=init_params, maxfev=8000) #Storing errors and params from fit
            else:
                parameters, covariance = curve_fit(model, x_arr, n_yield, p0=init_params, sigma=yield_errors, maxfev=8000) #Storing errors and params from fit
            print(parameters)

            fit_z = model(x_range, *parameters)
            #print(fit_z)
            if dataset_label == 'alpha_light' or dataset_label == 'alpha_charge':
                reduced_energies = []
                for i in y_arr: #Iterating over the energies
                    if i in reduced_energies:
                        continue
                    elif i not in reduced_energies:
                        reduced_energies.append(i)
                for i in np.arange(len(reduced_energies)):
                    current_energy = reduced_energies[i]
                    indices = [j for j, x in enumerate(y_arr) if x == current_energy] 
                    new_Efield_arr = x_arr[indices]
                    new_yield_arr = n_yield[indices]
                    plt.plot(new_Efield_arr, new_yield_arr, 'o', label=current_energy)
                plt.plot(x_range, fit_z, '-', label='fit')
                plt.xlabel(self.data[dataset_label].columns[x_index]) 
                plt.ylabel(self.data[dataset_label].columns[z_index])
                plt.title(dataset_label)
                legend_2d = plt.legend()
                legend_2d.set_title("Energy [keV]", prop = {'size':10})
                plt.show()
            else:
                plt.plot(x_arr, n_yield, 'o', label='data')
                plt.plot(x_range, fit_z, '-', label='fit')
                plt.xlabel(self.data[dataset_label].columns[x_index]) 
                plt.ylabel(self.data[dataset_label].columns[z_index])
                plt.title(dataset_label)
                plt.legend()
                plt.show()

            return parameters, x_range, fit_z

        if dimension==3:
            #Storing errors and params from fit
            parameters, covariance = curve_fit(model, (x_arr, y_arr), n_yield, p0=init_params, sigma=yield_errors) 
            print(parameters)
            #print(n_yield)

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.set_title(dataset_label)
            ax.set_xlabel(self.data[dataset_label].columns[x_index])
            ax.set_ylabel(self.data[dataset_label].columns[y_index])
            ax.set_zlabel('Yield Density')#self.data[dataset_label].columns[z_index]) 

            reduced_labels = []
            for i in labels:
                if i in reduced_labels:
                    continue
                elif i not in reduced_labels:
                    reduced_labels.append(i)
            for i in np.arange(len(reduced_labels)):
                current_label = reduced_labels[i]
                indices = [j for j, x in enumerate(labels) if x == current_label] 
                arr1 = x_arr[indices]
                arr2 = y_arr[indices]
                arr3 = n_yield[indices]
                ax.scatter(arr1, arr2, arr3, label=current_label)
    
            X, Y = np.meshgrid(x_range, y_range)
            z_range = model((X, Y), *parameters)
            Z = z_range
            ax.plot_surface(X, Y, Z, color='orange')
            ax.legend()
            fig.tight_layout()
            plt.show()

            return parameters, x_range, y_range

    def NRlight_yield_plots(self, func_index, parameters, x_range, y_range):
        #x->x_arr y->y_arr z->yield
        model, init_params, dimension = ms.selector(self, func_index)
        y_arr_min = min(y_range)
        y_arr_max = max(y_range)
        E_interval = (y_arr_max - y_arr_min)/10
        E_arr = np.arange(y_arr_min, y_arr_max, E_interval)

        for i in E_arr: #Iterating through y_arr values
            yields = model((x_range, i), *parameters)
            plt.plot(x_range, yields, label=round(i, 0))
        plt.xlabel('Energy') 
        plt.ylabel('Yield Density')
        legend = plt.legend()
        legend.set_title("E-Field [V/cm]", prop = {'size':10})
        plt.title('E v. Yield Density')
        plt.xscale('log')
        plt.show()

