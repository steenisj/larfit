'''
This script will allow us to perform least squares fitting on data.
'''

from dataclasses import dataclass
from turtle import color
from zlib import Z_BEST_COMPRESSION
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from iminuit import cost, Minuit

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
#-----------------------------------------------------------------------------#
    def data_initializer(self, dataset_label, x_index, y_index, z_index, func_index): #Gives us the relevant data and errors
        x_arr = pd.to_numeric(self.data[dataset_label][x_index]) 
        y_arr = pd.to_numeric(self.data[dataset_label][y_index]) 
        n_yield = pd.to_numeric(self.data[dataset_label][z_index])
        labels = self.data[dataset_label]['dataset']

        yield_err_index = 'yield_sl'
        if x_index == 'energy':
            x_err_index = 'energy_sl'
        if x_index == 'field':
            x_err_index = 'field_sl'
        yield_errors = pd.to_numeric(self.data[dataset_label][yield_err_index]) 
        x_arr_errors = pd.to_numeric(self.data[dataset_label][x_err_index])

        return x_arr, y_arr, n_yield, labels, yield_err_index, x_err_index, yield_errors, x_arr_errors
#-----------------------------------------------------------------------------#
    def range_generator(self, dataset_label, x_arr, y_arr): #Gives us our ranges for the fit plots
        x_min = np.min(x_arr)
        x_max = np.max(x_arr)
        y_min = np.min(y_arr)
        y_max = np.max(y_arr)
        
        sep = 0.1
        num_x = (x_max-x_min)/sep

        if num_x/1000 > 1 and dataset_label == 'er_charge':
            sep = 1

        x_range = np.arange(x_min, x_max, sep)
        num_x = (x_max-x_min)/sep
        y_interval = (y_max-y_min)/num_x
        y_range = np.arange(y_min, y_max, y_interval)
        return x_range, y_range
#-----------------------------------------------------------------------------#
    def fit_plotter(self, dimension, plot_arrays, index_arr, dataset_label, labels): #Plotting the fits and the data
        if dimension == 2:
            x_arr, y_arr, n_yield, x_range, y_range, fit_z, yield_errors, x_arr_errors = plot_arrays
            x_index, y_index, z_index = index_arr
        
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
                    new_yield_errors = yield_errors[indices]
                    new_x_arr_errors = x_arr_errors[indices]
                    plt.errorbar(new_Efield_arr, new_yield_arr, label = current_energy, xerr=new_x_arr_errors, yerr=new_yield_errors, fmt='o')
                plt.plot(x_range, fit_z, '-', label='fit')
                plt.xlabel(x_index) 
                plt.ylabel(z_index)
                plt.title(dataset_label)
                legend_2d = plt.legend()
                legend_2d.set_title("Energy [keV]", prop = {'size':10})
                plt.show()

            elif dataset_label == 'er_charge':
                reduced_labels = []
                for i in labels: #Iterating over the energies
                    if i in reduced_labels:
                        continue
                    elif i not in reduced_labels:
                        reduced_labels.append(i)
                for i in np.arange(len(reduced_labels)):
                    current_label = reduced_labels[i]
                    indices = [j for j, x in enumerate(labels) if x == current_label] 
                    new_Efield_arr = x_arr[indices]
                    new_yield_arr = n_yield[indices]
                    new_yield_errors = yield_errors[indices]
                    new_x_arr_errors = x_arr_errors[indices]
                    #plt.plot(new_Efield_arr, new_yield_arr, 'o', label=current_energy)
                    plt.errorbar(new_Efield_arr, new_yield_arr, label = current_label, xerr=new_x_arr_errors, yerr=new_yield_errors, fmt='o')
                plt.plot(x_range, fit_z, '-', label='fit')
                plt.xlabel(x_index) 
                plt.ylabel(z_index)
                plt.title(dataset_label)
                legend_2d = plt.legend()
                #legend_2d.set_title("Energy [keV]", prop = {'size':10})
                plt.show()

            else:
                plt.plot(x_arr, n_yield, 'o', label='data')
                plt.plot(x_range, fit_z, '-', label='fit')
                plt.xlabel(x_index) 
                plt.ylabel(z_index)
                plt.title(dataset_label)
                plt.legend()
                plt.show()

        if dimension == 3:
            x_arr, y_arr, n_yield, X, Y, Z_fit = plot_arrays
            x_index, y_index, z_index = index_arr

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.set_title(dataset_label)
            ax.set_xlabel(x_index)
            ax.set_ylabel(y_index)
            ax.set_zlabel(z_index)

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

            ax.plot_surface(X, Y, Z_fit, color='orange')
            ax.legend()
            fig.tight_layout()
            plt.show()
#-----------------------------------------------------------------------------#
    def dict_maker(self, param_list, init_params): #To make two dictionaries: One for init_params, and the other for the list of parameter names
        #For the initial values:
        dict = {}
        keys = param_list
        values = init_params
        for i in np.arange(len(keys)):
            dict[keys[i]] = values[i]
        #print(dict)
        return dict
#-----------------------------------------------------------------------------#
    def minuit_fit(self, dataset_label, x_index, y_index, z_index, func_index): #Fitting the data with minuit
        model, init_params, dimension, param_list = ms.selector(self, func_index)
        x_arr, y_arr, n_yield, labels, yield_err_index, x_err_index, yield_errors, x_arr_errors = LeastSquares.data_initializer(self, dataset_label, x_index, y_index, z_index, func_index)
        x_range, y_range = LeastSquares.range_generator(self, dataset_label, x_arr, y_arr)

        init_dict = LeastSquares.dict_maker(self, param_list, init_params)

        if dimension == 2 and dataset_label != 'er_charge':
            if dataset_label == 'alpha_light':
                def LSQ(A, B, C, D, E, F, G, H, I, J, K, L, M):
                    return np.sum((np.array(n_yield) - model(np.array(x_arr), A, B, C, D, E, F, G, H, I, J, K, L, M)) ** 2)

            elif dataset_label == 'alpha_charge':
                def LSQ(A, B, C, D, E, F, G, H, I, J): 
                    return np.sum((np.array(n_yield) - model(np.array(x_arr), A, B, C, D, E, F, G, H, I, J)) ** 2)

            elif dataset_label == 'nr_total':
                def LSQ(alpha, beta): 
                    return np.sum((np.array(n_yield) - model(np.array(x_arr), alpha, beta)) ** 2)

            else:
                print('Error: dataset not entered into classical_optimizer.py')
                return 0
            #c = cost.LeastSquares(x_arr, n_yield, 0, model) #Change from least squares // '''yield_errors'''

            minuit = Minuit(LSQ, **init_dict, pedantic=False)        
        
            minuit.get_param_states()
            minuit.migrad()
            fit_values = minuit.values

            param_values = []
            for value in minuit.values:
                #print(minuit.values[value])
                param_values.append(minuit.values[value])
            print('Parameters: ', param_values)        

            fit_z = model(x_range, *param_values) #2d fit stuff

            plot_arrays = x_arr, y_arr, n_yield, x_range, y_range, fit_z, yield_errors, x_arr_errors
            index_arr = x_index, y_index, z_index
            LeastSquares.fit_plotter(self, dimension, plot_arrays, index_arr, dataset_label, labels)

        elif dimension == 3  and dataset_label != 'er_charge':
            if dataset_label == 'nr_charge':
                def LSQ(gamma, delta, epsilon, zeta, eta):
                    return np.sum((np.array(n_yield) - model((np.array(x_arr), np.array(y_arr)), gamma, delta, epsilon, zeta, eta)) ** 2)

            elif dataset_label == 'nr_light':
                def LSQ(alpha, beta, gamma, delta, epsilon, zeta, eta):
                    return np.sum((np.array(n_yield) - model((np.array(x_arr), np.array(y_arr)), alpha, beta, gamma, delta, epsilon, zeta, eta)) ** 2)

            minuit = Minuit(LSQ, **init_dict, pedantic=False)
            minuit.migrad()

            param_values = []
            for value in minuit.values:
                #print(minuit.values[value])
                param_values.append(minuit.values[value])  

            print('Parameters: ', param_values)      

            X, Y = np.meshgrid(x_range, y_range)
            Z_fit = model((X,Y), *param_values)

            plot_arrays = x_arr, y_arr, n_yield, X, Y, Z_fit
            index_arr = x_index, y_index, z_index
            LeastSquares.fit_plotter(self, dimension, plot_arrays, index_arr, dataset_label, labels)

        #Put an else, error statement here
#-----------------------------------------------------------------------------#
    def curve_fit_least_squares(self, dataset_label, x_index, y_index, z_index, func_index): #Fitting the data with curve_fit()
        model, init_params, dimension, param_list = ms.selector(self, func_index)
        x_arr, y_arr, n_yield, labels, yield_err_index, x_err_index, yield_errors, x_arr_errors = LeastSquares.data_initializer(self, dataset_label, x_index, y_index, z_index, func_index)
        x_range, y_range = LeastSquares.range_generator(self, dataset_label, x_arr, y_arr)

        if dimension==2: #For two dimensional plots
            #print(yield_errors)
            if dataset_label == 'alpha_charge':
                parameters, covariance = curve_fit(model, x_arr, n_yield, p0=init_params, maxfev=8000) #Storing errors and params from fit
            elif dataset_label == 'er_total':
                parameters, covariance = curve_fit(model, x_arr, n_yield, p0=init_params, sigma=yield_errors, maxfev=8000)
            else:
                parameters, covariance = curve_fit(model, x_arr, n_yield, p0=init_params, sigma=yield_errors, maxfev=8000) #Storing errors and params from fit
            print('Parameters: ', parameters)

            fit_z = model(x_range, *parameters)
            #print(fit_z)
            plot_arrays = x_arr, y_arr, n_yield, x_range, y_range, fit_z, yield_errors, x_arr_errors
            index_arr = x_index, y_index, z_index
            LeastSquares.fit_plotter(self, dimension, plot_arrays, index_arr, dataset_label, labels)

            return parameters, x_range, fit_z

        if dimension==3: #For 3 dimensional plots
            #Storing errors and params from fit
            #x_y_points = (x_arr, y_arr) #Clean up and rename!
            #z_points = n_yield
            #new_labels = labels

            if dataset_label == 'er_charge':
                alpha, beta, gamma, doke_birks, alpha_params, beta_params, gamma_params, doke_birks_params = LeastSquares.ERChargeParamPuller(self, y_arr, n_yield, yield_errors)

                inst = dm
                inst.alpha = alpha
                inst.beta = beta
                inst.gamma = gamma
                inst.doke_birks = doke_birks
                parameters, covariance = curve_fit(inst.GetERElectronYields, (x_arr, y_arr), n_yield, sigma=yield_errors, p0=init_params)
                print('Parameters: ', parameters)

            else:
                parameters, covariance = curve_fit(model, (x_arr, y_arr), n_yield, sigma=yield_errors, p0=init_params)
                print('Parameters: ', parameters)

            X, Y = np.meshgrid(x_range, y_range)
            if dataset_label == 'er_charge':
                new_alpha, new_beta, new_gamma, new_DB = LeastSquares.ERChargeParamGetter(self, alpha_params, beta_params, gamma_params, doke_birks_params, Y)
                inst.alpha = new_alpha
                inst.beta = new_beta
                inst.gamma = new_gamma
                inst.doke_birks = new_DB
                z_range = inst.GetERElectronYields((X, Y), *parameters)
            else:
                z_range = model((X, Y), *parameters)
            #print(len((X,Y)))
            Z_fit = z_range

            plot_arrays = x_arr, y_arr, n_yield, X, Y, Z_fit
            index_arr = x_index, y_index, z_index
            LeastSquares.fit_plotter(self, dimension, plot_arrays, index_arr, dataset_label, labels)

            return parameters, x_range, y_range
#-----------------------------------------------------------------------------#
    def NR_yield_plots(self, func_index, parameters, x_range, y_range, dataset_label):
        #x->x_arr y->y_arr z->yield
        if dataset_label != 'nr_charge' and dataset_label != 'nr_light':
            print('Error: Wrong dataset selected. Check NR_yield_plots requirements!')
            return 0
        model, init_params, dimension, param_list = ms.selector(self, func_index)
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
#-----------------------------------------------------------------------------#
    def ERChargeParamPuller(self, y_arr, n_yield, yield_errors): #To reconfigure the er_charge params for display purposes only
        alpha_model, alpha_init_params, alpha_dim, alpha_param_list = ms.selector(self, 8)
        alpha_params, a_cov = curve_fit(alpha_model, y_arr, n_yield, sigma=yield_errors, p0=alpha_init_params)
        #print(alpha_params)
        alpha = dm.GetERElectronYieldsAlpha(y_arr, *alpha_params)
        #print(alpha)

        beta_model, beta_init_params, beta_dim, beta_param_list = ms.selector(self, 9)
        beta_params, b_cov = curve_fit(beta_model, y_arr, n_yield, sigma=yield_errors, p0=beta_init_params)
        beta = dm.GetERElectronYieldsBeta(y_arr, *beta_params)
       #print(beta)

        gamma_model, gamma_init_params, gamma_dim, gamma_param_list = ms. selector(self, 10)
        gamma_params, g_cov = curve_fit(gamma_model, y_arr, n_yield, sigma=yield_errors, p0=gamma_init_params)
        gamma = dm.GetERElectronYieldsGamma(y_arr, *gamma_params)
        #print(gamma)

        DB_model, DB_init_params, DB_dim, DB_param_list = ms.selector(self, 11)
        doke_birks_params, db_cov = curve_fit(DB_model, y_arr, n_yield, sigma=yield_errors, p0=DB_init_params) 
        doke_birks = dm.GetERElectronYieldsDokeBirks(y_arr, *doke_birks_params)
        #print(doke_birks)
        return alpha, beta, gamma, doke_birks, alpha_params, beta_params, gamma_params, doke_birks_params
#-----------------------------------------------------------------------------#
    def ERChargeParamGetter(self, alpha_params, beta_params, gamma_params, doke_birks_params, Y): #For obtaining the models using the params generated from ERChargeParamPuller (gives a new array of the same length as the plots)
        alpha = dm.GetERElectronYieldsAlpha(Y, *alpha_params)
        beta = dm.GetERElectronYieldsBeta(Y, *beta_params)
        gamma = dm.GetERElectronYieldsGamma(Y, *gamma_params)
        doke_birks = dm.GetERElectronYieldsDokeBirks(Y, *doke_birks_params)
        #print(doke_birks)
        return alpha, beta, gamma, doke_birks
#-----------------------------------------------------------------------------#
    def parabola_test(self, dataset_label, x_index, y_index, z_index, func_index): 
        #Uses minuit to fit a parabola to the data to ensure that it converges in the first place.
        model, init_params, dimension, param_list = ms.selector(self, func_index)
        x_arr, y_arr, n_yield, labels, yield_err_index, x_err_index, yield_errors, x_arr_errors = LeastSquares.data_initializer(self, dataset_label, x_index, y_index, z_index, func_index)
        x_range, y_range = LeastSquares.range_generator(self, dataset_label, x_arr, y_arr)
        init_dict = LeastSquares.dict_maker(self, param_list, init_params)

        if dimension == 2:
            def parab_2d(x_arr, p):
                return p*x_arr**2
            def LSQ(p):
                return np.sum((np.array(n_yield) - parab_2d(np.array(x_arr), p)) ** 2)
            minuit = Minuit(LSQ, p=-1, pedantic=False)

        if dimension == 3:
            def parab_3d(X, px, py):
                (x_arr, y_arr) = X
                return px*x_arr**2 + py*y_arr**2
            def LSQ(px, py):
                return np.sum((np.array(n_yield) - parab_3d((np.array(x_arr), np.array(y_arr)), px, py)) ** 2)
            minuit = Minuit(LSQ, px=-1, py=-1, pedantic=False)       
    
        minuit.get_param_states()
        minuit.migrad()
        fit_values = minuit.values

        param_values = []
        for value in minuit.values:
            #print(minuit.values[value])
            param_values.append(minuit.values[value])
        print('Parameters: ', param_values)

        if dimension == 2:
            fit_z = parab_2d(x_range, *param_values) #2d fit stuff
            plot_arrays = x_arr, y_arr, n_yield, x_range, y_range, fit_z, yield_errors, x_arr_errors
            index_arr = x_index, y_index, z_index
            LeastSquares.fit_plotter(self, dimension, plot_arrays, index_arr, dataset_label, labels)
        elif dimension == 3: 
            X, Y = np.meshgrid(x_range, y_range) #3d fit stuff
            Z_fit = parab_3d((X,Y), *param_values)
            plot_arrays = x_arr, y_arr, n_yield, X, Y, Z_fit
            index_arr = x_index, y_index, z_index
            LeastSquares.fit_plotter(self, dimension, plot_arrays, index_arr, dataset_label, labels)
        