from dataclasses import dataclass
from turtle import color
from zlib import Z_BEST_COMPRESSION
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
from iminuit import cost, Minuit, describe
from classical_optimizer import LeastSquares as ls
import random

import sys
sys.path.insert(0,'models') #Change to the /models folder
from default_models import DefaultModels as dm
from default_models import ModelSelector as ms


class ToyModel:
    def __init__(self, dataset_label, func_index, x_index, y_index, z_index, data, optimize):
        #Class attributes for LeastSquares and LArDataset
        self.data = data
        self.optimize = optimize

        self.dataset_label = dataset_label
        self.func_index = func_index
        self.x_index = x_index
        self.y_index = y_index
        self.z_index = z_index

        model, init_params, dimension, param_list, off_parameters = ms.selector(self, func_index)
        self.model = model
        self.init_params = init_params
        self.dimension = dimension
        self.param_list = param_list
        self.off_parameters = off_parameters

        dict = []
        rand_init_params = []
        #Let's make a random offset of the initial parameters to make sure that the fit actually works!
        z_err = 0.1 #--> For tweaking how much we deviate the initial parameters before the fit<---#
        for k in np.arange(len(self.model)):
            for i in np.arange(len(self.init_params[k])):
                rng = np.random.default_rng(1)
                rand_init_params.append(rng.normal(self.init_params[k], z_err))

        dict = ls.dict_maker(optimize)
        self.dict = dict
        self.rand_init_params = rand_init_params
#-----------------------------------------------------------------------------#
    def toy_data_generator(self, parameter_dict_arrays, m):
        #print(len(self.model))
        #print(m)
        if m+1 == len(self.model) and len(self.model)>1: #for er_charge
            alpha = self.model[0](y_data, *parameter_dict_arrays[0]['alpha']) #y_data --> specific for the er_charge ... this is hard-coded
            beta = self.model[1](y_data, *parameter_dict_arrays[1]['beta'])
            gamma = self.model[2](y_data, *parameter_dict_arrays[2]['gamma'])
            doke_birks = self.model[3](y_data, *parameter_dict_arrays[3]['doke_birks'])

            inst = dm
            inst.alpha = alpha
            inst.beta = beta
            inst.gamma = gamma
            inst.doke_birks = doke_birks 

        num_data_points = 200
        x_data = np.arange(0,1000)
        y_data = np.arange(0,1000)

        possible_x_data = np.arange(0,1000)
        possible_y_data = np.arange(0,1000)

        #One of the arrays in x_data_array
        x_data = []
        y_data = []

        for i in np.arange(num_data_points):
            random_num_x = random.choice(possible_x_data)
            random_num_y = random.choice(possible_y_data)
            x_data.append(random_num_x.astype(int))
            y_data.append(random_num_y.astype(int))

        x_data = np.array(x_data)
        y_data = np.array(y_data)

        if self.dimension[m] == 2:
            #print('Z DATA ACHIEVED')
            #print(self.init_params[m])
            z_data = self.model[m](x_data, *self.init_params[m])

            print(z_data)

            rng = np.random.default_rng(1)
            z_err = 1
            rand_z_data = rng.normal(z_data, z_err)

            #plt.scatter(x_data, rand_z_data)
            #plt.show()

        elif self.dimension[m] == 3 and len(self.model)>1 and m+1==len(self.model):
            z_data = inst.self.model[m]((x_data, y_data), *self.init_params[m])

            rng = np.random.default_rng(1)
            z_err = 1
            rand_z_data = rng.normal(z_data, z_err)

        elif self.dimension[m] == 3 and len(self.model)==1:
            z_data = self.model[m]((x_data, y_data), *self.init_params[m])

            rng = np.random.default_rng(1)
            z_err = 1
            rand_z_data = rng.normal(z_data, z_err)

        return x_data, y_data, rand_z_data
#-----------------------------------------------------------------------------#
    def minuit_data_fitter(self, x_data, y_data, z_data):
        for m in np.arange(len(self.model)):
            used_model = self.model[m]

            if self.dimension[m] == 2:
                def LSQ(*args):
                    return np.sum((np.array(z_data) - used_model(np.array(x_data), *args)) ** 2)

            elif self.dimension == 3:
                def LSQ(*args):
                    return np.sum((np.array(z_data) - used_model((np.array(x_data), np.array(y_data)), *args)) ** 2)        

            print('init_params: ', self.init_params[m])
            minuit = Minuit(LSQ, name=self.param_list[m], **self.dict[m], pedantic=False)

            minuit.get_param_states()
            minuit.migrad()
            fit_values = minuit.values

            param_values = []
            for value in minuit.values:
                #print(minuit.values[value])
                param_values.append(minuit.values[value])
            print('Parameters: ', param_values) 

            self.toy_plotter(x_data, y_data, z_data, param_values, m)       
#-----------------------------------------------------------------------------#
    def toy_plotter(self, x_data, y_data, z_data, param_values, m):
        #for m in np.arange(len(self.model)):
        x_range, y_range = ls.range_generator(self, x_data, y_data)
        if self.dimension[m] == 2:
            fit_z = self.model[m](x_range, *param_values) #2d fit stuff

            plt.scatter(x_data, z_data, label='Toy Data')
            plt.plot(x_range, fit_z, '-', label='fit', color='orange')
            plt.legend()
            plt.xlabel(self.x_index)
            plt.ylabel(self.z_index)
            plt.title(self.dataset_label)
            plt.show()

        if self.dimension[m] == 3:
            X, Y = np.meshgrid(x_range, y_range)
            Z_fit = self.model[m]((X,Y), *param_values)

            fig = plt.figure()
            ax = fig.gca(projection='3d')
            ax.scatter(x_data, y_data, z_data)
            ax.set_title(self.dataset_label)
            ax.set_xlabel(self.x_index)
            ax.set_ylabel(self.y_index)
            ax.set_zlabel(self.z_index)
            ax.plot_surface(X, Y, Z_fit, color='orange')
            #ax.legend()
            fig.tight_layout()
            plt.show()
#-----------------------------------------------------------------------------#
    def param_rollout(self, x_data, y_data, z_data, off_parameters, param_dict, plotting_option, m, parameter_dict_arrays):
        #Making a dict to force the fit parameters constant
        param_keys = list(self.dict[m].keys())
        fix_keys = []
        truths = []
        dict_copy = self.dict[m].copy()
        used_model = self.model[m]

        if m+1 == len(self.model) and len(self.model)>1: #for er_charge
                alpha = self.model[0](y_data, *parameter_dict_arrays[0]['alpha']) #y_data --> specific for the er_charge ... this is hard-coded
                beta = self.model[1](y_data, *parameter_dict_arrays[1]['beta'])
                gamma = self.model[2](y_data, *parameter_dict_arrays[2]['gamma'])
                doke_birks = self.model[3](y_data, *parameter_dict_arrays[3]['doke_birks'])

                inst = dm
                inst.alpha = alpha
                inst.beta = beta
                inst.gamma = gamma
                inst.doke_birks = doke_birks 

        for i in np.arange(len(param_keys)):
            if 'fix_' in param_keys[i]:
                continue
            #print(param_keys[i])
            fix_name = 'fix_' + param_keys[i]
            if len(off_parameters) != 0:
                if param_keys[i] in off_parameters.keys():
                    dict_copy[fix_name] = True
                    dict_copy[param_keys[i]] = off_parameters[param_keys[i]]
                elif param_keys[i] not in off_parameters.keys():
                    dict_copy[fix_name] = False
                    if len(param_dict) != 0:
                        dict_copy[param_keys[i]] = param_dict[param_keys[i]]
                    else:
                        continue
            else:
                dict_copy[fix_name] = False
                if len(param_dict) != 0:
                    dict_copy[param_keys[i]] = param_dict[param_keys[i]]
                else:
                    continue

        if self.dimension[m] == 2: ##This is all redundant! Merge with the fitting function above!!!!
            def LSQ(*args):
                return np.sum((np.array(z_data) - used_model(np.array(x_data), *args)) ** 2)

        elif self.dimension[m] == 3 and len(self.model)>1 and m+1==len(self.model):
            def LSQ(*args):
                return np.sum((np.array(z_data) - inst.used_model((np.array(x_data), np.array(y_data)), *args)) ** 2)

        elif self.dimension[m] == 3:
            def LSQ(*args):
                return np.sum((np.array(z_data) - used_model((np.array(x_data), np.array(y_data)), *args)) ** 2)
        minuit = Minuit(LSQ, name=self.param_list[m], **dict_copy, pedantic=False)
        minuit.get_param_states()
        minuit.migrad()
        fit_values = minuit.values

        param_values = []
        param_dict = {}
        for value in minuit.values:
            param_dict[value] = minuit.values[value]
            param_values.append(minuit.values[value])
        #print('Parameters: ', param_values) 
        for n in param_dict.keys():
            dict_copy[n] = round(dict_copy[n], 5)
            param_dict[n] = round(param_dict[n], 5)
        print('\n Initial_Dict', dict_copy)
        print('Fit_Params: ', param_dict, '\n')

        if plotting_option == True:
            self.toy_plotter(x_data, y_data, z_data, param_values, m) 

        return param_values, param_dict
#-----------------------------------------------------------------------------#
    def param_cycler(self, plotting_option):
        parameter_values_arrays = []
        parameter_dict_arrays = []

        for m in np.arange(len(self.model)):
            #print('m: ', m)
            #print(self.model[m])
            parameters = []
            parameters_dictionary = {}
            feed_off_params = self.off_parameters[m].copy()
            off_list = list(self.off_parameters[m].keys())
            cycle_count = 0   

            x_data, y_data, z_data = self.toy_data_generator(parameter_dict_arrays, m) 
            print(x_data, y_data, z_data)   

            i=-1
            while True:
                i += 1
                #print(i)
                newest_param = self.param_list[m][i]
                param_values, param_dict = self.param_rollout(x_data, y_data, z_data, feed_off_params, parameters_dictionary, plotting_option, m, parameter_dict_arrays)
                parameters = param_values
                parameters_dictionary = param_dict

                parameter_values_arrays.append(param_values)
                parameter_dict_arrays.append(param_dict)

                if cycle_count >= 100:
                    print("#------------------------------------------------------------------#")
                    print("\n Fit attempted too many times. Please re-run!\n")
                    print("#------------------------------------------------------------------#")
                    return 0
                elif param_dict[newest_param] == 0:
                    i = -1
                    cycle_count += 1 #To tell us how many times we have this error
                    feed_off_params = self.off_parameters[m].copy()
                    #print("Test: ", feed_off_params)
                    print("Fitting error, attempting to fit again.")
                    #return 0
                elif i < len(off_list):
                    #print(off_list[i])
                    del feed_off_params[off_list[i]] #Remove the entry from the dict
                else:
                    return 0
