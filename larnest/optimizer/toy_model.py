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
    def __init__(self, dataset_label, func_index, x_index, y_index, z_index):
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

        #Let's make a random offset of the initial parameters to make sure that the fit actually works!
        z_err = 0.1 #--> For tweaking how much we deviate the initial parameters before the fit<---#
        for i in np.arange(len(self.init_params)):
            rng = np.random.default_rng(1)
            self.rand_init_params = rng.normal(self.init_params, z_err)

        dict = ls.dict_maker(self, self.param_list, self.rand_init_params)
        self.dict = dict
#-----------------------------------------------------------------------------#
    def toy_data_generator(self):
        #print(dict)
        num_data_points = 200
        x_data = np.arange(0,1000)
        y_data = np.arange(0,1000)

        #print(type(x_data[0]))

        possible_x_data = np.arange(0,1000)
        possible_y_data = np.arange(0,1000)

        x_data = []
        y_data = []

        for i in np.arange(num_data_points):
            random_num_x = random.choice(possible_x_data)
            random_num_y = random.choice(possible_y_data)
            x_data.append(random_num_x.astype(int))
            y_data.append(random_num_y.astype(int))

        x_data = np.array(x_data)
        y_data = np.array(y_data)

        if self.dimension == 2 and self.dataset_label != 'er_charge':
            z_data = self.model(x_data, *self.init_params)

            rng = np.random.default_rng(1)
            z_err = 1
            rand_z_data = rng.normal(z_data, z_err)

            #plt.scatter(x_data, rand_z_data)
            #plt.show()

            return x_data, y_data, z_data

        elif self.dimension == 3 and self.dataset_label != 'er_charge':
            z_data = self.model((x_data, y_data), *self.init_params)

            rng = np.random.default_rng(1)
            z_err = 1
            rand_z_data = rng.normal(z_data, z_err)


            return x_data, y_data, z_data

        elif self.dataset_label == 'er_charge':
            pass
#-----------------------------------------------------------------------------#
    def minuit_data_fitter(self, x_data, y_data, z_data):
        #x_data, y_data, z_data = self.toy_data_generator()
        #x_range, y_range = ls.range_generator(ls, self.dataset_label, x_data, y_data)

        if self.dimension == 2:
            def LSQ(*args):
                return np.sum((np.array(z_data) - self.model(np.array(x_data), *args)) ** 2)

        elif self.dimension == 3:
            def LSQ(*args):
                return np.sum((np.array(z_data) - self.model((np.array(x_data), np.array(y_data)), *args)) ** 2)        

        print('init_params: ', self.init_params)
        minuit = Minuit(LSQ, name=self.param_list, **self.dict, pedantic=False)

        minuit.get_param_states()
        minuit.migrad()
        fit_values = minuit.values

        param_values = []
        for value in minuit.values:
            #print(minuit.values[value])
            param_values.append(minuit.values[value])
        print('Parameters: ', param_values) 

        self.toy_plotter(x_data, y_data, z_data, param_values)       
#-----------------------------------------------------------------------------#
    def toy_plotter(self, x_data, y_data, z_data, param_values):
        x_range, y_range = ls.range_generator(ls, self.dataset_label, x_data, y_data)
        if self.dimension == 2:
            fit_z = self.model(x_range, *param_values) #2d fit stuff

            plt.scatter(x_data, z_data, label='Toy Data')
            plt.plot(x_range, fit_z, '-', label='fit', color='orange')
            plt.legend()
            plt.xlabel(self.x_index) 
            plt.ylabel(self.z_index)
            plt.title(self.dataset_label)
            plt.show()

        if self.dimension == 3:
            X, Y = np.meshgrid(x_range, y_range)
            Z_fit = self.model((X,Y), *param_values)

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
    def param_rollout(self, x_data, y_data, z_data, off_parameters, param_dict, plotting_option):
        #Making a dict to force the fit parameters constant
        param_keys = list(self.dict.keys())
        fix_keys = []
        truths = []
        dict_copy = self.dict

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

        if self.dimension == 2:
            def LSQ(*args):
                return np.sum((np.array(z_data) - self.model(np.array(x_data), *args)) ** 2)

        elif self.dimension == 3:
            def LSQ(*args):
                return np.sum((np.array(z_data) - self.model((np.array(x_data), np.array(y_data)), *args)) ** 2)
        minuit = Minuit(LSQ, name=self.param_list, **dict_copy, pedantic=False)
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
            self.toy_plotter(x_data, y_data, z_data, param_values) 

        return param_values, param_dict
#-----------------------------------------------------------------------------#
    def param_cycler(self, off_parameters, x_data, y_data, z_data, plotting_option):
        parameters = []
        parameters_dictionary = {}
        feed_off_params = off_parameters.copy()
        off_list = list(off_parameters.keys())
        cycle_count = 0

        i=-1
        while True:
            i += 1
            #print(i)
            newest_param = self.param_list[i]
            param_values, param_dict = self.param_rollout(x_data, y_data, z_data, feed_off_params, parameters_dictionary, plotting_option)
            parameters = param_values
            parameters_dictionary = param_dict

            if cycle_count >= 100:
                print("#------------------------------------------------------------------#")
                print("\n Fit attempted too many times. Please re-run!\n")
                print("#------------------------------------------------------------------#")
                return 0
            elif param_dict[newest_param] == 0:
                i = -1
                cycle_count += 1 #To tell us how many times we have this error
                feed_off_params = off_parameters.copy()
                #print("Test: ", feed_off_params)
                print("Fitting error, attempting to fit again.")
                #return 0
            elif i < len(off_list):
                #print(off_list[i])
                del feed_off_params[off_list[i]] #Remove the entry from the dict
            else:
                return 0