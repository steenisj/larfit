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

        model, init_params, dimension, param_list = ms.selector(self, func_index)
        self.model = model
        self.init_params = init_params
        self.dimension = dimension
        self.param_list = param_list

        dict = ls.dict_maker(self, self.param_list, self.init_params)
        self.dict = dict

    def toy_data_generator(self):
        #print(dict)
        num_data_points = 1000
        x_data = np.arange(0,1000)
        y_data = np.arange(0,1000)

        print(type(x_data[0]))

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

        if self.dimension == 2:
            z_data = self.model(x_data, *self.init_params)

            rng = np.random.default_rng(1)
            z_err = 1
            rand_z_data = rng.normal(z_data, z_err)

            #plt.scatter(x_data, rand_z_data)
            #plt.show()

            return x_data, y_data, z_data

        if self.dimension == 3:
            z_data = self.model((x_data, y_data), *self.init_params)

            rng = np.random.default_rng(1)
            z_err = 1
            rand_z_data = rng.normal(z_data, z_err)


            return x_data, y_data, z_data

    def minuit_data_fitter(self):
        x_data, y_data, z_data = self.toy_data_generator()
        x_range, y_range = ls.range_generator(ls, self.dataset_label, x_data, y_data)

        if self.dimension == 2:
            def LSQ(*args):
                return np.sum((np.array(z_data) - self.model(np.array(x_data), *args)) ** 2)

        elif self.dimension == 3:
            def LSQ(*args):
                return np.sum((np.array(z_data) - self.model((np.array(x_data), np.array(y_data)), *args)) ** 2)

        minuit = Minuit(LSQ, name=self.param_list, **self.dict, pedantic=False)        

        #Let's make a random offset of the initial parameters to make sure that the fit actually works!
        z_err = 1
        for i in np.arange(len(self.init_params)):
            rng = np.random.default_rng(1)
            self.init_params = rng.normal(self.init_params, z_err)

        print('init_params: ', self.init_params)

        minuit.get_param_states()
        minuit.migrad()
        fit_values = minuit.values

        param_values = []
        for value in minuit.values:
            #print(minuit.values[value])
            param_values.append(minuit.values[value])
        print('Parameters: ', param_values)        

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
