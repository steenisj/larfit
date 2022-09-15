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
from classical_optimizer import LeastSquares as ls

import sys
sys.path.insert(0,'models') #Change to the /models folder
from default_models import DefaultModels as dm
from default_models import ModelSelector as ms


class ToyModel:
    def __init__(self, dataset_label, func_index):
        self.dataset_label = dataset_label
        self.func_index = func_index
        self.model, self.init_params, self.dimension, self.param_list = ms.selector(self, func_index)
        self.dict = ls.dict_maker(self, self.param_list, self.init_params)

    def toy_data_generator(self):
        #print(dict)
        x_data = np.arange(0,1000)
        y_data = np.arange(0,1000)

        if self.dimension == 2:
            z_data = self.model(x_data, **self.dict)

            rng = np.random.default_rng(1)
            z_err = 1
            rand_z_data = rng.normal(z_data, z_err)

            #plt.scatter(x_data, rand_z_data)
            #plt.show()

            return x_data, y_data, z_data

        if self.dimension == 3:
            z_data = self.model((x_data, y_data), **self.dict)

            rng = np.random.default_rng(1)
            z_err = 1
            rand_z_data = rng.normal(z_data, z_err)


            return x_data, y_data, z_data

    def minuit_data_fitter(self):
        x_data, y_data, z_data = self.toy_data_generator()
        x_range, y_range = ls.range_generator(ls, self.dataset_label, x_data, y_data)

        dict_keys = list(self.dict.keys())
        print(dict_keys)

        '''def LSQ(*args):
            return np.sum((np.array(z_data) - self.model(np.array(x_data), *args)) ** 2)

        minuit = Minuit(LSQ, **self.dict, pedantic=False)        
        
        minuit.get_param_states()
        minuit.migrad()
        fit_values = minuit.values

        param_values = []
        for value in minuit.values:
            #print(minuit.values[value])
            param_values.append(minuit.values[value])
        print('Parameters: ', param_values)        

        fit_z = self.model(x_range, *param_values) #2d fit stuff'''

        '''plot_arrays = x_data, y_data, fit_z, x_range, y_range, fit_z, yield_errors, x_arr_errors
        index_arr = x_index, y_index, z_index
        LeastSquares.fit_plotter(self, dimension, plot_arrays, index_arr, dataset_label, labels)'''