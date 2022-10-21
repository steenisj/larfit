import numpy as np
import scipy as sci
import pandas as pd
from datasets.lar_dataset import LArDataset as lar
from datasets.lar_dataset import DatasetInfo as dat
from models.default_models import DefaultModels as dm

import sys
sys.path.insert(0,'optimizer') #Change to the /optimizer folder
from toy_model import ToyModel as tm
from classical_optimizer import LeastSquares as ls

#Options for plotting the data:
dataset_label = 'nr_charge' #Successful options: nr_light, nr_charge, alpha_light, alpha_charge
fit_type = 'curve_fit' #Options: curve_fit, minuit, test
plotting_option = False #To determine whether to plot the fits for the rollout 'test' option

#-------------------------------------------------------------------------------#
if dataset_label == 'er_charge' or dataset_label == 'er_light':
    print('#-----------Be careful. This dataset fit is still buggy!-----------#')
else:
    print('#------------------------------------------------------------------#')
#-------------------------------------------------------------------------------#

file_location = "/mnt/c/Users/jahe0/Desktop/Physics Research/Graduate Research/larnest_data/"
data = lar(file_location) #Tell it where to find the data file.
dataset_info = dat(dataset_label)
data_return = lar.data_return(data)
x_index, y_index, z_index, func_index = dataset_info.dataset_info(dataset_label)
optimize = ls(data_return, dataset_label, x_index, y_index, z_index, func_index)




if __name__ == "__main__":
    print('dataset_label: ', dataset_label, '\nfit_type: ', fit_type)
    print('#------------------------------------------------------------------#')
    if fit_type == 'minuit':
        parameters, x_range, y_or_z_range = ls.minuit_fit(optimize)
        if dataset_label == 'nr_charge' or dataset_label == 'nr_light':
            ls.NR_yield_plots(optimize, parameters, x_range, y_or_z_range)

    elif fit_type == 'curve_fit':
        parameters, x_range, y_or_z_range = ls.curve_fit_least_squares(optimize)
        if dataset_label == 'nr_charge' or dataset_label == 'nr_light':
            ls.NR_yield_plots(optimize, parameters, x_range, y_or_z_range)

    elif fit_type == 'test':
        ls.parabola_test(optimize, dataset_label, x_index, y_index, z_index, func_index)

        toy_model = tm(dataset_label, func_index, x_index, y_index, z_index)
        x_data, y_data, z_data = tm.toy_data_generator(toy_model)        
        tm.param_cycler(toy_model, x_data, y_data, z_data, plotting_option)

    else:
        print('Error: Fit type not found!')