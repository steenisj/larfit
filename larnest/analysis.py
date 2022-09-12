import numpy as np
import scipy as sci
import pandas as pd
from datasets.lar_dataset import LArDataset as lar
from datasets.lar_dataset import DatasetInfo as dat
#from optimizer.classical_optimizer import MinuitFits
from optimizer.classical_optimizer import LeastSquares as ls
from models.default_models import DefaultModels as dm

#Options for plotting the data:
dataset_label = 'nr_total' #For indexing within datasets; THIS IS THE ONE TO CHANGE
fit_type = 'minuit' #For setting the fit method; THIS IS ANOTHER TO CHANGE

#-------------------------------------------------------------------------------#
if dataset_label == 'er_charge' or dataset_label == 'er_light':
    print('#-----------Be careful. This dataset fit is still buggy!-----------#')
else:
    print('#------------------------------------------------------------------#')
#-------------------------------------------------------------------------------#

file_location = "/mnt/c/Users/jahe0/Desktop/Physics Research/Graduate Research/larnest_data/"
data = lar(file_location) #Tell it where to find the data file.
dataset_info = dat(dataset_label)
x_index, y_index, z_index, func_index = dataset_info.dataset_info(dataset_label)




if __name__ == "__main__":
    print('dataset_label: ', dataset_label, '\nfit_type: ', fit_type)
    print('#------------------------------------------------------------------#')
    if fit_type == 'minuit':
        ls.minuit_fit(data, dataset_label, x_index, y_index, z_index, func_index)
    
    elif fit_type == 'curve_fit':
        parameters, x_range, y_or_z_range = ls.curve_fit_least_squares(data, dataset_label, x_index, y_index, z_index, func_index)
        ls.NR_yield_plots(data, func_index, parameters, x_range, y_or_z_range, dataset_label)