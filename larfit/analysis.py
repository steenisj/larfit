'''
To actually run the module
'''
import numpy as np
import scipy as sci
import pandas as pd
from datasets.lar_dataset import LArDataset as lar
from optimizer.classical_optimizer import LeastSquares as ls
from models.default_models import DefaultModels as dm

#Options for plotting the data: THIS IS WHAT THE USER MODIFIES
file_location = "/mnt/c/Users/jahe0/Desktop/Physics Research/Graduate Research/larnest_data/"
data = lar(file_location) #Tell it where to find the data file.
dataset_label = 'nr_light' #For indexing within datasets

if dataset_label == 'nr_light':
    x_index = 1 #Energy
    y_index = 4 #Efield
    z_index = 7 #Yield (non-energy-normalized)
    func_index = 2 #What model do we want to use?

if dataset_label == 'nr_charge':
    x_index = 1 #Energy
    y_index = 4 #Efield
    z_index = 7 #Yield (non-energy-normalized)
    func_index = 1 #What model do we want to use?

if dataset_label == 'nr_total':
    x_index = 1 #Energy
    y_index = 7 #Yield
    z_index = 7 #Dosn't matter
    func_index = 3 #What model do we want to use?


if __name__ == "__main__":
    #lar.print_data(data)
    parameters, x_range, y_range = ls.least_squares(data, dataset_label, x_index, y_index, z_index, func_index)
    #lar.plot_2d_data(data, dataset_label, x_index, y_index)
    #lar.plot_3d_data(data, dataset_label, x_index, y_index, z_index)
    ls.light_yield_plots(data, func_index, parameters, x_range, y_range)


