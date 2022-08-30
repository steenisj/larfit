import numpy as np
import scipy as sci
import pandas as pd
from datasets.lar_dataset import LArDataset as lar
from optimizer.classical_optimizer import LeastSquares as ls
from models.default_models import DefaultModels as dm

#Options for plotting the data: THIS IS WHAT THE USER MODIFIES
file_location = "/mnt/c/Users/jahe0/Desktop/Physics Research/Graduate Research/larnest_data/"
data = lar(file_location) #Tell it where to find the data file.
dataset_label = 'er_charge' #For indexing within datasets

#For 'er_charge' Model Options (only keep one True at a time):
Beta = True
Gamma = False
Doke_Birks = False

### Neutron Recoil Models ####
if dataset_label == 'nr_light':
    x_index = 1 #Energy
    y_index = 4 #Efield
    z_index = 7 #Yield (energy-normalized)
    func_index = 2 #What model do we want to use?

elif dataset_label == 'nr_charge':
    x_index = 1 #Energy
    y_index = 4 #Efield
    z_index = 7 #Yield (energy-normalized)
    func_index = 1 #What model do we want to use?

elif dataset_label == 'nr_total':
    x_index = 1 #Energy
    y_index = 4 #Efield
    z_index = 7 #Yield
    func_index = 3 #What model do we want to use?

### Alpha Models ####
elif dataset_label == 'alpha_light':
    x_index = 4 #Efield
    y_index = 1 #Energy
    z_index = 7 #Yields
    func_index = 4 #What model do we want to use?

elif dataset_label == 'alpha_charge':
    x_index = 4 #Efield
    y_index = 1 #Energy
    z_index = 7 #Yields
    func_index = 5 #What model do we want to use?

### Electron Recoil Models ####
elif dataset_label == 'er_charge' and Beta == True and Gamma == False and Doke_Birks == False:
    x_index = 4 #Energy
    y_index = 1 #Efield
    z_index = 7 #Yield (energy-normalized)
    func_index = 6 #What model do we want to use?

elif dataset_label == 'er_charge' and Beta == False and Gamma == True and Doke_Birks == False:
    x_index = 4 #Energy
    y_index = 1 #Efield
    z_index = 7 #Yield (energy-normalized)
    func_index = 7 #What model do we want to use?

elif dataset_label == 'er_charge' and Beta == False and Gamma == False and Doke_Birks == True:
    x_index = 4 #Energy
    y_index = 1 #Efield
    z_index = 7 #Yield (energy-normalized)
    func_index = 8 #What model do we want to use?

elif dataset_label == 'er_light': ###Not used yet!
    x_index = 1 #Energy
    y_index = 4 #Efield
    z_index = 7 #Yield (energy-normalized)
    func_index = 9 #What model do we want to use?

else:
    print("ERROR: Wrong type input. Take a peak at analysis.py")


if __name__ == "__main__":
    #lar.print_data(data)
    parameters, x_range, y_or_z_range = ls.curve_fit_least_squares(data, dataset_label, x_index, y_index, z_index, func_index)
    #ls.NRlight_yield_plots(data, func_index, parameters, x_range, y_range) #Can only be done for 3d data

    #ls.minuit_fit(data, dataset_label, x_index, y_index, z_index, func_index)
    #lar.plot_2d_data(data, dataset_label, x_index, y_index)
    #lar.plot_3d_data(data, dataset_label, x_index, y_index, z_index)



