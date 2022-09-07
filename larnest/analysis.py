import numpy as np
import scipy as sci
import pandas as pd
from datasets.lar_dataset import LArDataset as lar
from optimizer.classical_optimizer import MinuitFits
#from larnest.optimizer.classical_optimizer import MinuitFits
from optimizer.classical_optimizer import LeastSquares as ls
from optimizer.classical_optimizer import MinuitFits as mf
#from optimizer.classical_optimizer import FitRunner as fr
from models.default_models import DefaultModels as dm

#Options for plotting the data: THIS IS WHAT THE USER MODIFIES
file_location = "/mnt/c/Users/jahe0/Desktop/Physics Research/Graduate Research/larnest_data/"
data = lar(file_location) #Tell it where to find the data file.
dataset_label = 'nr_total' #For indexing within datasets
fit_type = 'minuit'

### Neutron Recoil Models ####
if dataset_label == 'nr_charge':
    x_index = 'energy' #Energy
    y_index = 'field' #Efield
    z_index = 'yield' #Yield (energy-normalized)
    func_index = 1 #What model do we want to use?

elif dataset_label == 'nr_light':
    x_index = 'energy' #Energy
    y_index = 'field' #Efield
    z_index = 'yield' #Yield (energy-normalized)
    func_index = 2 #What model do we want to use?

elif dataset_label == 'nr_total':
    x_index = 'energy' #Energy
    y_index = 'field' #Efield
    z_index = 'yield' #Yield
    func_index = 3 #What model do we want to use?

### Alpha Models ####
elif dataset_label == 'alpha_light':
    x_index = 'field' #Efield
    y_index = 'energy' #Energy
    z_index = 'yield' #Yields
    func_index = 4 #What model do we want to use?

elif dataset_label == 'alpha_charge':
    x_index = 'field' #Efield
    y_index = 'energy' #Energy
    z_index = 'yield' #Yields
    func_index = 5 #What model do we want to use?

### Electron Recoil Models ####
elif dataset_label == 'er_charge': #There is an issue with the curve_fit() for this!
    x_index = 'energy' #Energy
    y_index = 'field' #Efield
    z_index = 'yield' #Yield (energy-normalized)
    func_index = 6 #What model do we want to use?

elif dataset_label == 'er_light': ###Not used yet!
    x_index = 'energy' #Energy
    y_index = 'field' #Efield
    z_index = 'yield' #Yield (energy-normalized)
    func_index = 7 #What model do we want to use?

elif dataset_label == 'er_total': #no data yet!
    x_index = 'energy' #Energy
    y_index = 'field' #Efield
    z_index = 'yield' #Yield (energy-normalized)
    func_index = 12 #What model do we want to use?

else:
    print("ERROR: Wrong type input. Take a peak at analysis.py")


if __name__ == "__main__":
    #lar.print_data(data)
    parameters, x_range, y_or_z_range = ls.curve_fit_least_squares(data, dataset_label, x_index, y_index, z_index, func_index)
    ls.NR_yield_plots(data, func_index, parameters, x_range, y_or_z_range, dataset_label) #Can only be done for 3d data

    #mf.minuit_getter(func_index, dataset_label, x_index, y_index, z_index)

    #ls.minuit_fit(data, dataset_label, x_index, y_index, z_index, func_index)

    #lar.plot_2d_data(data, dataset_label, x_index, y_index)
    #lar.plot_3d_data(data, dataset_label, x_index, y_index, z_index)

    #chosen_class = fr()
    #fit_result = fr.fitter_selector(chosen_class, data, fit_type, dataset_label, x_index, y_index, z_index, func_index)



