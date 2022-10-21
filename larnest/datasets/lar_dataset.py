"""
Code for loading and manipulating 
LAr datasets. 
Edited by JHS 8/1/2022 Modified data file location
"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import csv
from IPython.display import display

class DatasetInfo:
    def __init__(self, dataset_label):
        self.dataset_label = dataset_label

    def dataset_info(self, dataset_label):
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
            func_index = 8 #What model do we want to use?

        else:
            print("ERROR: Wrong type input. Take a peak at analysis.py")

        return x_index, y_index, z_index, func_index

class LArDataset:
    def __init__(self,
        data_dir:   str,
        use_converted:  bool=True,
        use_excluded:   bool=False,
        zero_error_bar_scale:   float=0.0,
    ):
        self.data_dir = data_dir
        self.data = {}
        self.data['nr_total'] = pd.read_csv(
            self.data_dir + "nr_total_yield.csv",
            sep=",",
            header=0,
        )
        self.data['nr_charge'] = pd.read_csv(
            self.data_dir + "nr_charge_yield_alt.csv",
            sep=",",
            header=0,
        )
        self.data['nr_light'] = pd.read_csv(
            self.data_dir + "nr_light_yield.csv",
            sep=",",
            header=0,
        )
        self.data['er_charge'] = pd.read_csv(
            self.data_dir + "er_charge_yield.csv",
            sep=",",
            header=0,
        )
        self.data['er_light'] = pd.read_csv(
            self.data_dir + "er_light_yield.csv",
            sep=",",
            header=0,
        )
        self.data['alpha_charge'] = pd.read_csv(
            self.data_dir + "alpha_charge_yield.csv",
            sep=",",
            header=0,
        )
        self.data['alpha_light'] = pd.read_csv(
            self.data_dir + "alpha_light_yield.csv",
            sep=",",
            header=0,
        )
        if not use_converted:
            for dataset in self.data.keys():
                converted_indices = [j for j, x in enumerate(self.data[dataset]['converted']) if x == 1]
                self.data[dataset].drop(converted_indices, axis=0, inplace=True)
                self.data[dataset] = self.data[dataset].reset_index()
                del self.data[dataset]['index']
                #display(self.data[dataset].to_string())
                #self.data[dataset] = self.data[dataset][(self.data[dataset]['converted'] == 0)]
        if not use_excluded:
            for dataset in self.data.keys():
                excluded_indices = [j for j, x in enumerate(self.data[dataset]['excluded']) if x == 1]
                self.data[dataset].drop(excluded_indices, axis=0, inplace=True)
                self.data[dataset] = self.data[dataset].reset_index()
                del self.data[dataset]['index']
                #display(self.data[dataset].to_string())
                #self.data[dataset] = self.data[dataset][(self.data[dataset]['excluded'] == 0)]
        # for dataset in self.data.keys():
        #     self.data[dataset]['yield_sl'][(self.data[dataset]['yield_sl'] == 0.0)] = 0.05 * self.data[dataset]['yield'][(self.data[dataset]['yield_sl'] == 0.0)]
        #     self.data[dataset]['yield_sh'][(self.data[dataset]['yield_sh'] == 0.0)] = 0.05 * self.data[dataset]['yield'][(self.data[dataset]['yield_sh'] == 0.0)]

    def data_return(self):
        #print(type(self.data))
        return self.data

    def print_data(self):
        print(self.data)

    def plot_2d_data(self, dataset_label: str, x_index, y_index):
            #print(self.data['alpha_light'].field.to_string())
            x=self.data[dataset_label].columns[x_index]
            y=self.data[dataset_label].columns[y_index]            
            self.data[dataset_label].plot(x, y, kind='scatter')
            plt.show()

    def plot_3d_data(self, dataset_label: str, x_index, y_index, z_index):
        x_arr = pd.to_numeric(self.data[dataset_label].iloc[0:,x_index]).to_numpy() 
        y_arr = pd.to_numeric(self.data[dataset_label].iloc[0:,y_index]).to_numpy() 
        z_arr = pd.to_numeric(self.data[dataset_label].iloc[0:,z_index]).to_numpy() 

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.scatter(x_arr, y_arr, z_arr)
        fig.tight_layout()
        plt.show()