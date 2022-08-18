"""
Code for loading and manipulating 
LAr datasets. 
Edited by JHS 8/1/2022 Modified data file location
"""
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import csv

class LArDataset:
    """
    """
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
                self.data[dataset] = self.data[dataset][(self.data[dataset]['converted'] == 0)]
        if not use_excluded:
            for dataset in self.data.keys():
                self.data[dataset] = self.data[dataset][(self.data[dataset]['excluded'] == 0)]
        # for dataset in self.data.keys():
        #     self.data[dataset]['yield_sl'][(self.data[dataset]['yield_sl'] == 0.0)] = 0.05 * self.data[dataset]['yield'][(self.data[dataset]['yield_sl'] == 0.0)]
        #     self.data[dataset]['yield_sh'][(self.data[dataset]['yield_sh'] == 0.0)] = 0.05 * self.data[dataset]['yield'][(self.data[dataset]['yield_sh'] == 0.0)]

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