#!/usr/bin/env python3
import os
import platform

import pandas as pd
import scipy.integrate as integrate
from matplotlib import pyplot as plt

import csv2df

def main():
    pd.options.display.precision = 20
    df = pd.DataFrame()
    match platform.system():
        case 'Darwin':
            path = '/Users/kentahirahara/code/python3/Research-Project-LTI/lab_data'
        case 'Linux':
            path = '/home/kenn/code-python/KSOP/Research-Project-LTI/lab_data'
    files = os.listdir(path)
    files_dir = [f for f in files if os.path.isdir(os.path.join(path, f))]
    substrates = set()
    for directory in files_dir:
        os.chdir(f'{path}/{directory}')
        path_csv = os.getcwd()
        files = os.listdir(path_csv)
        csv_files = [f for f in files if os.path.isfile(os.path.join(path_csv, f)) and '.csv' in f]
        for csv_file in csv_files:
            csv_filepath = f'{path_csv}/{csv_file}'

            parameter_dict, date, time, substrate = csv2df.param_extractor(csv_filepath)
            parameter_dict['Date'] = date
            parameter_dict['Time'] = time
            parameter_dict['Substrate'] = substrate

            substrates.add(substrate)
            
            df_intensity_spectrum = pd.read_csv(csv_filepath, sep='\s+')
            data = df_intensity_spectrum.to_numpy()
            integrated_spectrum = integrate.simps(data[:,1], data[:,0])
            PL = integrated_spectrum / parameter_dict['A'] / parameter_dict['G']
            V = parameter_dict['E'] / 1000
            E_pump = - 1e-8*V**4 + 4e-8*V**3 + 8e-9*V**2 + 1e-9*V  + 4e-10
            parameter_dict['E_pump'] = E_pump
            n = csv2df.E_pump2n(E_pump)
            parameter_dict['n'] = n
            PLQY = PL / n
            parameter_dict['PLQY'] = PLQY
            s = pd.DataFrame(parameter_dict.values(), index=parameter_dict.keys()).T
            df = pd.concat([df,s])

    for substrate_ in substrates:
        df_substrate = df[df['Substrate'] == substrate_]
        df_substrate = df_substrate.sort_values('T')
        df_substrate = df_substrate.reindex(columns=['Date', 'Time', 'Substrate', 'T', 'A', 'G', 'E', 'E_pump', 'n', 'PLQY'])
        df_substrate.index = [i for i, _ in enumerate(df_substrate.index)]

        temp = df_substrate['T'].to_list()
        temp_set = set(temp)
        sorted_temp_set = sorted(list(temp_set))

        num_of_fig = len(sorted_temp_set)
        col = 3 if num_of_fig > 6 else 2
        row = num_of_fig // col + 1 if num_of_fig % col else num_of_fig // col

        fig = plt.figure(figsize=(16, 10), dpi=80)
        fig.suptitle(f'PLQY on {substrate_}', fontsize=20)

        for i, temp in enumerate(sorted_temp_set):
            df_single_temp = df_substrate[df_substrate['T'] == temp]
            df_single_temp = df_single_temp.sort_values('n')

            n_ndarray = df_single_temp['n'].to_numpy()
            PLQY_ndarray = df_single_temp['PLQY'].to_numpy()

            ax = fig.add_subplot(col, row, i+1)
            # ax.set_xlabel('Carrier density')
            # ax.set_ylabel(r'PLQY')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(f'{temp}K')
            # ax.set_xlim(1e15, 1e18)
            ax.scatter(n_ndarray, PLQY_ndarray)
    plt.show()

if __name__ == '__main__':
    main()