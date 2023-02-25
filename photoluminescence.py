#!/usr/bin/env python3
import re
import os
import platform

import numpy as np
import pandas as pd
import scipy.integrate as integrate
from matplotlib import pyplot as plt

def main():
    pd.options.display.precision = 20
    df = pd.DataFrame()
    match platform.system():
        case 'Darwin':
            path = '/Users/kentahirahara/code/python3/Research-Project-LTI/lab_data_original'
        case 'Linux':
            path = '/home/kenn/code-python/KSOP/Research-Project-LTI/lab_data'
    files = os.listdir(path)
    files_dir = [f for f in files if os.path.isdir(os.path.join(path, f)) and f != 'wrong_data']
    # print(files_dir)
    substrates = set()
    for directory in files_dir:
        os.chdir(f'{path}/{directory}')
        path_csv = os.getcwd()
        files = os.listdir(path_csv)
        csv_files = [f for f in files if os.path.isfile(os.path.join(path_csv, f)) and '.csv' in f]
        for csv_file in csv_files:
            csv_filepath = f'{path_csv}/{csv_file}'
            parameter_dict, date, time, substrate = param_extractor(csv_filepath)
            parameter_dict['Date'] = date
            parameter_dict['Time'] = time
            parameter_dict['Substrate'] = substrate

            substrates.add(substrate)

            df_intensity_spectrum = pd.read_csv(csv_filepath, sep='\s+')
            data = df_intensity_spectrum.to_numpy()
            integrated_spectrum = integrate.simps(data[:,1], data[:,0])
            PL = integrated_spectrum / parameter_dict['A'] / parameter_dict['G']
            V = parameter_dict['E'] / 1000
            E_pump = - 1e-8*V**4 + 4e-8*V**3 + 8e-9*V**2 + 1e-9*V + 4e-10
            parameter_dict['E_pump'] = E_pump
            n = E_pump2n(E_pump)
            parameter_dict['n'] = n
            PLQY = PL / n # PLQY = # of photons emitted / # of photons absorbed
            parameter_dict['PLQY'] = PLQY
            s = pd.DataFrame(parameter_dict.values(), index=parameter_dict.keys()).T
            df = pd.concat([df,s])
    
    df = df[df['n'] > 0] #carrier density has to be positive
    print(df)
    # df.to_csv(f'{path}/df_.csv')
    # print(substrates)

    for substrate_ in substrates:
        df_substrate = df[df['Substrate'] == substrate_]
        df_substrate = df_substrate.sort_values('T')
        df_substrate = df_substrate.reindex(columns=['Date', 'Time', 'Substrate', 'T', 'A', 'G', 'E', 'E_pump', 'n', 'PLQY'])
        df_substrate.index = [i for i, _ in enumerate(df_substrate.index)]

        csv_each_substrate = f'{path}/df_substrate_{substrate_}.csv'
        df_substrate.to_csv(csv_each_substrate)

        # create csv file to select which data is favorable to be used for linear fitting for lower carrier concentration
        df_fitting_select = df_substrate[]
        temp = df_substrate['T'].to_list()
        temp_set = set(temp)
        sorted_temp_set = sorted(list(temp_set))

        num_of_fig = len(sorted_temp_set)
        col = 3 if num_of_fig > 6 else 2
        row = num_of_fig // col + 1 if num_of_fig % col else num_of_fig // col

        fig = plt.figure(figsize=(16, 10), dpi=80)
        fig.suptitle(f'PLQY on {substrate_}', fontsize=20)
        print(sorted_temp_set)
        for i, temp in enumerate(sorted_temp_set):
            df_single_temp = df_substrate[df_substrate['T'] == temp]
            df_single_temp = df_single_temp.sort_values('n')

            n_list = df_single_temp['n'].values.tolist()
            PLQY_list = df_single_temp['PLQY'].values.tolist()
            log10n = np.log10(n_list)
            log10PLQY = np.log10(PLQY_list)
            data_for_fitting = len(log10n) // 2

            ax = fig.add_subplot(col, row, i+1)
            # ax.set_xlabel('Carrier density')
            # ax.set_ylabel(r'PLQY')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_title(f'{temp}K')
            # ax.set_xlim(1e15, 1e18)
            ax.scatter(n_list, PLQY_list)
    plt.show()

def param_extractor(filepath: str):
    split_path = filepath.split('/')
    filename = split_path[-1]
    parameter_info = filename.split(' ')
    parameter_list = parameter_info[0].split('_')
    month = parameter_info[2]
    date = f'{parameter_info[1]}-{month[:3]}-{parameter_info[3]}'
    time = parameter_info[4].replace('_', ':')
    parameter_dict = {}

    substrate = re.search('[a-z]+', parameter_list[0], flags=re.IGNORECASE).group()
    parameter_list = parameter_list[1:]
    for parameter_str in parameter_list:
        value = re.search(r'\d+', parameter_str)
        parameter = re.search('[a-z]+', parameter_str, flags=re.IGNORECASE)
        parameter_dict[parameter.group()] = int(value.group())

    return parameter_dict, date, time, substrate

def E_pump2n(E_pump):
    h, c = 6.626e-34, 3e10
    λ = 5.32e-5
    d, spotsize = 2.3e-5, 7.8e-3
    abs = 7.3e-1

    return E_pump * λ / h / c * abs / spotsize / d

if __name__ == '__main__':
    main()