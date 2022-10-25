#!/usr/bin/env python3
import re
import os
import platform

import pandas as pd
import scipy.integrate as integrate
from matplotlib import pyplot as plt

def param_extractor(filepath):
    split_path = filepath.split('/')
    filename = split_path[-1]
    parameter_info = filename.split(' ')
    parameter_list = parameter_info[0].split('_')
    month = parameter_info[2]
    date = f'{parameter_info[1]}-{month[:3]}-{parameter_info[3]}'
    time = parameter_info[4].replace('_', ':')
    
    parameter_dict = {}
    
    for parameter_str in parameter_list:
        value = re.search(r'\d+', parameter_str)
        parameter = re.search('[a-z]+',parameter_str, flags=re.IGNORECASE)
        parameter_dict[parameter.group()] = int(value.group())
    
    return [parameter_dict, date, time]

def E_pump2n(E_pump):
    h, c = 6.626e-34, 3e10
    λ = 5.32e-5
    d, spotsize = 2.3e-5, 7.8e-3
    abs = 7.3e-1
    
    return E_pump * λ / h / c * abs / spotsize / d

def main():
    pd.options.display.precision = 20
    df = pd.DataFrame()
    match platform.system():
        case 'Darwin':
            path = '/Users/kentahirahara/code/python3/Kenta'
        case 'Linux':
            path = '/home/kenn/code-python/KSOP/Research-Project-LTI'
    files = os.listdir(path)
    files_dir = [f for f in files if os.path.isdir(os.path.join(path, f))]
    for directory in files_dir:
        os.chdir(f'{path}/{directory}')
        path_csv = os.getcwd()
        files = os.listdir(path_csv)
        csv_files = [f for f in files if os.path.isfile(os.path.join(path_csv, f)) and '.csv' in f]
        for csv_file in csv_files:
            csv_filepath = f'{path_csv}/{csv_file}'
            parameter_dict = param_extractor(csv_filepath)[0]
            parameter_dict['Date'] = param_extractor(csv_filepath)[1]
            parameter_dict['Time'] = param_extractor(csv_filepath)[2]
            df_intensity_spectrum = pd.read_csv(csv_filepath, sep='\s+')
            data = df_intensity_spectrum.to_numpy()
            integrated_spectrum = integrate.simps(data[:,1], data[:,0])
            PL = integrated_spectrum / parameter_dict['A'] / parameter_dict['G']
            V = parameter_dict['E'] / 1000
            E_pump = - 0.00000001*V**4 + 0.00000004*V**3 + 0.000000008*V**2 + 0.000000001*V  + 0.0000000004
            # print(E_pump)
            parameter_dict['E_pump'] = E_pump
            n = E_pump2n(E_pump)
            parameter_dict['n'] = n
            PLQY = PL / n
            parameter_dict['PLQY'] = PLQY
            # print(parameter_dict)
            s = pd.DataFrame(parameter_dict.values(), index=parameter_dict.keys()).T
            # print(s)
            df = pd.concat([df,s])
            # print(param_extractor(csv_filepath))
    df = df.drop(columns='Glass')
    df = df.sort_values('T')
    df = df.reindex(columns=['Date', 'Time', 'T', 'A', 'G', 'E', 'E_pump', 'n', 'PLQY'])
    df.index = [i for i, _ in enumerate(df.index)]
    # print(df)

    temp = df['T'].to_list()
    temp_set = set(temp)
    sorted_temp_set = sorted(list(temp_set))
    # print(sorted_temp_set)
    num_of_fig = len(sorted_temp_set)
    col = 3 if num_of_fig > 6 else 2
    # row = sum(divmod(len(sorted_temp_set), col))
    row = num_of_fig // col + 1 if num_of_fig % col else num_of_fig // col
    
    fig = plt.figure(figsize=(16, 10), dpi=80)
    fig.suptitle('PLQY', fontsize=20)

    for i, temp in enumerate(sorted_temp_set):
        df_single_temp = df[df['T'] == temp]
        df_single_temp = df_single_temp.sort_values('n')
        # print(df_single_temp)
        n_ndarray = df_single_temp['n'].to_numpy()
        PLQY_ndarray = df_single_temp['PLQY'].to_numpy()
    
        ax = fig.add_subplot(col, row, i+1)
        # ax.set_xlabel('Carrier density')
        # ax.set_ylabel(r'PLQY')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f'{temp}K')
        ax.set_xlim(4e15, 6e17)
        ax.scatter(n_ndarray, PLQY_ndarray)
    plt.show()

if __name__ == '__main__':
    main()