#!/usr/bin/env python3
import re
import os
import platform
import math

import numpy as np
import pandas as pd
import scipy.integrate as integrate
from matplotlib import pyplot as plt

def main():
    # input original lab data
    main_dir = '/Users/kentahirahara/code/python3/Research-Project-LTI'
    lab_data_dir = f'{main_dir}/lab_data_original'
    # Pandas settings
    pd.options.display.precision = 20
    pd.set_option('display.max_rows', None)
    df = pd.DataFrame()
    # directory initialization
    match platform.system():
        case 'Darwin':
            path = lab_data_dir
        case 'Linux':
            path = '/home/kenn/code-python/KSOP/Research-Project-LTI/lab_data'
    main_files_dirs = os.listdir(main_dir)
    lab_data_files_dirs = os.listdir(lab_data_dir)
    main_dir_csvfiles = [f for f in main_files_dirs if os.path.isfile(os.path.join(main_dir, f)) and '.csv' in f]
    # print(main_dir_csvfiles)
    lab_data_files = [f for f in lab_data_files_dirs if os.path.isdir(os.path.join(path, f)) and f != '.ipynb_checkpoints']
    # print(lab_data_files)
    substrates = set()
    for directory in lab_data_files:
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
            PLQY = PL / n  # PLQY = # of photons emitted / # of photons absorbed
            parameter_dict['PLQY'] = PLQY
            s = pd.DataFrame(parameter_dict.values(), index=parameter_dict.keys()).T
            df = pd.concat([df,s])
    df = df[df['n'] > 0]  # carrier density has to be positive

    # Plot data for each substrate
    a = 1
    match a:
        # Create manual selector
        case 0:
            for substrate_ in substrates:
                df_substrate = df[df['Substrate'] == substrate_]
                df_substrate = df_substrate.sort_values('T')
                # print(df_substrate)
                df_substrate = df_substrate.reindex(columns=['Date', 'Time', 'Substrate', 'T', 'A', 'G', 'E', 'E_pump', 'n', 'PLQY'])
                df_substrate.index = [i for i, _ in enumerate(df_substrate.index)]

                # create csv file to select which data is favorable to be used for linear fitting for lower carrier concentration
                temp = df_substrate['T'].to_list()
                temp_set = set(temp)
                sorted_temp_set = sorted(list(temp_set))

                for i, temp in enumerate(sorted_temp_set):
                    df_single_temp = df_substrate[df_substrate['T'] == temp]
                    df_single_temp = df_single_temp.sort_values('n')
                    df_single_temp = df_single_temp.reset_index(drop=True)
                    if i == 0:
                        df_substrate_catenated = df_single_temp
                    else:
                        df_substrate_catenated = pd.concat([df_substrate_catenated, df_single_temp], axis=0, join='inner') 
                    df_substrate_catenated.reset_index(drop=True)
                    df_substrate_catenated = df_substrate_catenated.reset_index(drop=True)
                    csv_each_substrate = f'{path}/df_substrate_{substrate_}.csv'  # under ~/code/python3/Research-Project-LTI/lab_data_original
                    df_substrate_catenated.to_csv(csv_each_substrate, index=False)
                
        # Plot according to the selection of data for fitting
        case 1:
            # input manual__.csv files into dataframes
            manual_files = [f for f in  main_dir_csvfiles if 'manual_df_' in f]
            for manual_file in manual_files:
                df_manual_file = pd.read_csv(f'{main_dir}/{manual_file}', sep=',')
                temp_list = sorted(list(set(df_manual_file['T'].values.tolist())))
                fig = plt.figure(figsize=(16, 10), dpi=80)
                fig.suptitle(f'PLQY on {df_manual_file.iat[1,0]}', fontsize=20)
                for i, temp in enumerate(temp_list):
                    df_single_temp = df_manual_file[df_manual_file['T'] == temp]
                    # print(df_single_temp)
                    n_list = df_single_temp['n'].values.tolist()
                    PLQY_list = df_single_temp['PLQY'].values.tolist()
                    print(n_list)
                    # n_list = [float(i) for i in n_list]
                    # PLQY_list = [float(i) for i in PLQY_list]
                    # print('len(n_list) == len(PLQY_list): ', len(n_list) == len(PLQY_list))
                    n_list_for_fit = [n_list[i] for i, _ in enumerate(n_list) if df_single_temp.iat[i,4]]
                    PLQY_list_for_fit = [PLQY_list[i] for i, _ in enumerate(PLQY_list) if df_single_temp.iat[i,4]]
                    n_list_for_no_fit = [n_list[i] for i, _ in enumerate(n_list) if not df_single_temp.iat[i,4]]
                    PLQY_list_for_no_fit = [PLQY_list[i] for i, _ in enumerate(PLQY_list) if not df_single_temp.iat[i,4]]
                    ax = fig.add_subplot(4, 3, i+1)
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    ax.set_title(f'{temp}K')
                    # ax.set_xlim(1e15, 1e18)
                    # print(n_list_for_no_fit)
                    ax.scatter(n_list_for_fit, PLQY_list_for_fit, c='red')
                    ax.scatter(n_list_for_no_fit, PLQY_list_for_no_fit, c='blue')
                    horizontal_all, PLQY_fit = linear_fit_ln(n_list, n_list_for_fit, PLQY_list_for_fit)
                    ax.plot(horizontal_all, PLQY_fit)
        # read out dataframe into temperature dependent data
        # cateorize data into 0: not used 1: used for fitting
        # plot
    plt.show()

def param_extractor(filepath: str):
    '''
    Extract the parameter info out of the filename
    '''
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
    '''
    Convert the pump energy into the carrier density
    '''
    h, c = 6.626e-34, 3e10
    λ = 5.32e-5
    d, spotsize = 2.3e-5, 7.8e-3
    abs = 7.3e-1

    return E_pump * λ / h / c * abs / spotsize / d

def linear_fit_ln(n_all, x, y):
    '''
    Calculate linear fitting and return it so that it displays linear in semilog plot.
    '''
    if len(x) > 1:
        horizontal_all = np.linspace(n_all[0], n_all[-1], 100)
        log10_x = [math.log10(i) for i in x]
        log10_y = [math.log10(i) for i in y]
        log10_hor = [math.log10(i) for i in horizontal_all]
        a, b = np.polyfit(log10_x, log10_y, 1)
        fit_log10_y = [a * i + b for i in log10_hor]
        fit_y = [10**i for i in fit_log10_y]
        return horizontal_all, fit_y
    else:
        pass

if __name__ == '__main__':
    main()