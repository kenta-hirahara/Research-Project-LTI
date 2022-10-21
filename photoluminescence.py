#!/usr/bin/env python3
import re
import os

import pandas as pd
import scipy.integrate as integrate

def param_extractor(filepath):
    split_path = filepath.split('/')
    filename = split_path[-1]
    parameter_info = filename.split(' ')
    parameter_list = parameter_info[0].split('_')
    
    parameter_dict = {}
    
    for parameter_str in parameter_list:
        value = re.search(r'\d+', parameter_str)
        parameter = re.search('[a-z]+',parameter_str, flags=re.IGNORECASE)
        parameter_dict[parameter.group()] = int(value.group())
    
    return parameter_dict

def main():
    path = '/Users/kentahirahara/code/python3/Kenta'
    files = os.listdir(path)
    files_dir = [f for f in files if os.path.isdir(os.path.join(path, f))]
    os.chdir(f'{path}/{files_dir[0]}')
    path_csv = os.getcwd()
    files = os.listdir(path_csv)
    csv_files = [f for f in files if os.path.isfile(os.path.join(path_csv, f)) and '.csv' in f]
    
    csv_filepath = f'{path_csv}/{csv_files[0]}'
    
    parameter_dict = param_extractor(csv_filepath)

    df = pd.read_csv(csv_filepath, sep='\s+')
    data = df.to_numpy()

    integration = integrate.simps(data[:,1], data[:,0])

    PL = integration / parameter_dict['A'] / parameter_dict['G']
    V = parameter_dict['E'] / 1000
    E_pump = - 0.00000001*V**4 + 0.00000004*V**3 + 0.000000008*V**2 + 0.000000001*V  + 0.0000000004
    parameter_dict['E_pump'] =E_pump
    PLQY = PL / E_pump
    parameter_dict['PLQY'] = PLQY
    print(parameter_dict)

if __name__ == '__main__':
    main()