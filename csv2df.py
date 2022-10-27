import re

def param_extractor(filepath):
    split_path = filepath.split('/')
    filename = split_path[-1]
    parameter_info = filename.split(' ')
    parameter_list = parameter_info[0].split('_')
    month = parameter_info[2]
    date = f'{parameter_info[1]}-{month[:3]}-{parameter_info[3]}'
    time = parameter_info[4].replace('_', ':')
    parameter_dict = {}
    
    # print(parameter_list)
    substrate = re.search('[a-z]+', parameter_list[0], flags=re.IGNORECASE).group()
    parameter_list = parameter_list[1:]
    for parameter_str in parameter_list:
        value = re.search(r'\d+', parameter_str)
        parameter = re.search('[a-z]+', parameter_str, flags=re.IGNORECASE)
        parameter_dict[parameter.group()] = int(value.group())
    
    return [parameter_dict, date, time, substrate]

def E_pump2n(E_pump):
    h, c = 6.626e-34, 3e10
    λ = 5.32e-5
    d, spotsize = 2.3e-5, 7.8e-3
    abs = 7.3e-1
    
    return E_pump * λ / h / c * abs / spotsize / d