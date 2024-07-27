import json
import random
from typing import List, Union, Dict


class Elec_Use_Short_Term_Input:
    def __init__(self, data: Dict[str, List[Union[int, float]]]):
        self.data = data


class Predict_AC_Temp_Input:
    def __init__(self, data: Dict[str, Dict[str, List[Union[int, float]]]]):
        self.data = data


def generate_random_data_elec_use(key_list: list, num_values: int) -> Elec_Use_Short_Term_Input:
    data = {key_list[i]: [round(random.uniform(0, 100), 2) for _ in range(num_values)] for i in range(len(key_list))}
    return Elec_Use_Short_Term_Input(data=data)


def generate_random_data_predict_ac_temp(key_list: list, sub_keys_list: list, num_values: int) -> Predict_AC_Temp_Input:
    data = {
        key_list[i]: {sub_keys_list[j]: [round(random.uniform(0, 100), 2) for _ in range(num_values)] for j in
                      range(len(sub_keys_list))}
        for i in range(len(key_list))}
    return Predict_AC_Temp_Input(data=data)


def generate_random_data_train_ac_temp(keys, extend_keys, n, m):
    """
    生成一个字典，其中的key来自一个列表，同时对指定的key进行编号扩展
    所有key的value都是一个长度为m的list

    :param keys: list，初始关键字列表
    :param extend_keys: list，需要编号扩展的关键字列表
    :param n: int，编号扩展的数量
    :param m: int，列表长度
    :return: dict，生成的字典
    """
    result_dict = {}

    for key in keys:
        # 添加原始key
        result_dict[key] = [round(random.uniform(0, 100), 2) for _ in range(m)]

        # 如果key在扩展列表中，对其进行编号扩展
        if key in extend_keys:
            for i in range(1, n + 1):
                extended_key = f"{i}-{key}"
                result_dict[extended_key] = [round(random.uniform(0, 100), 2) for _ in range(m)]

    return result_dict


def save_to_json(obj, filename: str):
    with open(f'{filename}.json', 'w') as f:
        json.dump(obj.__dict__, f)


def save_to_json_2(dict, filename: str):
    with open(f'{filename}.json', 'w') as f:
        json.dump(dict, f)


# usage
# /train/elec_use_short_term
post = 'train_elec_use_short_term'
list1 = ['test1-Pt', 'test2-Pt', 'test3-Pt']
list2 = ['test4-Pt', 'test5-Pt', 'test6-Pt', 'TEMPOUT', 'HUMIDITY']
data1 = generate_random_data_elec_use(list1, num_values=2000)
data2 = generate_random_data_elec_use(list2, num_values=2000)

save_to_json(data1, 'example_input_data/' + f'{post}_example_input_data_1')
save_to_json(data2, 'example_input_data/' + f'{post}_example_input_data_2')

# /predict/elec_use_short_term
post = 'predict_elec_use_short_term'
list1 = ['test1-Pt', 'test2-Pt', 'test3-Pt']
list2 = ['test4-Pt', 'test5-Pt', 'test6-Pt', 'TEMPOUT', 'HUMIDITY']
data1 = generate_random_data_elec_use(list1, num_values=24)
data2 = generate_random_data_elec_use(list2, num_values=24)

save_to_json(data1, 'example_input_data/' + f'{post}_example_input_data_1')
save_to_json(data2, 'example_input_data/' + f'{post}_example_input_data_2')

# /train/ac-temp
post = 'train_ac_temp'

# 示例使用
keys = ["Pt", "TEMPROOM", "TEMPOUT", "HUMIDITY", "SPEED", "TEMPSET", "SFWDSDZ", "SFWD", "HFWD"]
extend_keys = ["TEMPROOM", "SPEED", "TEMPSET"]
n = 8  # 对特定关键字进行编号扩展的数量
m = 1888  # 每个key对应的list长度

data3 = generate_random_data_train_ac_temp(keys, extend_keys, n, m)
save_to_json_2(data3, 'example_input_data/' + f'{post}_example_K-6-6')

keys = ["Pt", "TEMPROOM", "TEMPOUT", "HUMIDITY", "SPEED", "TEMPSET", "SFWDSDZ", "SFWD", "HFWD"]
extend_keys = ["TEMPROOM", "SPEED", "TEMPSET"]
n = 10  # 对特定关键字进行编号扩展的数量
m = 2000  # 每个key对应的list长度

data4 = generate_random_data_train_ac_temp(keys, extend_keys, n, m)
save_to_json_2(data4, 'example_input_data/' + f'{post}_example_k-8-8')


# /predict/ac-temp
# 示例使用
post = 'train_ac_temp'
keys = ["Pt", "TEMPROOM", "TEMPOUT", "HUMIDITY", "SPEED", "TEMPSET", "SFWDSDZ", "SFWD", "HFWD"]
extend_keys = ["TEMPROOM", "SPEED", "TEMPSET"]
n = 8  # 对特定关键字进行编号扩展的数量
m = 1  # 每个key对应的list长度

data3 = generate_random_data_train_ac_temp(keys, extend_keys, n, m)

# 示例使用
keys = ["Pt", "TEMPROOM", "TEMPOUT", "HUMIDITY", "SPEED", "TEMPSET", "SFWDSDZ", "SFWD", "HFWD"]
extend_keys = ["TEMPROOM", "SPEED", "TEMPSET"]
n = 10  # 对特定关键字进行编号扩展的数量
m = 1  # 每个key对应的list长度

data4 = generate_random_data_train_ac_temp(keys, extend_keys, n, m)

data5 = {'data': {f'{post}_example_k-6-6': data3, f'{post}_example_k-8-8': data4}}

post = 'predict_ac_temp'
save_to_json_2(data5, 'example_input_data/' + f'{post}_example_input_data_1')


# 示例使用
post = 'train_ac_temp'
keys = ["Pt", "TEMPROOM", "TEMPOUT", "HUMIDITY", "SPEED", "TEMPSET", "SFWDSDZ", "SFWD", "HFWD"]
extend_keys = ["TEMPROOM", "SPEED", "TEMPSET"]
n = 8  # 对特定关键字进行编号扩展的数量
m = 12  # 每个key对应的list长度

data3 = generate_random_data_train_ac_temp(keys, extend_keys, n, m)

# 示例使用
keys = ["Pt", "TEMPROOM", "TEMPOUT", "HUMIDITY", "SPEED", "TEMPSET", "SFWDSDZ", "SFWD", "HFWD"]
extend_keys = ["TEMPROOM", "SPEED", "TEMPSET"]
n = 10  # 对特定关键字进行编号扩展的数量
m = 12  # 每个key对应的list长度

data4 = generate_random_data_train_ac_temp(keys, extend_keys, n, m)

data5 = {'data': {f'{post}_example_k-6-6': data3, f'{post}_example_k-8-8': data4}}

post = 'predict_ac_temp'
save_to_json_2(data5, 'example_input_data/' + f'{post}_example_input_data_2')
