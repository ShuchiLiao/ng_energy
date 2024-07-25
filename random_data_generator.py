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
        key_list[i]: {sub_keys_list[j]: [round(random.uniform(0, 100), 2) for _ in range(num_values)] for j in range(len(sub_keys_list))}
        for i in range(len(key_list))}
    return Predict_AC_Temp_Input(data=data)


def save_to_json(obj, filename: str):
    with open(f'{filename}.json', 'w') as f:
        json.dump(obj.__dict__, f, indent=4)


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
# predict_ac_temp_data = generate_random_data_predict_ac_temp(num_keys=3, num_sub_keys=4, num_values=8)
#
# save_to_json(elec_use_data, 'elec_use_data.json')
# save_to_json(predict_ac_temp_data, 'predict_ac_temp_data.json')
