import os
import json
import logging
import random
from io import StringIO
import pandas as pd
from datetime import datetime, timedelta
import holidays
import joblib


def setup_logger(name, log_file):
    """
    配置日志记录器
    :param name: 日志记录器的名称
    :param log_file: 日志文件名
    :return: 配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 清除旧的处理器（防止重复添加）
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建文件处理器
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)

    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M')
    handler.setFormatter(formatter)

    # 添加处理器到日志记录器
    logger.addHandler(handler)

    return logger


# functions used for short term electricity usage prediction
def process_eu_raw_data(data_file_path):
    """
    国网特定的json数据文件提取主要信息并转化为字典格式
    :param data_file_path：原始数据文件地址
    :return: 字典格式数据
    """
    with open(data_file_path, 'r', encoding='utf-8') as file:
        data = file.read()
    data = data.split('\n', 2)[2]
    parsed_data = json.loads(data)
    data_dict = {}

    for entry in parsed_data['data']['series']:
        name = entry['name']
        y_axis_values = [value['yAxis'] for value in entry['values']]
        data_dict[name] = y_axis_values
    with open(f'{data_file_path}_jsonData.json', 'w') as f:
        json.dump(data_dict, f)
    return data_dict

    # example
    # data = process_eu_raw_data("办公空调机组电表功率.json")
    # print(len(data))
    # for key, value in data.items():
    #     print(key)
    #     print(len(value))
    #     predictorNG = ElectricityUsagePredictor(input_shape=(24, 1), output_shape=1, epochs=50,
    #                                             make_plot=True)
    #     X, y = predictorNG.prepare_data(data_list=value)
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)
    #     predictorNG.train_model(X, y)
    #     predictorNG.evaluate_model(X_test, y_test)


def generate_eu_random_data(num_keys, key_header, m, n, file_name):
    """
    生成包含指定数量键值对的字典，每个键对应一个形状为(m, n)的2D随机数列表。
    这里键为电表名称。列表为用电量，温度，湿度等时序数据

    参数:
    num_keys (int): 字典中的键的数量。
    key_header(str): 键名称前缀。
    m (int): 2D列表的行数。
    n (int): 2D列表的列数。
    file_name(str): 保存文件名称。
    
    返回:
    null
    """
    data = {}
    for i in range(num_keys):
        key = f"{key_header}_{i + 1}"
        value = [[random.random() for _ in range(n)] for _ in range(m)]
        data[key] = value

    with open(file_name, 'w') as json_file:
        json.dump({"data": data}, json_file, indent=4)


def generated_eu_random_data_examples():
    # 示例使用
    num_keys = 6  # 电表数量
    key_header = 'C023_D07584-Pt_test'  # 电表名字
    m = 3  # feature numbers, 比如用电量，温度，湿度
    n = 8888  # 每个feature 有多少时间步的数据点

    # fake train data
    generate_eu_random_data(num_keys, key_header, m, n,
                            'request_body_train_multiple_features_random_data.json')

    generate_eu_random_data(6, 'C023_D07588-Pt_test', 1, 8888,
                            'request_body_train_single_feature_random_data.json')

    # fake predict data
    generate_eu_random_data(num_keys, key_header, 3, 48,
                            'request_body_predict_multiple_features_random_data.json')

    generate_eu_random_data(6, 'C023_D07588-Pt_test', 1, 24,
                            'request_body_predict_single_feature_random_data.json')


# functions used for long term electricity usage prediction
def process_weather_data(weather_content):
    # 读取上传的csv文件内容，用pandas加载，并去除前5行无用信息：
    data0 = pd.read_csv(StringIO(weather_content.decode('utf-8'))).iloc[5:, 0]

    # data.shape = (m, 1), 所有信息都在一列，用；分开。因此先分离信息：
    data0_split = data0.str.split(';', expand=True).reset_index(drop=True)

    # data.shape = (m, ~30), 但entry中的元素都是string的形式，因此去除“”：
    data1 = data0_split.astype(str).apply(lambda x: x.str.strip('"'))
    # 使用第一行作为列名
    data1.columns = data1.iloc[0]
    # 删除第一行（已经作为列名）
    data1 = data1[1:]

    # 提取第一列，‘T'列， ’U'列，‘RRR'列， 分别对应['Time', 'Temperature', 'Humidity', 'Precipitation']：
    data2 = data1[[data1.columns[0], 'T', 'U', 'RRR']]
    # 重命名列名
    data2.columns = ['Time', 'Temperature', 'Humidity', 'Precipitation']

    # 除时间列外（第一列）的所有列转换为数值类型。如果在转换过程中遇到无法转换为数值的值，它会将这些值转换为 NaN
    for column in data2.columns[1:]:
        data2[column] = pd.to_numeric(data2[column], errors='coerce')
        # 将列中的NaN转化为0
        data2[column] = data2[column].fillna(0)

    # 处理第一列时间
    data2['Time'] = pd.to_datetime(data2['Time'], format='%d.%m.%Y %H:%M')
    data2 = data2.set_index('Time').sort_values(by='Time')

    # 每小时作线性插值补充数据
    data2_resampled = data2.resample('h').asfreq()
    data = data2_resampled.interpolate(method='time').reset_index()
    # print(data0.head())
    # print(data0_split.head())
    # print(data1.head())
    # print(data2.head())
    # print(data)

    return data


#load_json_data(usage_data):
def process_long_term_usage_data(usage_content):
    # with open(file_name, 'r') as file:
    #     # Skip non-JSON lines (if you know how many lines to skip, use file.readlines()[N:])
    #     json_data = ''
    #     start_reading = False
    #     for line in file:
    #         if line.strip().startswith('{'):
    #             start_reading = True
    #         if start_reading:
    #             json_data += line
    #     # Now json_data should contain the full JSON as a string
    #     data = json.loads(json_data)
    usage_data = json.loads(usage_content.decode('utf-8'))
    data = usage_data
    time = [entry['x'] for entry in data['data']]
    usage = [entry['y'] for entry in data['data']]
    result = {'Time': time, 'Usage': usage}
    df = pd.DataFrame(result)
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values(by='Time')
    # df.to_csv('usage_data' + '.csv', index=False)
    return df


# load_json_data_file(file_name):
def process_long_term_usage_data_file(file_name):
    with open(file_name, 'r') as file:
        # Skip non-JSON lines (if you know how many lines to skip, use file.readlines()[N:])
        json_data = ''
        start_reading = False
        for line in file:
            if line.strip().startswith('{'):
                start_reading = True
            if start_reading:
                json_data += line
        # Now json_data should contain the full JSON as a string
        data = json.loads(json_data)
    time = [entry['x'] for entry in data['data']]
    usage = [entry['y'] for entry in data['data']]
    result = {'Time': time, 'Usage': usage}
    df = pd.DataFrame(result)
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values(by='Time')
    # df.to_csv('usage_data' + '.csv', index=False)
    return df


# prepare_data_method_1
def prepare_long_term_data(weather_content, usage_data, train=True):
    tempdata = process_weather_data(weather_content)
    temp_data = tempdata.copy()
    electricity_data = process_long_term_usage_data(usage_data)
    print(temp_data.head())
    print(electricity_data.head())

    temp_data.insert(loc=1, column='Year', value=temp_data['Time'].dt.year)
    temp_data.insert(loc=2, column='Month', value=temp_data['Time'].dt.month)
    temp_data.insert(loc=3, column='Day', value=temp_data['Time'].dt.day)
    temp_data.insert(loc=4, column='Hour', value=temp_data['Time'].dt.hour)

    cn_holidays = holidays.China()

    if train:
        temp_data.insert(loc=5, column='Week of day', value=temp_data['Time'].dt.weekday)
        temp_data.insert(loc=6, column='Weekend or holiday',
                         value=temp_data['Time'].dt.date.apply(lambda x: 1 if x in cn_holidays else 0))
        temp_data['Weekend'] = temp_data['Time'].dt.weekday.isin([5, 6])
        temp_data['Weekend or holiday'] = temp_data['Weekend or holiday'] | temp_data['Weekend']
        temp_data['Weekend or holiday'] = temp_data['Weekend or holiday'].astype(int)
        merged = pd.merge(temp_data, electricity_data, on=['Time'], how='inner')
        len_use = int(len(merged) * 0.3)
        temp_data_train = merged.iloc[len_use:]  # “未来” 的情况
        usage_data_train = merged.iloc[:len_use]  # 历史用电量
        usage_data_train.insert(loc=len(usage_data_train.columns) - 1, column='Historical usage',
                                value=usage_data_train.groupby(['Week of day', 'Hour'])['Usage'].transform('mean'))
        # 过去一段时间每天每一时刻的平均历史用电量，附在“未来”情况中作一定修正。
        data = temp_data_train.merge(usage_data_train[['Week of day', 'Hour', 'Historical usage']].drop_duplicates(),
                                     on=['Week of day', 'Hour'], how='left')
        columns = data.columns.tolist()
        columns[-2:] = reversed(columns[-2:])
        data = data[columns]
        # 最后每列就是一个feature：['Year', 'Month', 'Day', 'Hour', 'Week of day', 'Weekend or holiday',
        #                 'Temperature', 'Humidity', 'Precipitation', 'Historical usage', 'Usage']

    else:
        # modify the date to next year
        start_day = temp_data['Time'][0]
        future_start_day = datetime(int(start_day.year + 1), start_day.month, start_day.day, start_day.hour)
        temp_data['Future time'] = [future_start_day + timedelta(hours=i) for i in range(len(temp_data))]

        temp_data.insert(loc=5, column='Week of day', value=temp_data['Future time'].dt.weekday)
        temp_data.insert(loc=6, column='Weekend or holiday',
                         value=temp_data['Future time'].dt.date.apply(lambda x: 1 if x in cn_holidays else 0))
        temp_data['Weekend'] = temp_data['Future time'].dt.weekday.isin([5, 6])
        temp_data['Weekend or holiday'] = temp_data['Weekend or holiday'] | temp_data['Weekend']
        temp_data['Weekend or holiday'] = temp_data['Weekend or holiday'].astype(int)
        electricity_data.insert(loc=1, column='Hour', value=electricity_data['Time'].dt.hour)
        electricity_data.insert(loc=2, column='Week of day', value=electricity_data['Time'].dt.weekday)

        electricity_data.insert(loc=len(electricity_data.columns) - 1, column='Historical usage',
                                value=electricity_data.groupby(['Week of day', 'Hour'])['Usage'].transform('mean'))
        electricity_data.drop(columns=['Time']).reset_index()

        data = temp_data.merge(electricity_data[['Week of day', 'Hour', 'Historical usage']].drop_duplicates(),
                               on=['Week of day', 'Hour'], how='left')
        data.drop(columns=['Future time'], inplace=True)

    data = data.sort_values(by='Time')
    data.drop(columns=['Weekend', 'Time'], inplace=True)
    # data.to_csv('res.csv')
    print(data.columns)
    return data


# functions used for ac tmp setup

#prepare_json_data(data_file_folder, data_name):
def process_ac_raw_data(data_file_folder, data_name):
    data_dict = {}
    for filename in os.listdir(data_file_folder):
        file_path = os.path.join(data_file_folder, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read()
        data = data.split('\n', 2)[2]
        parsed_data = json.loads(data)

        for entry in parsed_data['data']['series']:
            name = entry['name']
            y_axis_values = [value['yAxis'] for value in entry['values']]
            data_dict[name] = y_axis_values

        def key_priority(key):
            if key.endswith("Pt"):
                return (1, key)
            elif key.endswith("SFWD"):
                return (2, key)
            elif key.endswith("HFWD"):
                return (3, key)
            elif key.endswith("SPEED"):
                return (4, key)
            elif key.endswith("TEMPROOM"):
                return (5, key)
            elif key.endswith("TEMPSET"):
                return (6, key)
            else:
                return (7, key)

    # save the data
    sorted_dict = dict(sorted(data_dict.items(), key=lambda item: key_priority(item[0])))
    with open(f'data/{data_name}.json', 'w') as f:
        json.dump(sorted_dict, f)


def process_ac_json_data(df, priority_list):

    priority_dict = {suffix: i for i, suffix in enumerate(priority_list)}

    def key_priority(key):
        suffix = next((suffix for suffix in priority_list if key.endswith(suffix)), None)
        return (priority_dict.get(suffix, len(priority_list)), key)

    sorted_columns = sorted(df.columns, key=key_priority)
    sorted_data = df[sorted_columns]
    data = sorted_data.dropna()
    data_columns = data.columns

    return data.to_numpy(), data_columns


# functions for scaling
def fit_transform_3d(data, scaler, scaler_save_path):
    # 将数据重塑为2D以应用 scaler
    data_reshaped = data.reshape(-1, data.shape[-1])

    # 拟合并转换2D数据
    data_scaled_reshaped = scaler.fit_transform(data_reshaped)

    # 将数据重新形状回3D
    data_scaled = data_scaled_reshaped.reshape(data.shape)

    joblib.dump(scaler, scaler_save_path)

    return data_scaled


def transform_3d(data, scaler):
    # 将数据重塑为2D以应用 scaler
    data_reshaped = data.reshape(-1, data.shape[-1])

    # 转换2D数据
    data_scaled_reshaped = scaler.transform(data_reshaped)

    # 将数据重新形状回3D
    data_scaled = data_scaled_reshaped.reshape(data.shape)

    return data_scaled


def inverse_transform_3d(data, scaler):
    # 将数据重塑为2D以应用 scaler 的 inverse_transform
    data_reshaped = data.reshape(-1, data.shape[-1])

    # 逆变换2D数据
    data_inversed_reshaped = scaler.inverse_transform(data_reshaped)

    # 将数据重新形状回3D
    data_inversed = data_inversed_reshaped.reshape(data.shape)

    return data_inversed


if __name__ == '__main__':
    # generated_eu_random_data_examples()
    # process_ac_raw_data('./dataset2_20240319/K-8-1', 'K-8-1')
    # process_long_term_usage_data_file('./data/2023.12.01-2024.4.11历史每小时用电量.json')
    exit()
