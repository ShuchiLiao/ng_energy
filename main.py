from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import Dict, List, Union
from Predictor import ElecUseShortTermPredictor, ElecUseLongTermPredictor, AirCondTempPredictor
from Optimizer import GenericOptimizer
from utils import prepare_long_term_data
from constants import (LSTM_UNIT, EUST_FC_UNIT,
                       FEATURE, EULT_FC_UNIT, EULT_MODEL_NAME,
                       PRIORITY_LIST, GRU_UNIT, AC_FC_UNIT,
                       EXTRA_VAR_NUM, ROOM_VAR_NUM
                       )
import gc
import json
import os
import logging

logging.basicConfig(filename='long_term_train_log', level=logging.INFO, filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "welcome!"}


# return {"message": "hello!"}


class ElecUseShortTermInput(BaseModel):
    data: dict[str, List[Union[int, float]]]


class PredictACTempInput(BaseModel):
    data: dict[str, Dict[str, List[Union[int, float]]]]


@app.post('/train/elec_use_short_term')
async def train_elec_use_short_term(
        file: UploadFile = File(...),
        timesteps: int = Query(24, alias="TrainStep"),  # 历史时间步长
        features: int = Query(1, alias="FeatureNum"),  # 输入特征数量
        predictsteps: int = Query(1, alias="PredictStep"),  # 预测时间步长
        epochs: int = Query(50, alias="Epochs"),  # 训练轮数
        batchsize: int = Query(64, alias="BatchSize"),  # 批量大小
        learningrate: float = Query(0.001, alias='LearningRate')  # 学习率
):
    """
    根据 input_type 和其他参数训练模型。
    :param file: training data
    :param timesteps: 历史时间步长。
    :param features: 输入特征数量。
    :param predictsteps: 预测时间步长。
    :param epochs: 训练轮数。
    :param batchsize: 批量大小。
    :param learningrate: 优化器学习率
    """
    try:
        # output_shape[1] = 1, 只有用电量需要预测
        predictor = ElecUseShortTermPredictor(input_shape=(timesteps, features),
                                              output_shape=(predictsteps, 1))

        # 定义神经网络结构
        predictor.model_layers = [('lstm', int(LSTM_UNIT[0][0]), float(LSTM_UNIT[0][1])),
                                  ('lstm', int(LSTM_UNIT[1][0]), float(LSTM_UNIT[1][1])),
                                  ('linear', int(EUST_FC_UNIT[0][0]), float(EUST_FC_UNIT[0][1]))
                                  ]
        if file:
            json_content = await file.read()
            json_data = json.loads(json_content.decode('utf-8'))
            input_data = json_data['data']
        else:
            raise HTTPException(status_code=400, detail="File not provided")

        # 预处理input_data中的数据，
        # 将非用电量数据提取出来（如温度和湿度=additional_data : 2d list(fea_num, time_steps))
        # 将非用电量数据从input_data中去除，防止影响后面for loop模型训练(updated_input data是一个dict: {str: List}
        updated_input_data, additional_data = predictor.preprocess_data(input_data)

        # 开始为每个电表训练预测模型
        for key, value in updated_input_data.items():
            predictor.model_name = key
            train_data = [] if features == 1 else additional_data.copy()
            # bug : 没有加.copy()，导致train_data size一直增长！这个小bug整了我好久！
            train_data.insert(0, value)  # 加上用电量，2dlist
            data = np.array(train_data).T  # convert to [time_steps, feature_num]

            X_train, y_train, X_test, y_test = predictor.prepare_xy(data)

            predictor.train_model(X_train, y_train, X_test, y_test,
                                  batch_size=batchsize, epochs=epochs, learning_rate=learningrate)

        predictor.clear_session()  # clear session to free up memory

    except Exception as e:
        # Handle exceptions, log the error, and return an appropriate response
        error_message = f"An error occurred during training: {str(e)}"
        return JSONResponse(content={"error": error_message}, status_code=500)
    finally:
        gc.collect()


@app.post("/predict/elec_use_short_term")
# async def predict(input_data: InputData):
def predict_elec_use_short_term(
        input_data: ElecUseShortTermInput,
        timesteps: int = Query(24, alias="TrainStep"),  # 历史时间步长
        features: int = Query(1, alias="FeatureNum"),  # 输入特征数量
        predictsteps: int = Query(1, alias="PredictStep"),  # 预测时间步长
):
    """
    根据输入数据进行预测。
    :param input_data: InputData 实例，包含数据。
    :param timesteps: 历史时间步长。
    :param features: 输入特征数量。
    :param predictsteps: 预测时间步长。
    """
    try:
        predictions = {}
        errors = {}  # 用于记录加载模型失败的模型

        # output_shape[1] = 1, 只有用电量需要预测
        predictor = ElecUseShortTermPredictor(input_shape=(timesteps, features),
                                              output_shape=(predictsteps, 1))

        # 构建神经网络结构，需要和训练时的结构保持一致，不然无法加载模型
        predictor.model_layers = [('lstm', int(LSTM_UNIT[0][0]), float(LSTM_UNIT[0][1])),
                                  ('lstm', int(LSTM_UNIT[1][0]), float(LSTM_UNIT[1][1])),
                                  ('linear', int(EUST_FC_UNIT[0][0]), float(EUST_FC_UNIT[0][1]))
                                  ]

        # bug, if input_data['data'], gives error:
        #ElecUseShortTermInput' object is not subscriptable
        input_data = input_data.data

        # 预处理input_data中的数据，
        # 将非用电量数据提取出来（如温度和湿度=additional_data : 2d list(fea_num, time_steps))
        # 将非用电量数据从input_data中去除，防止影响后面for loop模型训练(updated_input data是一个dict: {str: List}
        updated_input_data, additional_data = predictor.preprocess_data(input_data)

        # 开始利用每个电表模型进行预测
        for key, value in updated_input_data.items():
            predictor.model_name = key

            prediction_data = [] if features == 1 else additional_data.copy()
            prediction_data.insert(0, value)  # 加上用电量，2dlist
            data = np.array(prediction_data).T  # 转化为（sequence_len, feature_num)

            # 进行预测
            prediction = predictor.predict(data)

            predictions[key] = prediction[:, 0].tolist()

        predictor.clear_session()  # 清除会话以释放内存

        response_content = {"predictions": predictions}
        if errors:
            response_content["errors"] = errors

        return JSONResponse(content=response_content, status_code=200)

    except Exception as e:
        # Handle exceptions, log the error, and return an appropriate response
        error_message = f"An error occurred during prediction: {str(e)}"
        return JSONResponse(content={"error": error_message}, status_code=500)

    finally:
        gc.collect()


@app.post('/train/total_elec_use_long_term')
async def train_total_elec_use_long_term(
        weather: UploadFile = File(...),
        usage: UploadFile = File(...),
        epochs: int = Query(50, alias="Epochs"),  # 训练轮数
        batchsize: int = Query(64, alias="BatchSize"),  # 批量大小
        learningrate: float = Query(0.001, alias='LearningRate')  # 学习率
):
    """
    @param weather: 网上下载的csv天气文件(网址见接口说明文件）
    @param usage: 长期用电量数据
    @param epochs: 迭代次数
    @param batchsize: 批处理量
    @param learningrate: 学习率
    """
    # Check if the first file is a CSV
    if not weather.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="The first file must be a CSV file.")

    # Check if the second file is a JSON
    if usage.content_type != 'application/json':
        raise HTTPException(status_code=400, detail="The second file must be a JSON file.")

    # Read files if the file types are correct
    weather_content = await weather.read()
    usage_content = await usage.read()

    # 预处理数据
    input_data = prepare_long_term_data(weather_content, usage_content, train=True)
    # train_data is a dataframe with columns of
    # [Year  Month  Day  Hour  Week of day  Weekend or holiday
    # Temperature   Humidity  Precipitation  Historical usage|Usage]

    try:
        predictor = ElecUseLongTermPredictor(input_shape=(1, len(FEATURE) - 1),  # input 没有最后‘usage’列
                                             output_shape=(1, 1),  # 只预测用电量
                                             features=FEATURE)

        #定义模型名称（只有一个模型）
        predictor.model_name = EULT_MODEL_NAME

        # 定义神经网络结构(线性网络结构，数据不足以做rnn网络）
        predictor.model_layers = [('linear', int(EULT_FC_UNIT[0][0]), float(EULT_FC_UNIT[0][1])),
                                  ('linear', int(EULT_FC_UNIT[1][0]), float(EULT_FC_UNIT[1][1])),
                                  ('linear', int(EULT_FC_UNIT[2][0]), float(EULT_FC_UNIT[2][1]))
                                  ]

        # 再次处理数据
        # input_data 是dataframe
        # 这里只是选择了features列并调整了下温度值，然后返回np.array
        data = predictor.preprocess_data(input_data)

        X_train, y_train, X_test, y_test = predictor.prepare_xy(data)
        predictor.train_model(X_train, y_train, X_test, y_test,
                              batch_size=batchsize, epochs=epochs, learning_rate=learningrate)

        predictor.clear_session()  # clear session to free up memory

    except Exception as e:
        # Handle exceptions, log the error, and return an appropriate response
        error_message = f"An error occurred during training: {str(e)}"
        return JSONResponse(content={"error": error_message}, status_code=500)
    finally:
        gc.collect()


@app.post('/predict/total_elec_use_long_term')
async def predict_total_elec_use_long_term(
        weather: UploadFile = File(...),
        usage: UploadFile = File(...),
):
    """
    @param weather: 网上下载的csv天气文件(网址见接口说明文件）
    @param usage: 近期用电量数据
    """
    # Check if the first file is a CSV
    if not weather.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="The first file must be a CSV file.")

    # Check if the second file is a JSON
    if usage.content_type != 'application/json':
        raise HTTPException(status_code=400, detail="The second file must be a JSON file.")

    # Read files if the file types are correct
    weather_content = await weather.read()
    usage_content = await usage.read()

    # 预处理数据
    input_data = prepare_long_term_data(weather_content, usage_content, train=False)
    # train_data is a dataframe with columns of
    # [Year  Month  Day  Hour  Week of day  Weekend or holiday
    # Temperature   Humidity  Precipitation  Historical_usage]

    features = FEATURE[:-1]  # 没有usage data

    try:
        predictor = ElecUseLongTermPredictor(input_shape=(1, len(features)),
                                             output_shape=(1, 1),  # 只预测用电量
                                             features=features)

        # 定义模型名称（只有一个模型）
        predictor.model_name = EULT_MODEL_NAME

        # 定义神经网络结构(线性网络结构，数据不足以做rnn网络）
        predictor.model_layers = [('linear', int(EULT_FC_UNIT[0][0]), float(EULT_FC_UNIT[0][1])),
                                  ('linear', int(EULT_FC_UNIT[1][0]), float(EULT_FC_UNIT[1][1])),
                                  ('linear', int(EULT_FC_UNIT[2][0]), float(EULT_FC_UNIT[2][1]))
                                  ]

        # 再次处理数据
        # input_data 是dataframe
        # 这里只是选择了features列并调整了下温度值，然后返回np.array
        data = predictor.preprocess_data(input_data)

        prediction = predictor.predict(data)
        result = {'elec_usage_long_term_predictions': prediction[:, 0].tolist()}

        predictor.clear_session()  # clear session to free up memory

        return JSONResponse(content=result, status_code=200)

    except Exception as e:
        # Handle exceptions, log the error, and return an appropriate response
        error_message = f"An error occurred during training: {str(e)}"
        return JSONResponse(content={"error": error_message}, status_code=500)
    finally:
        gc.collect()


@app.post('/train/ac_temp')
async def train_ac_temp(
        data_file: UploadFile = File(...),  # training data, 包含SFWD，ROOMTEMP这些
        timesteps: int = Query(1, alias="TrainStep"),  # 历史时间步长
        predictsteps: int = Query(1, alias="PredictStep"),  # 预测时间步长
        epochs: int = Query(50, alias="Epochs"),  # 训练轮数
        batchsize: int = Query(64, alias="BatchSize"),  # 批量大小
        learningrate: float = Query(0.001, alias='LearningRate')  # 学习率
):
    """
    :param data_file: 包含SFWD，ROOMTEMP等与AC相关的历史数据
    :param timesteps: 历史时间步长。
    :param predictsteps: 预测时间步长。
    :param epochs: 训练轮数。
    :param batchsize: 批量大小。
    :param learningrate: 优化器学习率
    """
    # Check if the second file is a JSON
    if data_file.content_type != 'application/json':
        raise HTTPException(status_code=400, detail="The files must be JSON files.")

    model_name, _ = os.path.splitext(data_file.filename)
    # Read files if the file types are correct
    json_content = await data_file.read()
    json_data = json.loads(json_content.decode('utf-8'))
    data = json_data["data"]

    # 文件名就是模型名，或者说楼层名，比如K8-1-1
    # 里面是字典形式的数据如：{’SFWD':[], 'ROOMTEMP':[]},可由爬虫工具得到。
    df = pd.DataFrame(data)

    in_feature_num = df.shape[1]
    if (df.shape[1] - EXTRA_VAR_NUM) % ROOM_VAR_NUM == 0:  # 如果不能整除，说明多（少）数据。
        out_feature_num = int(int((df.shape[1] - EXTRA_VAR_NUM) / ROOM_VAR_NUM) + 1)  # 房间数目加1（总功率）
    else:
        raise ValueError("room numbers is not an integer, check your data. ")

    try:
        # output_shape[1] = 1, 只有用电量需要预测
        predictor = AirCondTempPredictor(input_shape=(timesteps, in_feature_num),
                                         output_shape=(predictsteps, out_feature_num),
                                         priority_list=PRIORITY_LIST)

        # 定义模型名称：
        predictor.model_name = model_name
        # 定义神经网络结构
        predictor.model_layers = [('gru', int(GRU_UNIT[0][0]), float(GRU_UNIT[0][1])),
                                  ('gru', int(GRU_UNIT[1][0]), float(GRU_UNIT[1][1])),
                                  ('linear', int(AC_FC_UNIT[0][0]), float(AC_FC_UNIT[0][1])),
                                  ('linear', int(AC_FC_UNIT[1][0]), float(AC_FC_UNIT[1][1]))]

        # 预处理数据:将df column按照priority list排序并返回2d array
        data, data_columns = predictor.preprocess_data(df)

        X_train, y_train, X_test, y_test = predictor.prepare_xy(data)

        predictor.train_model(X_train, y_train, X_test, y_test,
                              batch_size=batchsize, epochs=epochs, learning_rate=learningrate)

        predictor.clear_session()  # clear session to free up memory

    except Exception as e:
        # Handle exceptions, log the error, and return an appropriate response
        error_message = f"An error occurred during training: {str(e)}"
        return JSONResponse(content={"error": error_message}, status_code=500)
    finally:
        gc.collect()


@app.post("/predict/ac_temp")
# async def predict(input_data: InputData):
def predict_ac_temp(
        input_data: PredictACTempInput,
        timesteps: int = Query(1, alias="TrainStep"),  # 历史时间步长
        predictsteps: int = Query(1, alias="PredictStep"),  # 预测时间步长
):
    """
    根据输入数据进行预测。
    :param input_data: InputData 实例，包含温度数据等。格式为{data:{楼层：{’SFWD':[]...}...}}
    :param timesteps: 历史时间步长。
    :param predictsteps: 预测时间步长。
    """
    try:
        predictions = {}
        # 开始d对每层楼的模型进行预测
        for key, value in input_data.data.items():

            df = pd.DataFrame(value)  # 提取每个楼层中个房间ac信息value={'SFWD':[..], ...}

            in_feature_num = df.shape[1]
            if (df.shape[1] - EXTRA_VAR_NUM) % ROOM_VAR_NUM == 0:  # 如果不能整除，说明多（少）数据。
                out_feature_num = int(int((df.shape[1] - EXTRA_VAR_NUM) / ROOM_VAR_NUM) + 1)  # 房间数目加1（总功率）
            else:
                raise ValueError("room numbers is not an integer, check your data. ")

            predictor = AirCondTempPredictor(input_shape=(timesteps, in_feature_num),
                                             output_shape=(predictsteps, out_feature_num),
                                             priority_list=PRIORITY_LIST)

            # 模型名称
            predictor.model_name = key
            # 模型神经网络结构
            predictor.model_layers = [('gru', int(GRU_UNIT[0][0]), float(GRU_UNIT[0][1])),
                                      ('gru', int(GRU_UNIT[1][0]), float(GRU_UNIT[1][1])),
                                      ('linear', int(AC_FC_UNIT[0][0]), float(AC_FC_UNIT[0][1])),
                                      ('linear', int(AC_FC_UNIT[1][0]), float(AC_FC_UNIT[1][1]))]

            # 预处理数据:将df column按照priority list排序并返回2d array
            data, data_columns = predictor.preprocess_data(df)
            data = data.reshape(timesteps, in_feature_num)  # 保证 shape=[steps, in_feature_num)
            # 进行预测
            prediction = predictor.predict(data)  # 输入格式[predictsteps, out_feature_num]

            predictions[key] = {data_columns[i]: prediction[:, i].tolist() for i in range(out_feature_num)}

            predictor.clear_session()  # 清除会话以释放内存
        return JSONResponse(content=predictions, status_code=200)

    except Exception as e:
        # Handle exceptions, log the error, and return an appropriate response
        error_message = f"An error occurred during prediction: {str(e)}"
        return JSONResponse(content={"error": error_message}, status_code=500)

    finally:
        gc.collect()


@app.post('/optimize/ac_temp')
def optimize_ac_temp(input_data: PredictACTempInput,
                     timesteps: int = Query(1, alias="TrainStep"),  # 历史时间步长
                     predictsteps: int = Query(1, alias="PredictStep"),  # 预测时间步长
                     population: int = Query(default=20, alias="Population"),  # 种群数量
                     cxpb: float = Query(default=0.7, alias="CrossOverProbability"),  # 交叉概率
                     mupb: float = Query(default=0.2, alias="MutationProbability"),  # 变异概率
                     generation: int = Query(default=50, alias="Generations")  # 遗传代数
                     ):
    """

    目前只考虑下一时刻的功率和温度做优化。
    （extensibility：预测未来n小时功率和温度，最优化未来n小时的平均功率和不同温度能耗。）
    @param input_data: InputData 实例，包含温度数据等。格式为{data:{楼层：{’SFWD':[]...}...}}
    @param timesteps: 历史时间步长。=1
    @param predictsteps: 预测时间步长。=1
    @param population: 种群数量
    @param cxpb: 交叉概率，（0-1）之间
    @param mupb: 变异概率，（0-1）之间
    @param generation: 遗传代数
    """

    try:
        optimizations = {}
        for key, value in input_data.data.items():

            df = pd.DataFrame(value)  # 提取每个楼层中个房间ac信息value={'SFWD':[..], ...}

            in_feature_num = df.shape[1]
            if (df.shape[1] - EXTRA_VAR_NUM) % ROOM_VAR_NUM == 0:  # 如果不能整除，说明多（少）数据。
                out_feature_num = int(int((df.shape[1] - EXTRA_VAR_NUM) / ROOM_VAR_NUM) + 1)  # 房间数目加1（总功率）
            else:
                raise ValueError("room numbers is not an integer, check your data. ")

            predictor = AirCondTempPredictor(input_shape=(timesteps, in_feature_num),
                                             output_shape=(predictsteps, out_feature_num),
                                             priority_list=PRIORITY_LIST)

            # 模型名称
            predictor.model_name = key
            # 模型神经网络结构
            predictor.model_layers = [('gru', int(GRU_UNIT[0][0]), float(GRU_UNIT[0][1])),
                                      ('gru', int(GRU_UNIT[1][0]), float(GRU_UNIT[1][1])),
                                      ('linear', int(AC_FC_UNIT[0][0]), float(AC_FC_UNIT[0][1])),
                                      ('linear', int(AC_FC_UNIT[1][0]), float(AC_FC_UNIT[1][1]))]
            # 注意，预测方法用ga_predict()，所以需要先加载模型
            if not predictor.load_model():
                error_message = f"Loading model failed. "
                return JSONResponse(content={"error": error_message}, status_code=500)

            # 预处理数据:将df column按照priority list排序并返回2d array
            data, data_columns = predictor.preprocess_data(df)  # data
            #print(np.shape(data)): return (1, in_feature_num)
            # data = data.reshape(timesteps, in_feature_num)  # 保证 shape=[steps, in_feature_num),不需要，因为已经是了

            # 转化为list，因为遗传算法中数据注册形式时python 2dlist
            re = data.tolist()

            # 根据data，改变data最后['TEMPSET',"SFWDSDZ"]的值，来预测，看那个预测结果的objective function loss最小。
            room_num = int((df.shape[1] - EXTRA_VAR_NUM) / ROOM_VAR_NUM)
            setup_length = room_num + 1  # room number + 1 (SFWDSDZ)
            residual = re[0][:-int(setup_length)]  # 数据中不变的那部分

            optimizer = GenericOptimizer(residual, room_numbers=room_num,
                                         model=predictor, population_size=population,
                                         cxpb=cxpb, mupb=mupb, generations=generation)
            optimizations[key] = optimizer.optimize_and_advise(data_columns)

            predictor.clear_session()

        print(optimizations)

        return JSONResponse(content=optimizations, status_code=200)

    except Exception as e:
        # Handle exceptions, log the error, and return an appropriate response
        error_message = f"An error occurred: {str(e)}"
        return JSONResponse(content={"error": error_message}, status_code=500)

    finally:
        gc.collect()
