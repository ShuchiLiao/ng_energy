from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Query
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
from pydantic import BaseModel
from typing import Dict, List, Union
from Predictor import ElecUseShortTermPredictor, ElecUseLongTermPredictor, AirCondTempPredictor
from Optimizer import GenericOptimizer
from utils import prepare_long_term_data
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


class Elec_Use_Short_Term_Input(BaseModel):
    data: dict[str, List[Union[int, float]]]


class Predict_AC_Temp_Input(BaseModel):
    data: dict[str, Dict[str, List[Union[int, float]]]]


@app.post('/train/elec_use_short_term')
async def train_elec_use_short_term(
        file: UploadFile = File(...),
        timesteps: int = Query(24, alias="TrainStep"),  # 输入时间步长
        features: int = Query(1, alias="FeatureNum"),  # 输入特征数量
        predictsteps: int = Query(1, alias="PredictStep"),  # 预测时间步长
        epochs: int = Query(50, alias="Epochs"),  # 训练轮数
        batchsize: int = Query(64, alias="BatchSize"),  # 批量大小
        learningrate: float = Query(0.001, alias='LearningRate')  # 学习率
):
    """
    根据 input_type 和其他参数训练模型。
    :param file: training data
    :param timesteps: 输入时间步长。
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
        predictor.model_layers = [('lstm', int(50), float(0.5)),
                                  ('lstm', int(100), float(0.4)),
                                  ('linear', int(64), float(0.3))]
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
        input_data: Elec_Use_Short_Term_Input,
        timesteps: int = Query(24, alias="TrainStep"),  # 输入时间步长
        features: int = Query(1, alias="FeatureNum"),  # 输入特征数量
        predictsteps: int = Query(1, alias="PredictStep"),  # 预测时间步长
):
    """
    根据输入数据进行预测。
    :param input_data: InputData 实例，包含数据。
    :param timesteps: 输入时间步长。
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
        predictor.model_layers = [('lstm', int(50), float(0.5)),
                                  ('lstm', int(100), float(0.4)),
                                  ('linear', int(64), float(0.3))]

        # bug, if input_data['data'], gives error:
        #Elec_Use_Short_Term_Input' object is not subscriptable
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
    # Temperature   Humidity  Precipitation  Historical usage Usage]

    features = ['Day', 'Hour', 'Week of day', 'Weekend or holiday',
                'Temperature', 'Humidity', 'Precipitation', 'Historical usage', 'Usage']

    try:
        predictor = ElecUseLongTermPredictor(input_shape=(1, len(features) - 1),  # input 没有最后‘usage’列
                                             output_shape=(1, 1),  # 只预测用电量
                                             features=features)

        #定义模型名称（只有一个模型）
        predictor.model_name = 'total_elec_use_long_term'

        # 定义神经网络结构(线性网络结构，数据不足以做rnn网络）
        predictor.model_layers = [('linear', int(128), float(0.4)),
                                  ('linear', int(128), float(0.4)),
                                  ('linear', int(64), float(0.3))]

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

    features = ['Day', 'Hour', 'Week of day', 'Weekend or holiday',
                'Temperature', 'Humidity', 'Precipitation', 'Historical usage']

    try:
        predictor = ElecUseLongTermPredictor(input_shape=(1, len(features)),
                                             output_shape=(1, 1),  # 只预测用电量
                                             features=features)

        # 定义模型名称（只有一个模型）
        predictor.model_name = 'total_elec_use_long_term'

        # 定义神经网络结构(线性网络结构，数据不足以做rnn网络）
        predictor.model_layers = [('linear', int(128), float(0.4)),
                                  ('linear', int(128), float(0.4)),
                                  ('linear', int(64), float(0.3))]

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
        setup_file: UploadFile = File(...),  # training data, 包含SFWD，ROOMTEMP这些
        temp_file: UploadFile = File(...),  # 温度湿度表
        timesteps: int = Query(1, alias="TrainStep"),  # 输入时间步长
        predictsteps: int = Query(1, alias="PredictStep"),  # 预测时间步长
        epochs: int = Query(50, alias="Epochs"),  # 训练轮数
        batchsize: int = Query(64, alias="BatchSize"),  # 批量大小
        learningrate: float = Query(0.001, alias='LearningRate')  # 学习率
):
    """
    :param setup_file: 包含SFWD，ROOMTEMP等与AC相关的历史数据
    :param temp_file: 温度湿度历史数据
    :param timesteps: 输入时间步长。
    :param predictsteps: 预测时间步长。
    :param epochs: 训练轮数。
    :param batchsize: 批量大小。
    :param learningrate: 优化器学习率
    """
    # Check if the second file is a JSON
    if setup_file.content_type != 'application/json' or temp_file.content_type != 'application/json':
        raise HTTPException(status_code=400, detail="The files must be JSON files.")

    model_name, _ = os.path.splitext(setup_file.filename)
    # Read files if the file types are correct
    setup_data = await setup_file.read()
    temp_data = await  temp_file.read()
    input_setup_data = json.loads(setup_data.decode('utf-8'))
    input_temp_data = json.loads(temp_data.decode('utf-8'))

    df1 = pd.DataFrame(input_temp_data)
    df2 = pd.DataFrame(input_temp_data)
    df = pd.concat([df1, df2], axis=1)

    priority_list = ["Pt", "TEMPROOM", "TEMPOUT", "HUMIDITY", "SFWD", "HFWD", "SPEED",
                     "TEMPSET", "SFWDSDZ"]

    in_feature_num = df.shape[1]
    out_feature_num = int(int((df.shape[1] - 6) / 3) + 1)  # 房间数目加1（总功率）

    try:
        # output_shape[1] = 1, 只有用电量需要预测
        predictor = AirCondTempPredictor(input_shape=(timesteps, in_feature_num),
                                         output_shape=(predictsteps, out_feature_num),
                                         priority_list=priority_list)

        # 定义模型名称：
        predictor.model_name = model_name
        # 定义神经网络结构
        predictor.model_layers = [('gru', int(50), float(0.5)),
                                  ('gru', int(100), float(0.5)),
                                  ('linear', int(128), float(0.4)),
                                  ('linear', int(64), float(0.4))]

        # 预处理数据
        data = predictor.preprocess_data(df)
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
        input_data: Predict_AC_Temp_Input,
        timesteps: int = Query(1, alias="TrainStep"),  # 输入时间步长
        predictsteps: int = Query(1, alias="PredictStep"),  # 预测时间步长
):
    """
    根据输入数据进行预测。
    :param input_data: InputData 实例，包含温度数据等。格式为{data:{楼层：{’SFWD':[]...}...}}
    :param timesteps: 输入时间步长。
    :param predictsteps: 预测时间步长。
    """
    try:
        priority_list = ["Pt", "TEMPROOM", "TEMPOUT", "HUMIDITY", "SFWD", "HFWD", "SPEED",
                         "TEMPSET", "SFWDSDZ"]
        predictions = {}

        # 开始d对每层楼的模型进行预测
        for key, value in input_data.data.items():
            in_feature_num = len(value)
            out_feature_num = int(int((len(value) - 6) / 3) + 1)  # 房间数目加1（总功率）

            predictor = AirCondTempPredictor(input_shape=(timesteps, in_feature_num),
                                             output_shape=(predictsteps, out_feature_num),
                                             priority_list=priority_list)

            # 模型名称
            predictor.model_name = key
            # 模型神经网络结构
            predictor.model_layers = [('gru', int(50), float(0.5)),
                                      ('gru', int(100), float(0.5)),
                                      ('linear', int(128), float(0.4)),
                                      ('linear', int(64), float(0.4))]

            df = pd.DataFrame(value)

            data = predictor.preprocess_data(df)
            data = data.reshape(predictsteps, out_feature_num)

            # 进行预测
            prediction = predictor.predict(data)
            prediction = prediction.tolist()

            result = {'下一小时预测室内温度': prediction[:-1],
                      '下一小时预测总功率': prediction[-1]}
            predictions[key] = result

            predictor.clear_session()  # 清除会话以释放内存

        return JSONResponse(content=predictions, status_code=200)

    except Exception as e:
        # Handle exceptions, log the error, and return an appropriate response
        error_message = f"An error occurred during prediction: {str(e)}"
        return JSONResponse(content={"error": error_message}, status_code=500)

    finally:
        gc.collect()


@app.post('/optimize/ac_temp')
def optimize_ac_temp(input_data: Predict_AC_Temp_Input,
                     timesteps: int = Query(1, alias="TrainStep"),  # 输入时间步长
                     predictsteps: int = Query(1, alias="PredictStep"),  # 预测时间步长
                     ):
    """
    根据输入数据进行预测。
    :param input_data: InputData 实例，包含温度数据等。格式为{data:{楼层：{’SFWD':[]...}...}}
    :param timesteps: 输入时间步长。
    :param predictsteps: 预测时间步长。
    """
    try:
        priority_list = ["TEMPOUT", "HUMIDITY", "SFWDSDZ", "SFWD", "HFWD", "SPEED", "TEMPSET", "TEMPROOM", "Pt"]
        optimizations = {}

        # 开始d对每层楼的模型进行预测
        for key, value in input_data.data.items():
            in_feature_num = len(value)
            out_feature_num = int(int((len(value) - 6) / 3) + 1)  # 房间数目加1（总功率）

            predictor = AirCondTempPredictor(input_shape=(timesteps, in_feature_num),
                                             output_shape=(1, out_feature_num),  #只做下一步预测优化
                                             priority_list=priority_list)

            # 模型名称
            predictor.model_name = key
            # 模型神经网络结构
            predictor.model_layers = [('gru', int(50), float(0.5)),
                                      ('gru', int(100), float(0.5)),
                                      ('linear', int(128), float(0.4)),
                                      ('linear', int(64), float(0.4))]

            df = pd.DataFrame(value)

            data = predictor.preprocess_data(df)
            data = data.reshape(-1)  # 展开数据

            optimizer = GenericOptimizer(data, room_numbers=out_feature_num - 1,
                                         model=predictor,
                                         roomsettemp_range=(19, 23),
                                         windtemp_range=(10, 14), population_size=20, generations=10)
            optimizations[key] = optimizer.optimize_and_advise()

            predictor = None
            optimizer = None
        # print(optimizations)
        return JSONResponse(content=optimizations, status_code=200)

    except Exception as e:
        # Handle exceptions, log the error, and return an appropriate response
        error_message = f"An error occurred: {str(e)}"
        return JSONResponse(content={"error": error_message}, status_code=500)
