import gc
import os
import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils import (setup_logger, process_ac_json_data,
                   fit_transform_3d, transform_3d, inverse_transform_3d)


class Predictor:
    """Base class for predictors"""

    def __init__(self, input_shape, output_shape):

        self.input_shape = input_shape
        # (in_sequence_len, in_feature_num) for lstm/gru model,
        # (1, in_feature_num) for linear only model(命名的时候需要）
        self.output_shape = output_shape
        # (out_sequence_len, out_feature_num) for lstm/gru model,
        # (1, out_feature_num) for linear only model(命名的时候需要）

        self._model_name = None
        self._model_layers = []
        # for example: [('lstm', 64),('lstm',128), ('linear', 64)]: 2层lstm，隐藏层features分别为128和64,以此类推
        self.model_path = None
        self.model = None

        self.scaler_X_path = None
        self.scaler_X = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y_path = None
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))

        self.device = "cpu"

        self.train_log = setup_logger(f'{self.__class__.__name__}_train',
                                      'logs/' + f'{self.__class__.__name__}_train.log')

        self.train_detail_log = setup_logger(f'{self.__class__.__name__}_train_detail',
                                             'logs/' + f'{self.__class__.__name__}_train_detail.log')

        self.predict_log = setup_logger(f'{self.__class__.__name__}_predict',
                                        'logs/' + f'{self.__class__.__name__}_predict.log')

    @property
    def model_layers(self):
        return self._model_layers

    @model_layers.setter
    def model_layers(self, model_layers):
        self._model_layers = model_layers
        self.model = self.build_model()
        self.model.to(self.device)

    @property
    def model_name(self):
        return self._model_name

    @model_name.setter
    def model_name(self, model_name):
        self._model_name = (f'{self.__class__.__name__}-{model_name}-'
                            f'in_{self.input_shape[0]}_{self.input_shape[1]}-'
                            f'out_{self.output_shape[0]}_{self.output_shape[1]}')

        self.model_path = 'models/' + self._model_name + '.pth'
        self.scaler_X_path = 'scalers/' + self._model_name + '_X.joblib'
        self.scaler_y_path = 'scalers/' + self._model_name + '_y.joblib'

    def build_model(self):
        class Net(nn.Module):
            def __init__(self, input_size, output_size, model_layers):
                super(Net, self).__init__()
                self.input_size = input_size
                self.output_size = output_size
                self.model_layers = model_layers
                # model_layers format:[(layer_name, hidden_units_size, dropout_probability)]
                self.end_of_rnn_layers = -1
                # where lstm and gru layers end. -1 means default no rnn layers.
                # need it to know where to transform the rnn output to match the
                # output_sequence_length, then forward to linear layers

                self.layers = self.build_layers()

            def build_layers(self):
                layers = []
                current_size = self.input_size[1]
                layer_num = 0

                # for consistency:
                # lstm and gru input/output：tensor shape of [batch_size, seq_len, fea_num]
                # linear input/output：tensor shape of[batch_size, seq_len, fea_num] if lstm layers exist,
                # or [sample_len, fea_num] if only linear layers

                for layer in self.model_layers:

                    layer_type, hidden_size, probability = layer

                    if layer_type.lower() == 'lstm':
                        layers.append(nn.LSTM(current_size, hidden_size, batch_first=True))
                        layers.append(nn.Dropout(probability))
                        layer_num += 2
                        self.end_of_rnn_layers = layer_num - 1

                    elif layer_type.lower() == 'gru':
                        layers.append(nn.GRU(current_size, hidden_size, batch_first=True))
                        layers.append(nn.Dropout(probability))
                        layer_num += 2
                        self.end_of_rnn_layers = layer_num - 1

                    elif layer_type.lower() == 'linear':
                        layers.append(nn.Linear(current_size, hidden_size))
                        layers.append(nn.ReLU())       # bug 1: nn.ReLU 改为 nn.ReLU(); 实例
                        layers.append(nn.Dropout(probability))
                        layer_num += 3

                    else:
                        print('Layer type not acceptable, check your layer type')
                        hidden_size = current_size

                    current_size = hidden_size  # layer output size now become input size

                # 最后计算输出
                layers.append(nn.Linear(current_size, self.output_size[1]))
                print("layer numbers: ", layer_num)
                print("end of rnn layers:, ", self.end_of_rnn_layers)
                print("layers ", layers)
                return nn.Sequential(*layers)

            def forward(self, x):
                layer_num = 0
                for layer in self.layers:

                    if isinstance(layer, nn.LSTM) or isinstance(layer, nn.GRU):
                        x, _ = layer(x)
                    else:
                        x = layer(x)

                    if layer_num == self.end_of_rnn_layers:
                        x = x[:, -self.output_size[0]:, :]  # 取最后时间步的输出到全连接层做预测

                    layer_num += 1
                # 保持shape为（batch size, out_sequence_len, feature_num)
                return x

        return Net(self.input_shape, self.output_shape, self._model_layers)

    def preprocess_data(self, data):
        # return 2d array[sample_num, feature_num] or
        # [all_time_steps, feature_num] of all history data for training/validation
        # remove outliers(optional)
        raise NotImplementedError

    def prepare_xy(self, data):
        self.train_log.info(f'Data processing for {self._model_name}. Training starts soon. ')
        # data:2d array [sample_num, feature_num] (not implemented here)
        # or [all_time_steps, feature_num] of all history data for training/validation(implemented here)

        X = []
        y = []
        for i in range(len(data) - self.input_shape[0] - self.output_shape[0]):
            X_ = data[i:i + self.input_shape[0], -self.input_shape[1]:]
            y_ = data[i + self.input_shape[0]: i + self.input_shape[0]
                                               + self.output_shape[0], :self.output_shape[1]]

            X.append(X_.reshape(self.input_shape[0], self.input_shape[1]))
            y.append(y_.reshape(self.output_shape[0], self.output_shape[1]))
            # 对于electricity-usage，input_shape[1]就是所有列（用电量，温度，湿度等）
            # output_shape[1]就是用电量列（最后一列）

            # 对于ac——temp-setup, input-shape[1]就是所有列
            # 对于ac——temp-setup, output-shape[1]就是室内温度和Pt列（最后数列）

        # X_shape:(sample_num, in_seq_len, in_feature_num),
        # y_shape:(sample_num, out_seq_len, out_feature_num=1)
        # 为了 避免先scale造成数据泄露模型过拟合，需要先划分训练集和验证集,然后再scale
        X_train, X_val, y_train, y_val = train_test_split(np.array(X), np.array(y),
                                                          test_size=0.1, shuffle=True)

        X_train_normalized = fit_transform_3d(data=X_train, scaler=self.scaler_X,
                                              scaler_save_path=self.scaler_X_path)
        y_train_normalized = fit_transform_3d(data=y_train, scaler=self.scaler_y,
                                              scaler_save_path=self.scaler_y_path)

        X_val_normalized = transform_3d(data=X_val, scaler=self.scaler_X)
        y_val_normalized = transform_3d(data=y_val, scaler=self.scaler_y)

        return X_train_normalized, y_train_normalized, X_val_normalized, y_val_normalized

    def train_one_epoch(self, train_loader, criterion, optimizer):
        epoch_loss = 0.
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(train_loader)

    def validate_one_epoch(self, val_loader, criterion):
        val_loss = 0.
        total_error = 0.
        total_y_v = 0.
        total_samples = len(val_loader)

        self.model.eval()
        with torch.no_grad():
            for X_v, y_v in val_loader:
                val_outputs = self.model(X_v)
                val_loss += criterion(val_outputs, y_v).item()

                predictions = val_outputs.cpu().numpy()
                actuals = y_v.cpu().numpy()
                absolute_error = np.abs(predictions - actuals)
                total_error += np.sum(absolute_error)
                total_y_v += np.sum(actuals)

        val_loss /= len(val_loader)
        mean_absolute_error = total_error / total_samples
        mean_y = total_y_v / total_samples

        if mean_y != 0:
            MAE_ = np.abs(mean_absolute_error / mean_y) * 100
            MAE_error_info = f"MAE Error Percentage: {MAE_:.2f} %"
        else:
            MAE_ = np.abs(mean_absolute_error - mean_y)
            MAE_error_info = f"MAE Error: {MAE_:2f}"

        return val_loss, MAE_error_info

    def train_model(self, X_train, y_train, X_val, y_val, batch_size=64, epochs=50, learning_rate=0.001):
        # X, y shapes are:[batch_size, sequence_len/1, feature_len]
        # or [sample_num, feature_len]
        if self.load_model():
            self.train_log.info(f'Update the model: {self.model_path}. ')
            self.train_detail_log.info(f'Update the model: {self.model_path}. ')
        else:
            self.train_log.info(f'Built a new model: {self.model_path}. ')
            self.train_detail_log.info(f'Built a new model: {self.model_path}. ')
            # self.model = self.build_model()
            # self.model.to(self.device)

        # loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        X_train = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)

        # CLASS torch.utils.data.TensorDataset(*tensors)
        # Dataset wrapping tensors.
        # Each sample will be retrieved by indexing tensors along the first dimension.
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)

        # CLASS torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=None, ...)
        # batch_size (int, optional) – how many samples per batch to load (default: 1).
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size // 4, shuffle=False)

        # training process
        self.train_log.info(f"start_training for {self._model_name}. ")
        self.train_detail_log.info(f"start_training for {self._model_name}. ")

        self.model.train()
        best_loss = np.inf
        epochs_no_improve = 0
        early_stop = False
        final_MAE_info = ' '

        for epoch in range(epochs):

            train_loss = self.train_one_epoch(train_loader, criterion, optimizer)
            val_loss, MAE_error_info = self.validate_one_epoch(val_loader, criterion)

            self.train_detail_log.info(f"Epoch {epoch + 1}/{epochs}, "
                                       f"Loss: {train_loss:.4f}, "
                                       f"Validation Loss: {val_loss:.4f}, "
                                       f"{MAE_error_info}")

            self.model.train()
            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
                # 保存最优模型
                final_MAE_info = MAE_error_info
                self.save_model()

            else:
                epochs_no_improve += 1

            if epochs_no_improve >= 10:
                early_stop = True
                self.train_log.info(f"Epoch {epoch + 1}/{epochs}, Final Model "
                                    f"{final_MAE_info}")
                break

        if not early_stop:
            # 保存最终模型
            self.train_log.info(f"Epoch {epochs}/{epochs}, Final Model "
                                f"{final_MAE_info}")
            self.save_model()

        self.train_detail_log.info("Training done. Model saved. \n")
        self.train_log.info("Training done. Model saved. \n")

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            self.scaler_X = joblib.load(self.scaler_X_path)
            self.scaler_y = joblib.load(self.scaler_y_path)
            self.model.to(self.device)
            return True  # 模型成功加载
        else:
            return False  # 模型文件不存在

    def predict(self, X_i):
        # X is numpy array [sequence_len, feature_num]
        Xi = X_i.reshape(1, X_i.shape[0], X_i.shape[1])
        # X shape is [batch_size, sequence_len, feature_num] for rnn models.(implemented here)
        # X shape is [sample_num, feature_num] for linear only models.(implemented in subclass)
        if self.load_model():
            self.predict_log.info(f'{self.model_path} is loaded. Making predictions.')
        else:
            self.predict_log.error(f"Error loading model {self.model_path}")
            return None

        # X is one instance (batch_size = 1)
        # first scale
        X = transform_3d(data=Xi, scaler=self.scaler_X)

        # then convert to tensor
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        # match the shape (batch_size, sequence_length, feature_num)

        self.model.eval()
        with torch.no_grad():
            y_pred_normalized = self.model(X).cpu().numpy()

        # y_pred_normalized也为3d

        y_pred = inverse_transform_3d(data=y_pred_normalized, scaler=self.scaler_y)

        # remove batch size = 1:
        y_i = y_pred.reshape(self.output_shape[0], self.output_shape[1])

        self.predict_log.info(f'Done. \n')
        return y_i

    def clear_session(self):
        self.model = None
        gc.collect()
        # torch.cuda.empty_cache()


class ElecUseShortTermPredictor(Predictor):
    def __init__(self, input_shape, output_shape):
        super().__init__(input_shape, output_shape)

    def preprocess_data(self, input_data):
        additional_data = []
        additional_data_keys = []
        updated_input_data = input_data
        if self.input_shape[1] > 1:  # 不止有用电量数据，还有其他数据
            for key, value in input_data.items():
                if not key.endswith('-Pt'):
                    additional_data.append(value)  # 将非用电量数据提取出来（如温度和湿度）,加入到train
                    additional_data_keys.append(key)

            updated_input_data = {k : v for k, v in input_data.items() if k not in additional_data_keys}
            # 将非用电量数据从input_data中去除，防止影响后面for loop模型训练

        # data_list shape[n, m], ,m：time step number， n：feature number,需要转置为[m, n]
        return updated_input_data, additional_data


class AirCondTempPredictor(Predictor):
    def __init__(self, input_shape, output_shape, priority_list):
        super().__init__(input_shape, output_shape)
        self.priority_list = priority_list

    def preprocess_data(self, df):
        # 为含有所有数据的dataframe，接下来进行排序等处理

        data = process_ac_json_data(df, self.priority_list)
        # data：返回经过固定排序的二维array。（m=total_time_steps, n=feature_num)

        return data


class ElecUseLongTermPredictor(Predictor):
    def __init__(self, input_shape, output_shape, features):
        super().__init__(input_shape, output_shape)
        self.features = features

    def preprocess_data(self, data):
        # data is a dataframe with columns of
        # [Year  Month  Day  Hour  Week of day  Weekend or holiday
        # Temperature   Humidity  Precipitation  Historical usage （option=Usage）],保留需要的feature
        df = data[self.features]
        # data['Month'] = data['Month'].apply(lambda x: 0 if x in [4, 10] else 1)
        df['Temperature'] = abs(df['Temperature'] - 18)

        return df.to_numpy()

    def prepare_xy(self, data):
        self.train_log.info(f'Data processing for {self._model_name}. Training starts soon. ')
        # m, y shape [sample_num , feature num]
        X = data[:, :-1]
        y = data[:, -1]

        X_train, X_val, y_train, y_val = train_test_split(X, y.reshape(-1, 1),  # 保持2d array
                                                          test_size=0.1, shuffle=True)

        X_train_normalized = self.scaler_X.fit_transform(X_train)
        joblib.dump(self.scaler_X, self.scaler_X_path)

        y_train_normalized = self.scaler_y.fit_transform(y_train)
        joblib.dump(self.scaler_y, self.scaler_y_path)

        X_val_normalized = self.scaler_X.transform(X_val)
        y_val_normalized = self.scaler_y.transform(y_val)

        return X_train_normalized, y_train_normalized, X_val_normalized, y_val_normalized

    def predict(self, X):
        # X is a dataframe with columns of
        # [Year  Month  Day  Hour  Week of day  Weekend or holiday
        # Temperature   Humidity  Precipitation  Historical usage ]
        if self.load_model():
            self.predict_log.info(f'{self.model_path} is loaded. Making predictions.')
        else:
            self.predict_log.error(f"Error loading model {self.model_path}")
            return None

        # X 的shape就是[m: sample num, n-1: total feature num-1],

        X = self.scaler_X.transform(X)

        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            y_pred_normalized = self.model(X).cpu().numpy()

        y_pred = y_pred_normalized.reshape(-1, 1)  #保证2d array
        y_i = self.scaler_y.inverse_transform(y_pred)

        self.predict_log.info(f'Done. \n')

        return y_i


