# elec_use_short_term

LSTM_UNIT = [(50, 0.5), (100, 0.4)]
EUST_FC_UNIT = [(64, 0.4)]

# elec_use_long_term
FEATURE = ['Day', 'Hour', 'Week of day', 'Weekend or holiday',
           'Temperature', 'Humidity', 'Precipitation', 'Historical usage', 'Usage']

EULT_MODEL_NAME = 'electricity_usage_long_term'

EULT_FC_UNIT = [(128, 0.4), (128, 0.4), (64, 0.3)]

# ac-temp-setup
# 数据需要按下面排序
PRIORITY_LIST = ["Pt", "TEMPROOM", "TEMPOUT", "HUMIDITY", "SFWD", "HFWD", "SPEED", "TEMPSET", "SFWDSDZ"]
EXTRA_VAR_NUM = 6
ROOM_VAR_NUM = 3

GRU_UNIT = [(100, 0.5), (100, 0.5)]
AC_FC_UNIT = [(128, 0.4), (64, 0.3)]

# optimize-ac-temp
ROOM_SET_TEMP_RANGE = (20, 26)
SFWDSDZ_RANGE = (10, 14)
COMFORTABLE_ROOM_TEMP = 23
OB3_SCALE_FACTOR = 4  # objective 3 的量级会较小于ob1，ob2，因此需要一个scale factor
