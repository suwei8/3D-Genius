import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input

# 数据加载与预处理
def load_data(file_path):
    lottery_data = pd.read_csv(file_path)
    lottery_data[['百位', '十位', '个位']] = lottery_data['开奖号码'].str.split(',', expand=True).astype(int)
    lottery_data['和值'] = lottery_data['百位'] + lottery_data['十位'] + lottery_data['个位']
    lottery_data['跨度'] = lottery_data[['百位', '十位', '个位']].max(axis=1) - lottery_data[['百位', '十位', '个位']].min(axis=1)
    return lottery_data

# 特征工程：生成周期性特征、热号冷号等
def generate_features(lottery_data, n_lags=5):
    features = []
    for i in range(n_lags, len(lottery_data)):
        row = lottery_data.iloc[i]
        lag_rows = lottery_data.iloc[i - n_lags:i]

        # 计算热号和冷号
        hot_numbers = lottery_data.iloc[:i][['百位', '十位', '个位']].stack().value_counts().head(3).index.tolist()
        cold_numbers = lottery_data.iloc[:i][['百位', '十位', '个位']].stack().value_counts().tail(3).index.tolist()

        feature = {
            '和值': row['和值'],
            '跨度': row['跨度'],
            '热号': sum([1 for num in row[['百位', '十位', '个位']] if num in hot_numbers]),
            '冷号': sum([1 for num in row[['百位', '十位', '个位']] if num in cold_numbers]),
        }

        # 最近 n_lags 期的特征
        for j in range(n_lags):
            feature.update({
                f'lag_{j+1}_和值': lag_rows.iloc[j]['和值'],
                f'lag_{j+1}_跨度': lag_rows.iloc[j]['跨度'],
            })

        features.append(feature)
    return pd.DataFrame(features)

# 准备 LSTM 数据
def prepare_lstm_data(features, target, n_lags=5):
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    X, y = [], []
    for i in range(n_lags, len(features_scaled)):
        X.append(features_scaled[i - n_lags:i])  # 最近 n_lags 期作为输入
        y.append(target[i])  # 当前期作为输出

    X, y = np.array(X), np.array(y)  # 转换为数组
    return X, y, scaler

# 构建 LSTM 模型
def build_lstm_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64, return_sequences=False),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# 训练机器学习模型
def train_ml_model(X_train, y_train, model_type='XGB'):
    if model_type == 'XGB':
        model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6)
    elif model_type == 'RF':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# 主函数
if __name__ == "__main__":
    file_path = "./lottery_test_data.csv"  # 数据文件路径
    lottery_data = load_data(file_path)

    # 确定下一期目标
    next_period = lottery_data['期号'].max() + 1

    # 特征工程
    n_lags = 5
    features = generate_features(lottery_data, n_lags)
    target = lottery_data.iloc[n_lags:][['百位', '十位', '个位']].apply(lambda x: int(''.join(map(str, x))), axis=1)

    # 准备 LSTM 数据
    X, y, scaler = prepare_lstm_data(features.values, target.values, n_lags)

    # 划分训练集和验证集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 训练 LSTM 模型
    lstm_model = build_lstm_model((n_lags, X.shape[2]))
    lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2, validation_data=(X_test, y_test))

    # 综合机器学习模型（XGBoost 和随机森林）
    X_flat_train = X_train.reshape(X_train.shape[0], -1)  # 平铺 LSTM 特征用于机器学习模型
    X_flat_test = X_test.reshape(X_test.shape[0], -1)

    xgb_model = train_ml_model(X_flat_train, y_train, model_type='XGB')
    rf_model = train_ml_model(X_flat_train, y_train, model_type='RF')

    # 各模型独立预测
    lstm_predictions = [int(round(lstm_model.predict(X_test)[i][0])) for i in range(10)]
    xgb_predictions = [int(round(xgb_model.predict(X_flat_test)[i])) for i in range(10)]
    rf_predictions = [int(round(rf_model.predict(X_flat_test)[i])) for i in range(10)]

    # 输出独立预测结果
    print(f"Predicted 3D Numbers for next period {next_period}:")
    print("LSTM Predictions:", [str(p).zfill(3) for p in lstm_predictions])
    print("XGBoost Predictions:", [str(p).zfill(3) for p in xgb_predictions])
    print("RandomForest Predictions:", [str(p).zfill(3) for p in rf_predictions])