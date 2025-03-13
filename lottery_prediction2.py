import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention

# 数据加载与预处理
def load_data(file_path):
    lottery_data = pd.read_csv(file_path)
    lottery_data[['百位', '十位', '个位']] = lottery_data['开奖号码'].str.split(',', expand=True).astype(int)
    lottery_data['和值'] = lottery_data['百位'] + lottery_data['十位'] + lottery_data['个位']
    lottery_data['跨度'] = lottery_data[['百位', '十位', '个位']].max(axis=1) - lottery_data[['百位', '十位', '个位']].min(axis=1)
    return lottery_data

# 特征工程
def generate_features(lottery_data, n_lags=5):
    features = []
    for i in range(n_lags, len(lottery_data)):
        row = lottery_data.iloc[i]
        lag_rows = lottery_data.iloc[i - n_lags:i]

        # 计算特征
        feature = {
            '和值': row['和值'],
            '跨度': row['跨度'],
        }

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
def build_lstm_with_attention(input_shape):
    inputs = Input(shape=input_shape)
    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    attention = Attention()([x, x])  # Attention 层
    x = LSTM(64)(attention)
    x = Dropout(0.2)(x)
    outputs = Dense(1, activation='linear')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model

# 更新 generate_diverse_predictions 函数
def generate_diverse_predictions(model, X_test, lottery_data, num_predictions=10):
    predictions = []
    historical_distribution = lottery_data[['百位', '十位', '个位']].stack().value_counts(normalize=True)

    for _ in range(num_predictions):
        noise = np.random.normal(0, 1, size=X_test.shape) * np.std(X_test)  # 根据标准差生成噪声
        pred = model.predict(X_test + noise).flatten()
        predictions.extend(pred)

    # 根据历史分布筛选并去重
    filtered_predictions = [
        int(round(p)) for p in predictions
        if int(round(p)) in historical_distribution.index and 0 <= p <= 999
    ]
    
    # 根据历史概率排序，并返回前 num_predictions 个结果
    filtered_predictions = sorted(
        list(set(filtered_predictions)),
        key=lambda x: historical_distribution.get(x, 0),
        reverse=True
    )
    return filtered_predictions[:num_predictions]

# 主程序更新
if __name__ == "__main__":
    file_path = "./lottery_test_data.csv"
    lottery_data = load_data(file_path)

    # 数据分割
    train_size = int(len(lottery_data) * 2 / 3)
    train_data = lottery_data[:train_size]
    test_data = lottery_data[train_size:]

    # 特征工程
    n_lags = 5
    train_features = generate_features(train_data, n_lags)
    test_features = generate_features(test_data, n_lags)

    train_target = train_data.iloc[n_lags:][['百位', '十位', '个位']].apply(lambda x: int(''.join(map(str, x))), axis=1)
    test_target = test_data.iloc[n_lags:][['百位', '十位', '个位']].apply(lambda x: int(''.join(map(str, x))), axis=1)

    # 准备 LSTM 数据
    X_train, y_train, scaler = prepare_lstm_data(train_features.values, train_target.values, n_lags)
    X_test, y_test, _ = prepare_lstm_data(test_features.values, test_target.values, n_lags)

    # 训练 LSTM 模型
    lstm_model = build_lstm_with_attention((n_lags, X_train.shape[2]))
    lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=2, validation_data=(X_test, y_test))

    # 使用改进后的扰动逻辑生成多样化预测
    diverse_predictions = generate_diverse_predictions(lstm_model, X_test, lottery_data, num_predictions=10)
    print("Diverse Predictions:", diverse_predictions)