import pandas as pd
import numpy as np
import os
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GRU, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
from math import sqrt
from datetime import datetime
import pytz
from zoneinfo import ZoneInfo
import requests
import json
# -------------------
# 禁用 GPU
# -------------------
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# -------------------
# 配置
# -------------------
sequence_length = 10  # 时间序列长度
model_path_ge = "TrendHunter/lstm_model_ge.keras"
model_path_shi = "TrendHunter/lstm_model_shi.keras"
model_path_bai = "TrendHunter/lstm_model_bai.keras"

# 日志设置
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -------------------
# 数据加载与特征工程
# -------------------
logger.info("开始分析数据...")

try:
    historical_file_path = "lottery_data.csv"
    if not os.path.exists(historical_file_path):
        raise FileNotFoundError(f"文件 {historical_file_path} 不存在！")

    # 加载并排序数据
    df = pd.read_csv(historical_file_path)
    df = df.sort_values(by="期号", ascending=True).reset_index(drop=True)

    # 修改此处列名为实际列名 '开奖日期'，并解析日期格式
    df['开奖日期'] = pd.to_datetime(df['开奖日期'], format="%Y-%m-%d")

    # 分拆开奖号码为个位、十位、百位
    df['开奖号码'] = df['开奖号码'].apply(lambda x: [int(i) for i in x.split(',')])
    df['个位'] = df['开奖号码'].apply(lambda x: x[0])
    df['十位'] = df['开奖号码'].apply(lambda x: x[1])
    df['百位'] = df['开奖号码'].apply(lambda x: x[2])

    # 统计各数字出现频率
    all_numbers = np.concatenate([df['个位'].values, df['十位'].values, df['百位'].values])
    number_frequency = pd.Series(all_numbers).value_counts().sort_values(ascending=False)

    logger.info("数据分析完成！")
except Exception as e:
    logger.error(f"数据加载失败：{e}")
    exit(1)


# -------------------
# 时间序列生成
# -------------------
def prepare_data(df, column):
    """
    将目标列转换为时间序列数据，适用于 LSTM/GRU 模型。
    """
    try:
        numbers = df[column].values
        X, y = [], []
        for i in range(len(numbers) - sequence_length):
            X.append(numbers[i:i + sequence_length])
            y.append(numbers[i + sequence_length])
        X = np.array(X).reshape(-1, sequence_length, 1)
        y = np.array(y)
        return X, y
    except Exception as e:
        logger.error(f"时间序列数据准备失败：{e}")
        raise

logger.info("准备时间序列数据...")
X_ge, y_ge = prepare_data(df, '个位')
X_shi, y_shi = prepare_data(df, '十位')
X_bai, y_bai = prepare_data(df, '百位')

# -------------------
# GRU 模型构建
# -------------------
def build_deep_gru_model(input_shape):
    model = Sequential([
        Input(shape=input_shape),
        GRU(256, activation='relu', return_sequences=True),
        Dropout(0.3),
        GRU(128, activation='relu', return_sequences=True),
        Dropout(0.3),
        GRU(64, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.0003), loss='mean_squared_error')
    return model

# -------------------
# 模型训练与评估
# -------------------
def train_and_evaluate_model(X, y, model_path):
    """
    训练并评估 GRU 模型。
    """
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if os.path.exists(model_path):
            model = load_model(model_path, compile=False)
            model.compile(optimizer=Adam(learning_rate=0.0003), loss='mean_squared_error')
            logger.info(f"加载模型成功：{model_path}")
        else:
            model = build_deep_gru_model((sequence_length, 1))
            logger.info("构建新模型...")

        # 添加早停回调
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=60, batch_size=64, verbose=1, callbacks=[early_stopping])

        # 模型性能评估
        y_pred = model.predict(X_test).flatten()
        rmse = sqrt(mean_squared_error(y_test, y_pred))
        logger.info(f"模型评估 - RMSE: {rmse:.4f}")

        # 保存模型
        model.save(model_path)
        logger.info(f"模型已保存至：{model_path}")
        return model, rmse
    except Exception as e:
        logger.error(f"模型训练失败：{e}")
        exit(1)

logger.info("训练深度 GRU 模型...")
model_ge, rmse_ge = train_and_evaluate_model(X_ge, y_ge, model_path_ge)
model_shi, rmse_shi = train_and_evaluate_model(X_shi, y_shi, model_path_shi)
model_bai, rmse_bai = train_and_evaluate_model(X_bai, y_bai, model_path_bai)

# -------------------
# 下一期预测
# -------------------
logger.info("开始预测下一期号码...")
last_ge = X_ge[-1].reshape(1, sequence_length, 1)
last_shi = X_shi[-1].reshape(1, sequence_length, 1)
last_bai = X_bai[-1].reshape(1, sequence_length, 1)

next_prediction_ge = model_ge.predict(last_ge)[0][0]
next_prediction_shi = model_shi.predict(last_shi)[0][0]
next_prediction_bai = model_bai.predict(last_bai)[0][0]

logger.info(f"下一期预测号码：个位：{round(next_prediction_ge)}，十位：{round(next_prediction_shi)}，百位：{round(next_prediction_bai)}")

# 获取出现频率最高的数字
sorted_numbers = number_frequency.index.tolist()[:6]
# sorted_numbers = number_frequency.index.tolist()
now = datetime.now(ZoneInfo("Asia/Shanghai"))
today_date = now.strftime('%Y-%m-%d')

# -------------------
# 微信模板消息
# -------------------
# 定义需要发送的用户列表
to_users = ["oXUv66MibUi7VInLBf7AHqMIY438", "oXUv66DvDIoQG39Vnspwj97QVLn4"]

# 循环发送模板消息
for to_user in to_users:
    template_data = {
        "to_user": to_user,
        "template_id" : "nyQ-0vYb0bl5EZWT2OK8jX46NNsnrzWXxminYjO2Y8A",  # 去掉逗号，确保是字符串
        "data": {
            "thing4": f"{today_date}-模型分析结果",
            "thing31": f"模型结果({round(next_prediction_ge)},{round(next_prediction_shi)},{round(next_prediction_bai)})",
            "thing40": f"模型评分({round(rmse_ge, 2)},{round(rmse_shi, 2)},{round(rmse_bai, 2)})",
            "thing5": "频值-" + ",".join(map(str, sorted_numbers)),
            "remark": "点击查看详情: test"
        },
        "url": "https://3d.13982.com/",
        "url_params": {
            "order_id": "395248",
            "user": "苏"
        }
    }

    # 发送 POST 请求
    try:
        url = "http://140.238.6.167:5001/send_template"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": "sw63828"
        }
        response = requests.post(url, headers=headers, data=json.dumps(template_data))
        response.raise_for_status()
        response_json = response.json()
        print(f"模板消息已发送至用户 {to_user}，响应状态码: {response.status_code}")
        print(f"响应内容: {json.dumps(response_json, ensure_ascii=False, indent=2)}")
    except requests.exceptions.RequestException as e:
        print(f"发送给用户 {to_user} 时失败: {e}")
    except ValueError:
        print(f"发送给用户 {to_user} 的响应内容不是有效的 JSON 格式！")


# -------------------
# 保存报告
# -------------------
# 修改报告生成中的列名，确保一致
# 获取北京时间
beijing_tz = pytz.timezone('Asia/Shanghai')
beijing_time = datetime.now(beijing_tz).strftime('%Y-%m-%d %H:%M:%S')

report = f"""
# 福彩3D 分析与预测报告

## 1. 报告生成时间
- {beijing_time}

## 2. 历史数据摘要
- 总记录数：{len(df)}
- 数据时间范围：{df['开奖日期'].min().strftime('%Y-%m-%d')} 至 {df['开奖日期'].max().strftime('%Y-%m-%d')}
- 最近一期数据：
  - **期号**：{df['期号'].iloc[-1]}
  - **开奖时间**：{df['开奖日期'].iloc[-1]}  # 这里改为 '开奖日期'
  - **开奖号码**：{df['个位'].iloc[-1]}{df['十位'].iloc[-1]}{df['百位'].iloc[-1]}

## 3. 模型评估结果
- 个位模型 RMSE：{rmse_ge:.4f}
- 十位模型 RMSE：{rmse_shi:.4f}
- 百位模型 RMSE：{rmse_bai:.4f}

## 4. 下一期预测
- **个位**：{round(next_prediction_ge)}
- **十位**：{round(next_prediction_shi)}
- **百位**：{round(next_prediction_bai)}

## 5. 数字出现频率
{number_frequency.to_string()}
"""

report_file = f"TrendHunter/Analyze_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
try:
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, 'w', encoding='utf-8') as file:
        file.write(report)
    logger.info(f"报告已保存至：{report_file}")
except Exception as e:
    logger.error(f"报告保存失败：{e}")

# -------------------
# 邮件发送
# -------------------
try:
    sender_email = "157574738@qq.com"
    sender_password = "haqqlsniftgtbgjg"
    receiver_email = "395240832@qq.com"
    smtp_server = "smtp.qq.com"
    smtp_port = 587

    subject = f"3D分析报告 - {os.path.basename(report_file)}"
    with open(report_file, 'r', encoding='utf-8') as file:
        email_body = file.read()

    # 创建邮件
    message = MIMEMultipart()
    message['From'] = sender_email
    message['To'] = receiver_email
    message['Subject'] = subject
    message.attach(MIMEText(email_body, 'plain', 'utf-8'))

    # 发送邮件
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(sender_email, sender_password)
    server.send_message(message)
    server.quit()

    logger.info(f"邮件已发送至 {receiver_email}")
except Exception as e:
    logger.error(f"邮件发送失败：{e}")
