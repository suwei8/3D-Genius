import pandas as pd
import numpy as np
import os
import requests
import json
from datetime import datetime
from zoneinfo import ZoneInfo

# 定义北京时间时区
beijing_timezone = ZoneInfo("Asia/Shanghai")
now = datetime.now(beijing_timezone)


def generate_html_table(dataframe, columns):
    """将 Pandas DataFrame 转换为 HTML 表格"""
    html_table = "<table border='1' style='border-collapse: collapse; width: 100%; text-align: center;'>"
    html_table += "<tr>" + "".join(f"<th>{col}</th>" for col in columns) + "</tr>"
    for _, row in dataframe.iterrows():
        html_table += "<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>"
    html_table += "</table>"
    return html_table


def analyze_and_predict_lottery_data(file_path):
    try:
        data = pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"文件读取错误: {e}")

    try:
        data['开奖日期'] = pd.to_datetime(data['开奖日期']).dt.tz_localize('Asia/Shanghai')
    except Exception as e:
        raise Exception(f"解析开奖日期错误: {e}")

    try:
        data['开奖号码'] = data['开奖号码'].apply(lambda x: list(map(int, x.split(','))))
    except Exception as e:
        raise Exception(f"解析开奖号码错误: {e}")

    data.sort_values('期号', inplace=True)
    all_numbers = [num for nums in data['开奖号码'] for num in nums]
    last_30_numbers = [num for nums in data['开奖号码'][-30:] for num in nums]

    overall_freq = pd.Series(all_numbers).value_counts(normalize=True).sort_values(ascending=False).reset_index()
    overall_freq.columns = ['号码', '频率']
    recent_freq = pd.Series(last_30_numbers).value_counts(normalize=True).sort_values(ascending=False).reset_index()
    recent_freq.columns = ['号码', '频率']

    predicted_numbers = []
    for position in range(3):
        weights = overall_freq['频率'] + recent_freq['频率']
        weights /= weights.sum()
        predicted_number = np.random.choice(range(10), p=weights)
        predicted_numbers.append(predicted_number)

    latest_draw = data.iloc[-1]
    latest_draw_numbers = latest_draw['开奖号码']
    latest_draw_info = f"<b>期号</b>: {latest_draw['期号']} <br><b>开奖时间</b>: {latest_draw['开奖日期'].strftime('%Y-%m-%d')} <br><b>开奖号码</b>: {','.join(map(str, latest_draw_numbers))}"

    data['对子'] = data['开奖号码'].apply(lambda x: len(set(x)) == 2)
    pair_draws = data[data['对子']].sort_values('开奖日期')

    if not pair_draws.empty:
        latest_pair_draw = pair_draws.iloc[-1]
        days_since_pair = (now - latest_pair_draw['开奖日期']).days
        latest_pair_draw_numbers = latest_pair_draw['开奖号码']
    else:
        days_since_pair = "无记录"
        latest_pair_draw_numbers = "无记录"

    return {
        'predicted_numbers': predicted_numbers,
        'latest_draw_info': latest_draw_info,
        'latest_draw_numbers': latest_draw_numbers,
        'days_since_pair': days_since_pair,
        'latest_pair_draw_numbers': latest_pair_draw_numbers
    }


if __name__ == "__main__":
    file_path = 'lottery_data.csv'
    try:
        results = analyze_and_predict_lottery_data(file_path)
        today_date = now.strftime('%Y-%m-%d')
        url = "http://140.238.6.167:5001/send_template"
        headers = {
            "Content-Type": "application/json",
            "x-api-key": "sw63828"
        }

        to_users = [
            "oXUv66MibUi7VInLBf7AHqMIY438",
            "oXUv66DvDIoQG39Vnspwj97QVLn4",
            "oXUv66HUVNyZ0Hd8RWKmkVV1dkAs"
        ]

        common_data = {
            "data": {
                "thing4": f"{today_date}-订单计划数据",
                "thing31": f"今天推算：{','.join(map(str, results['predicted_numbers']))}",
                "thing40": f"昨天数据：{','.join(map(str, results['latest_draw_numbers']))}",
                "thing5": f"上期双子：已{results['days_since_pair']}天, {','.join(map(str, results['latest_pair_draw_numbers']))}",
                "remark": "点击查看详情",
            },
            "url": "https://3d.13982.com/"
        }

        for user_id in to_users:
            data = {
                "template_id":"nyQ-0vYb0bl5EZWT2OK8jX46NNsnrzWXxminYjO2Y8A",  # 去掉逗号，确保是字符串
                "to_user": user_id,
                **common_data
            }

            response = requests.post(url, headers=headers, data=json.dumps(data))
            if response.status_code == 200:
                print(f"消息成功发送给用户 {user_id}")
            else:
                print(f"HTTP错误 {response.status_code} 发送给用户 {user_id}: {response.text}")

    except Exception as e:
        print(f"运行脚本时出错: {e}")
