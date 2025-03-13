import pandas as pd
from datetime import datetime
from collections import Counter
import os
import time
import requests
import json

# 获取当前日期信息
now = datetime.now()
today_date = now.strftime('%Y-%m-%d')
today_month = now.month
today_day = now.day
today_weekday = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'][now.weekday()]
report_filename = f"wangqitongji{now.strftime('%Y%m%d%H%M')}.html"
report_path = os.path.join("wangqitongji_html", report_filename)

# 打印今日日期信息
print(f"今日日期：{today_date}")
print(f"所属月份：{today_month}")
print(f"所属星期几：{today_weekday}\n")

# 读取数据
data_path = 'lottery_data.csv'
df = pd.read_csv(data_path)

# 转换日期列为 datetime 格式
df['开奖日期'] = pd.to_datetime(df['开奖日期'])
df = df.sort_values(by='开奖日期')
df['号码列表'] = df['开奖号码'].apply(lambda x: list(map(int, x.split(','))))

# 获取最新一期开奖号码
latest_draw = df.iloc[-1]
latest_numbers = list(map(int, latest_draw['开奖号码'].split(',')))

# 推算：基于数字位置的下一期分析
def analyze_next_digit(data, target_digit, position):
    next_numbers = []
    for i in range(len(data) - 1):
        current_numbers = list(map(int, data.iloc[i]['开奖号码'].split(',')))
        if current_numbers[position] == target_digit:
            next_draw = list(map(int, data.iloc[i + 1]['开奖号码'].split(',')))
            next_numbers.append(next_draw[position])
    return Counter(next_numbers).most_common()

# 根据频率推算
def get_number_frequency(data):
    if data.empty:
        return {}
    numbers = [int(num) for sublist in data['开奖号码'].str.split(',') for num in sublist]
    return Counter(numbers).most_common()

# 获取频率数据
def get_recent_5_draws():
    return df.sort_values(by='开奖日期', ascending=False).head(5)

def get_all_past_years_same_day_data():
    return df[(df['开奖日期'].dt.month == today_month) & (df['开奖日期'].dt.day == today_day)]

def get_same_day_in_all_months():
    return df[df['开奖日期'].dt.day == today_day]

def get_last_week_same_weekday_data():
    target_weekday = now.weekday()
    return df[df['开奖日期'].dt.weekday == target_weekday]

# 计算频率数据
recent_5_data = get_recent_5_draws()
recent_5_frequency = get_number_frequency(recent_5_data)

same_day_data = get_all_past_years_same_day_data()
same_day_frequency = get_number_frequency(same_day_data)

same_day_in_months_data = get_same_day_in_all_months()
same_day_in_months_frequency = get_number_frequency(same_day_in_months_data)

last_week_same_weekday_data = get_last_week_same_weekday_data()
last_week_frequency = get_number_frequency(last_week_same_weekday_data)

# 推算结果
hundred_pred = analyze_next_digit(df, latest_numbers[0], 0)  # 百位
ten_pred = analyze_next_digit(df, latest_numbers[1], 1)      # 十位
unit_pred = analyze_next_digit(df, latest_numbers[2], 2)     # 个位

def predict_next_number(hundred_pred, ten_pred, unit_pred):
    return [
        f"{hundred[0]}{ten[0]}{unit[0]}"
        for hundred in hundred_pred[:3]
        for ten in ten_pred[:3]
        for unit in unit_pred[:3]
    ]

predicted_numbers = predict_next_number(hundred_pred, ten_pred, unit_pred)

def predict_based_on_lowest_frequency(frequency, label):
    lowest_numbers = [str(num) for num, _ in sorted(frequency, key=lambda x: x[1])[:3]]
    return f"{label}推算：{','.join(lowest_numbers)}"

# 生成报告内容
def generate_report():
    html_content = f"""
    <!DOCTYPE html>
    <html lang="zh">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>TrendHunter Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
            ul {{ padding-left: 20px; }}
            p, li {{ line-height: 1.6; word-wrap: break-word; overflow-wrap: break-word; }}
            .container {{ max-width: 800px; margin: auto; }}
            .highlight {{ background-color: #f0f8ff; padding: 10px; border-radius: 5px; }}
            #passwordBox {{ margin-top: 50px; text-align: center; }}
            #passwordInput {{ padding: 10px; font-size: 16px; width: 200px; margin-bottom: 10px; }}
            #errorMessage {{ color: red; display: none; }}
        </style>
        <script>
            function checkPassword() {{
                let inputPassword = document.getElementById('passwordInput').value;
                if (inputPassword !== "6") {{
                    document.getElementById('errorMessage').style.display = 'block';
                }} else {{
                    document.getElementById('errorMessage').style.display = 'none';
                    document.getElementById('report-content').style.display = 'block';
                    document.getElementById('passwordBox').style.display = 'none';
                }}
            }}
        </script>
    </head>
    <body>
        <div id="passwordBox">
            <h2>请输入密码以查看内容：</h2>
            <input type="password" id="passwordInput" placeholder="输入密码">
            <br>
            <button onclick="checkPassword()">确认</button>
            <p id="errorMessage">密码错误！请重试。</p>
        </div>
        
        <div id="report-content" class="container" style="display: none;">
            <h1>推算报告</h1>
            <p>今日日期：{today_date}</p>
            <p>所属月份：{today_month}</p>
            <p>所属星期几：{today_weekday}</p>
            <p>上期开奖号码：{latest_numbers}</p>
            
            <h2>一、推算下期开奖号码结果：</h2>
            <ul>
                <li>{predict_based_on_lowest_frequency(recent_5_frequency, "近5期")}</li>
                <li>{predict_based_on_lowest_frequency(same_day_frequency, "往年今日")}</li>
                <li>{predict_based_on_lowest_frequency(same_day_in_months_frequency, "往月今日")}</li>
                <li>{predict_based_on_lowest_frequency(last_week_frequency, "以往同周")}</li>
                <li>历史数位： {','.join(predicted_numbers)}</li>
            </ul>
            
            <h2>二、历史统计简报数据</h2>
            <h3>1、根据上期号码的历史数字位置统计</h3>
            <p class="highlight">百位预测：{[f"{num}: {count}次" for num, count in hundred_pred[:3]]}</p>
            <p class="highlight">十位预测：{[f"{num}: {count}次" for num, count in ten_pred[:3]]}</p>
            <p class="highlight">个位预测：{[f"{num}: {count}次" for num, count in unit_pred[:3]]}</p>
            
            <h3>2、“最近5期”数字频率：</h3>
            <p>频率排序：{','.join(map(str, [num for num, _ in recent_5_frequency]))}</p>
            <p>频率次数：{[f"{num}: {count}次" for num, count in recent_5_frequency]}</p>
            
            <h3>3、“往年今日”数字频率：</h3>
            <p>频率排序：{','.join(map(str, [num for num, _ in same_day_frequency]))}</p>
            <p>频率次数：{[f"{num}: {count}次" for num, count in same_day_frequency]}</p>
            
            <h3>4、“往月今日”数字频率：</h3>
            <p>频率排序：{','.join(map(str, [num for num, _ in same_day_in_months_frequency]))}</p>
            <p>频率次数：{[f"{num}: {count}次" for num, count in same_day_in_months_frequency]}</p>
            
            <h3>5、“以往同周”数字频率：</h3>
            <p>频率排序：{','.join(map(str, [num for num, _ in last_week_frequency]))}</p>
            <p>频率次数：{[f"{num}: {count}次" for num, count in last_week_frequency]}</p>
            
            <h2>三、历史开奖号码简报</h2>
            <h3>1、最近5期开奖号码统计：</h3>
            <p>{'<br>'.join([f"{row['开奖日期'].strftime('%Y-%m-%d')} - 开奖号码：{row['开奖号码']}" for _, row in recent_5_data.iterrows()])}</p>
            
            <h3>2、往年今日开奖号码统计：</h3>
            <p>{'<br>'.join([f"{row['开奖日期'].strftime('%Y-%m-%d')} - 开奖号码：{row['开奖号码']}" for _, row in same_day_data.iterrows()])}</p>
        </div>
    </body>
    </html>
    """
    return html_content


# 保存报告
os.makedirs("wangqitongji_html", exist_ok=True)
with open(report_path, "w", encoding="utf-8") as file:
    file.write(generate_report())

print(f"报告已生成：{report_path}")



# 发送微信模板消息
thing4 = predict_based_on_lowest_frequency(recent_5_frequency, "近5期")
thing31 = predict_based_on_lowest_frequency(same_day_frequency, "往年今日")
thing40 = predict_based_on_lowest_frequency(same_day_in_months_frequency, "往月今日")
thing5 = predict_based_on_lowest_frequency(last_week_frequency, "以往同周")
remark = ', '.join(predicted_numbers)  # 这部分可以保留，因为 predicted_numbers 是列表
t_url = f"https://3d.13982.com/" + report_path.replace("\\", "/")
t_url = t_url.replace(".html", "")

print(f"thing4: {thing4}")
print(f"thing31: {thing31}")
print(f"thing40: {thing40}")
print(f"thing5: {thing5}")
print(f"remark: {remark}")
print(f"t_url: {t_url}")


url = "http://140.238.6.167:5001/send_template"
to_users = ["oXUv66MibUi7VInLBf7AHqMIY438", "oXUv66DvDIoQG39Vnspwj97QVLn4"]

for to_user in to_users:
    while True:
        payload = json.dumps({
            "to_user": to_user,
            "template_id" : "nyQ-0vYb0bl5EZWT2OK8jX46NNsnrzWXxminYjO2Y8A",  # 去掉逗号，确保是字符串
            "data": {
                "thing4": thing4,
                "thing31": thing31,
                "thing40": thing40,
                "thing5": thing5,
                "remark": remark
            },
            "url_params": {
                "order_id": "395248",
                "user": "苏"
            },
            "url": t_url
        })
        headers = {
            'x-api-key': 'sw63828',
            'User-Agent': 'github/actions',
            'Content-Type': 'application/json'
        }

        response = requests.request("POST", url, headers=headers, data=payload)
        print("response.text:" + response.text)

        # 解析 response.text 并检查 status
        try:
            response_data = json.loads(response.text)
            if response_data.get("status") == "success":
                print(f"消息发送成功给用户 {to_user}")
                break  # 跳出循环，发送下一个用户
            else:
                print(f"消息发送失败，5秒后重试... 用户 {to_user}")
                time.sleep(5)
        except json.JSONDecodeError:
            print("响应解析失败，5秒后重试...")
            time.sleep(5)