import requests
import pandas as pd
import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo

# 定义北京时间时区
beijing_timezone = ZoneInfo("Asia/Shanghai")

# API 基础 URL
BASE_URL = "https://mix.lottery.sina.com.cn/gateway/index/entry"

# 请求参数
PARAMS = {
    "format": "json",
    "__caller__": "wap",
    "__version__": "1.0.0",
    "__verno__": "10000",
    "cat1": "gameOpenList",
    "lottoType": "102",     # 福彩3D类型
    "paginationType": "1",  # 分页类型
    "page": "1",            # 固定请求第一页
    "pageSize": "20",       # 每页数据数量（固定为20，实际只用第一条）
    "dpc": "1"              # 固定参数
}

# 数据文件路径
output_file = "lottery_data.csv"

# 加载现有数据
if os.path.exists(output_file):
    try:
        df_existing = pd.read_csv(output_file, dtype={'期号': str, '开奖日期': str, '开奖号码': str})
        print(f"已加载 {len(df_existing)} 条现有数据。")
    except Exception as e:
        print(f"读取现有 CSV 文件时出错: {e}")
        exit(1)
else:
    # 如果文件不存在，创建一个空的 DataFrame
    df_existing = pd.DataFrame(columns=['期号', '开奖日期', '开奖号码'])
    print("CSV 文件不存在，将创建新文件。")

# 请求第一页数据
print("开始采集数据...")
try:
    response = requests.get(BASE_URL, params=PARAMS, timeout=10)
    response.raise_for_status()  # 检查请求是否成功
except requests.exceptions.RequestException as e:
    print(f"请求失败: {e}")
    exit(1)

try:
    data = response.json()
except ValueError:
    print("响应内容不是有效的 JSON 格式！")
    exit(1)

print("成功获取 API 响应。")

if data.get("result") and data["result"].get("data"):
    # 只获取第一页的第一条数据
    first_item = data["result"]["data"][0]
    issue_no = str(first_item.get("issueNo", "未知期号")).strip()
    open_time_full = first_item.get("openTime", "未知日期")
    open_results = first_item.get("openResults", [])

    print(f"解析到的期号: {issue_no}")
    print(f"开奖时间: {open_time_full}")
    print(f"开奖号码: {open_results}")

    # 确保 open_results 是列表
    if not isinstance(open_results, list):
        print(f"警告: 'openResults' 不是列表，当前类型: {type(open_results)}")
        open_results = [str(open_results)]
    else:
        # 确保所有元素都是字符串
        open_results = [str(num).strip() for num in open_results]

    # 仅提取日期部分
    try:
        open_date = datetime.strptime(open_time_full, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
    except ValueError:
        if " " in open_time_full:
            open_date = open_time_full.split(" ")[0]
        else:
            open_date = open_time_full  # 保留原值

    print(f"格式化后的开奖日期: {open_date}")

    # 检查 issue_no 是否已存在
    if issue_no not in df_existing['期号'].values:
        # 准备新数据行
        new_row = {
            '期号': issue_no,
            '开奖日期': open_date,
            '开奖号码': ','.join(open_results)
        }

        # 将新行添加到现有数据的顶部
        df_updated = pd.concat([pd.DataFrame([new_row]), df_existing], ignore_index=True)

        # 按照期号降序排序
        df_updated['期号'] = df_updated['期号'].astype(str)  # 确保期号为字符串
        df_updated_sorted = df_updated.sort_values(by='期号', ascending=False).reset_index(drop=True)

        # 保存回 CSV 文件
        try:
            df_updated_sorted.to_csv(output_file, index=False, encoding='utf-8')
            print(f"采集到新数据并已插入到顶部: 期号={issue_no}, 日期={open_date}, 开奖号码={','.join(open_results)}")
        except Exception as e:
            print(f"保存数据到 CSV 文件时发生错误: {e}")
    else:
        print(f"数据已存在，无需追加: 期号={issue_no}, 日期={open_date}, 开奖号码={','.join(open_results)}")

    # 获取上一期和上上期数据
    df_existing_sorted = df_updated_sorted.sort_values(by='期号', ascending=False).reset_index(drop=True)

    last_issue = None
    second_last_issue = None

    current_index = df_existing_sorted[df_existing_sorted['期号'] == issue_no].index[0]
    if current_index + 1 < len(df_existing_sorted):
        last_issue = df_existing_sorted.iloc[current_index + 1]
    if current_index + 2 < len(df_existing_sorted):
        second_last_issue = df_existing_sorted.iloc[current_index + 2]

    # 打印调试信息
    if last_issue is not None:
        print(f"上一期: 期号={last_issue['期号']}, 开奖号码={last_issue['开奖号码']}")
    else:
        print("没有找到上一期数据。")

    if second_last_issue is not None:
        print(f"上上期: 期号={second_last_issue['期号']}, 开奖号码={second_last_issue['开奖号码']}")
    else:
        print("没有找到上上期数据。")

    # 定义需要发送的用户列表
    to_users = ["oXUv66MibUi7VInLBf7AHqMIY438", "oXUv66DvDIoQG39Vnspwj97QVLn4"]

    # 循环发送模板消息
    for to_user in to_users:
        template_data = {
            "to_user": to_user,
            "template_id" : "nyQ-0vYb0bl5EZWT2OK8jX46NNsnrzWXxminYjO2Y8A",  # 去掉逗号，确保是字符串
            "data": {
                "thing4": open_date + "-开奖信息",
                "thing31": ','.join(open_results),
                "thing40": f"上期({last_issue['期号']}):{last_issue['开奖号码']}" if last_issue is not None else "上期号码: 无",
                "thing5": f"上上期({second_last_issue['期号']}):{second_last_issue['开奖号码']}" if second_last_issue is not None else "上上期号码: 无",
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
