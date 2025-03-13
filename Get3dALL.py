import requests
import csv
import os
from datetime import datetime

# 获取脚本所在的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
output_file = os.path.join(script_dir, "lottery_data.csv")

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
    "pageSize": "20",       # 每页数据数量
    "dpc": "1"              # 固定参数
}

# 数据存储列表
lottery_info = []

# 删除现有的 CSV 文件（如果存在）
if os.path.exists(output_file):
    try:
        os.remove(output_file)
        print(f"已删除现有文件: {output_file}")
    except Exception as e:
        print(f"删除文件 {output_file} 失败: {e}")
        # 根据需要，可以选择在这里退出脚本
        # import sys
        # sys.exit(1)

# 分页采集数据
print("开始采集数据...")
page = 1
while True:
    print(f"正在请求第 {page} 页数据...")
    PARAMS["page"] = page  # 更新页码
    try:
        response = requests.get(BASE_URL, params=PARAMS, timeout=10)
    except requests.exceptions.RequestException as e:
        print(f"请求第 {page} 页数据时发生错误: {e}")
        break

    if response.status_code == 200:
        try:
            data = response.json()
        except ValueError:
            print("响应内容不是有效的 JSON 格式，停止采集！")
            break

        if data.get("result") and data["result"].get("data"):
            items = data["result"]["data"]  # 获取开奖数据列表
            for item in items:
                issue_no = item.get("issueNo", "未知期号")
                open_time_full = item.get("openTime", "未知日期")

                # 使用 datetime 模块验证和格式化日期
                try:
                    open_date = datetime.strptime(open_time_full, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d")
                except ValueError:
                    if " " in open_time_full:
                        open_date = open_time_full.split(" ")[0]
                    else:
                        open_date = open_time_full  # 保留原值

                open_results = item.get("openResults", [])
                lottery_info.append({
                    "期号": issue_no,
                    "开奖日期": open_date,
                    "开奖号码": open_results
                })
                print(f"采集到数据: 期号={issue_no}, 日期={open_date}, 开奖号码={','.join(open_results)}")

            # 检查是否还有更多页
            total_pages = data["result"]["pagination"].get("totalPage", 1)
            if page >= total_pages:
                print("所有数据采集完成！")
                break

            page += 1
        else:
            print("无法解析数据，停止采集！")
            break
    else:
        print(f"请求失败，状态码: {response.status_code}")
        break

# 保存数据到 CSV 文件
try:
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['期号', '开奖日期', '开奖号码'])  # 表头
        for info in lottery_info:
            writer.writerow([info['期号'], info['开奖日期'], ','.join(info['开奖号码'])])
    print(f"全部数据已采集完成，共采集到 {len(lottery_info)} 组数据，已保存为 {output_file}")

    # 验证文件内容
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"文件 {output_file} 已包含 {len(lines)-1} 条记录（不包括表头）。")
    else:
        print(f"文件 {output_file} 未找到，请检查权限和路径。")
except Exception as e:
    print(f"保存数据到 CSV 文件时发生错误: {e}")
