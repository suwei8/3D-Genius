import pandas as pd

# 1. 读取 lottery_data.csv 文件
df = pd.read_csv('lottery_data.csv')

# 2. 定义一个函数，用于计算 "0.236结果" 和 "0.206结果"
def calculate_result(lottery_number, multiplier):
    # 将 "开奖号码" 字符串中的逗号去除，转换为整数
    number = int(lottery_number.replace(',', ''))
    return round(number * multiplier, 3)

# 3. 在原数据框中增加新列
df['0.236结果'] = df['开奖号码'].apply(lambda x: calculate_result(x, 0.236))
df['0.206结果'] = df['开奖号码'].apply(lambda x: calculate_result(x, 0.206))

# 4. 将更新后的数据框保存为 take_lottery_data.csv
df.to_csv('take_lottery_data.csv', index=False)

print("新文件 'take_lottery_data.csv' 已保存成功！")
