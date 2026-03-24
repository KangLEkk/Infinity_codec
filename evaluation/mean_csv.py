import pandas as pd

# 1. 读取 CSV 文件
file_path = '/workspace/mean_results_by_k_transmit.csv'
df = pd.read_csv(file_path)

# 2. 按照 k_transmit 分组并计算所有数值列的平均值
# numeric_only=True 确保脚本只处理数字列，忽略文件名等字符串
mean_results = df.groupby('k_transmit').mean(numeric_only=True)

# 3. 打印结果到控制台
print("按 k_transmit 分组的平均值统计：")
print(mean_results)

# 4. 将结果保存到新的 CSV 文件中
output_path = 'mean_results_by_k_transmit3.csv'
mean_results.to_csv(output_path)
print(f"\n结果已成功保存至: {output_path}")