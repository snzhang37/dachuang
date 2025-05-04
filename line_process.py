import pandas as pd

distance_margin = 800
ots_margin = 10
osnr_margin = 0.01

file_name = 'example4'

# 读取 CSV 文件
file_path = './example/' + file_name + '.oms.csv'
df = pd.read_csv(file_path)

# 对每一行中的 omsId 和 remoteOmsId 进行排序，使边的方向一致
df['sorted_omsId'] = df[['omsId', 'remoteOmsId']].min(axis=1)
df['sorted_remoteOmsId'] = df[['omsId', 'remoteOmsId']].max(axis=1)

# 删除排序后重复的边
df_filtered = df.drop_duplicates(subset=['sorted_omsId', 'sorted_remoteOmsId'], keep='first')

# 删除辅助列，保持原文件格式
df_filtered = df_filtered.drop(columns=['sorted_omsId', 'sorted_remoteOmsId'])

# 对每一行中的 src 和 snk 进行排序，使边的方向一致
df['sorted_src'] = df[['src', 'snk']].min(axis=1)
df['sorted_snk'] = df[['src', 'snk']].max(axis=1)

# 删除排序后重复的边
df_filtered = df.drop_duplicates(subset=['sorted_src', 'sorted_snk'], keep='first')

# 删除辅助列，保持原文件格式
df_filtered = df_filtered.drop(columns=['sorted_src', 'sorted_snk'])


# 检查列是否存在
required_columns = ['distance', 'ots', 'osnr']
for col in required_columns:
    if col not in df_filtered.columns:
        raise ValueError(f"Column '{col}' not found in the CSV file.")

# 筛选条件：删除任意列大于指定阈值的行
df_filtered = df_filtered[(df_filtered['distance'] <= distance_margin) &
                 (df_filtered['ots'] <= ots_margin) &
                 (df_filtered['osnr'] <= osnr_margin)]

# 保存处理后的 CSV 文件
output_path = './example/' + file_name + '.oms_processed.csv'
df_filtered.to_csv(output_path, index=False)

output_path

# 读取 CSV 文件
file_path = './example/' + file_name + '.service.csv'
df = pd.read_csv(file_path)

# 打乱行
shuffled_df = df.sample(frac=1).reset_index(drop=True)  # 不设置 random_state

# 保存处理后的 CSV 文件
output_path = './example/' + file_name + '.service.csv'
shuffled_df.to_csv(output_path, index=False)

output_path

