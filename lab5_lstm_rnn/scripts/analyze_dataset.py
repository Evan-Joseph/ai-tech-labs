#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析潍坊市开放目录，筛选符合条件的高密市时间序列数据集
"""

import pandas as pd
import numpy as np

# 读取Excel文件
df = pd.read_excel('/Users/joseph/Desktop/【人工智能技术】课程实验/实验区/lab5_lstm_rnn/data/raw/潍坊市开放目录.xlsx')

# 条件1: 无条件获取
cond_open = df['开放类型'] == '无条件开放'

# 条件4: 高密市数据
cond_gaomi = df['所属行政区域'].fillna('').str.contains('高密', case=False)

# 初步筛选
df_filtered = df[cond_open & cond_gaomi].copy()

print('='*100)
print('【符合条件的时间序列数据集分析】')
print('条件：高密市 + 无条件开放 + 数据规模适合 + 时间序列形态')
print('='*100)

# 定义时间序列关键词（在资源名称、资源描述、信息项中搜索）
time_series_keywords = [
    '年度', '月度', '季度', '年份', '月份', '日期', '时间',
    '统计', '历年', '逐月', '逐年', '动态', '变化', '趋势',
    '监测', '气象', '天气', '温度', '降水', '湿度',
    '价格', '销量', '产量', '营收', '收入', '支出',
    '人数', '数量', '总量', '流量', '用量', '能耗',
    '水质', '空气', 'PM', '污染', '排放',
    '用电', '用水', '用气', '供暖',
    '交通', '客流', '车流', '货运',
    '降雨', '雨量', '水位', '流速',
    '生产', '经济', 'GDP', '指数'
]

# 检查是否包含时间序列相关关键词
def check_time_series(row):
    text = str(row['资源名称']) + ' ' + str(row['资源描述']) + ' ' + str(row['信息项'])
    for kw in time_series_keywords:
        if kw.lower() in text.lower():
            return True
    return False

# 解析数据容量（转为数字，用于排序）
def parse_capacity(val):
    try:
        return int(val)
    except:
        return 0

df_filtered['is_time_series'] = df_filtered.apply(check_time_series, axis=1)
df_filtered['capacity_num'] = df_filtered['数据容量'].apply(parse_capacity)

# 条件3: 时间序列数据形态
df_ts = df_filtered[df_filtered['is_time_series']].copy()

# 条件2: 数据规模（至少有一定数据量）
# ARIMA: 最少50-100个样本点
# LSTM/Transformer: 至少几百到几千样本点
# 我们取数据容量 >= 100 作为基础门槛
df_ts_large = df_ts[df_ts['capacity_num'] >= 100].sort_values('capacity_num', ascending=False)

print(f'\n初步筛选结果:')
print(f'  高密市无条件开放数据集总数: {len(df_filtered)}')
print(f'  包含时间序列特征的数据集: {len(df_ts)}')
print(f'  数据容量>=100（适合模型训练）: {len(df_ts_large)}')

# 进一步分析：按数据规模分类
print('\n' + '='*100)
print('【按数据规模分类】')
print('='*100)

# 分类：适合不同模型的数据规模
categories = {
    '大规模 (>10000, 适合LSTM/Transformer深度学习)': df_ts_large[df_ts_large['capacity_num'] > 10000],
    '中规模 (1000-10000, 适合各类模型)': df_ts_large[(df_ts_large['capacity_num'] >= 1000) & (df_ts_large['capacity_num'] <= 10000)],
    '小规模 (100-1000, 可用于ARIMA/简单LSTM)': df_ts_large[(df_ts_large['capacity_num'] >= 100) & (df_ts_large['capacity_num'] < 1000)]
}

for cat_name, cat_df in categories.items():
    print(f'\n{cat_name}: {len(cat_df)} 个')
    if len(cat_df) > 0:
        for idx, row in cat_df.head(10).iterrows():
            print(f"\n  【{row['资源名称']}】")
            print(f"     数据容量: {row['数据容量']} | 部门: {row['部门名称']}")
            print(f"     数据领域: {row['数据领域']} | 更新频率: {row['更新频率']}")
            print(f"     信息项: {row['信息项']}")

# 重点推荐
print('\n' + '='*100)
print('【★ 重点推荐的时间序列数据集 ★】')
print('='*100)

# 挑选最适合的数据集（数据量大 + 明显时间序列特征）
best_keywords = ['年度', '月度', '统计', '监测', '历年', '变化', '趋势', '产量', '价格', '用量', '气象', '天气', '污染', '交通']

def score_dataset(row):
    score = 0
    text = str(row['资源名称']) + ' ' + str(row['资源描述']) + ' ' + str(row['信息项'])
    for kw in best_keywords:
        if kw in text:
            score += 2
    # 数据量加分
    if row['capacity_num'] > 10000:
        score += 5
    elif row['capacity_num'] > 5000:
        score += 3
    elif row['capacity_num'] > 1000:
        score += 2
    # 更新频率加分（定期更新说明有时间维度）
    freq = str(row['更新频率'])
    if '年' in freq or '月' in freq or '季' in freq or '周' in freq or '日' in freq:
        score += 3
    return score

df_ts_large['score'] = df_ts_large.apply(score_dataset, axis=1)
df_top = df_ts_large.sort_values('score', ascending=False).head(10)

for rank, (idx, row) in enumerate(df_top.iterrows(), 1):
    print(f"\n{rank}. 【{row['资源名称']}】 (推荐度评分: {row['score']})")
    print(f"   - 数据容量: {row['数据容量']} 条")
    print(f"   - 数据规模分类: {row['数据范围']}")
    print(f"   - 部门名称: {row['部门名称']}")
    print(f"   - 数据领域: {row['数据领域']} | 行业: {row['数据行业']}")
    print(f"   - 更新频率: {row['更新频率']}")
    print(f"   - 信息项: {row['信息项']}")
    desc = str(row['资源描述'])[:200] if pd.notna(row['资源描述']) else 'N/A'
    print(f"   - 描述: {desc}")
    print(f"   - 发布日期: {row['发布日期']}")
    print(f"   - 最后更新: {row['数据更新日期']}")

# 保存筛选结果
output_file = '/Users/joseph/Desktop/【人工智能技术】课程实验/实验区/lab5_lstm_rnn/docs/高密市时间序列数据集筛选结果.csv'
import os
os.makedirs(os.path.dirname(output_file), exist_ok=True)
df_ts_large[['资源名称', '部门名称', '数据领域', '数据行业', '数据范围', '数据容量', '更新频率', '信息项', '资源描述', 'score']].to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n\n筛选结果已保存至: {output_file}")
