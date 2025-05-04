import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

file_path = 'new_example/example4.oms_processed.xlsx'
df = pd.read_excel(file_path)

distance = list(df['distance'].values)
ots = df['ots'].values
osnr = df['osnr'].values

x = [0,50,100,150,200,250,300,350,400]
y_max = []
y_min = []
for i in range(len(x)-1):
    y_max.append(max([item for item in distance if x[i] < item <= x[i+1]]))
    y_min.append(min([item for item in distance if x[i] < item <= x[i+1]]))
osnr_max = [distance.index(item) for item in y_max]
osnr_min = [distance.index(item) for item in y_min]
osnr_max = [osnr[item] for item in osnr_max]
osnr_min = [osnr[item] for item in osnr_min]

x = np.array([50,100,150,200,250,300,350,400])

# 定义拟合函数
def func(x, a, b):
    return a * x+b

# curve_fit 返回拟合参数和协方差矩阵
# popt, pcov = curve_fit(func, x, osnr_min)

# 绘制数据和拟合曲线
plt.plot(distance, osnr, 'bo', label='data')
# plt.plot(x, func(x, *popt), 'r-', label='curve')
plt.xlabel('distance')
plt.ylabel('osnr')
plt.legend()
plt.show()

# print("拟合参数：", popt)
