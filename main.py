import numpy as np
import matplotlib.pyplot as plt

# 设置序列长度 N
N = 26

# 创建时间点索引 n
n = np.arange(N)

# 绘制几个不同的 k 值对应的基函数
plt.figure(figsize=(12, 8))

bar_value = [0.00155656, 0.00079951, 0.0004533, 0.00024176, 0.00022244, 0.00014968,
             0.00018983, 0.00021409, 0.00031831, 0.00022561, 0.00032443, 0.00029503,
             0.00021994, 0.00017342, 0.00015452, 0.00028628, 0.00043679, 0.00025806,
             0.00021546, 0.00026356, 0.00027223, 0.00037592, 0.00059175, 0.000617,
             0.00062559, 0.00048798]
fig, ax1 = plt.subplots()
ax1.bar(range(len(bar_value)), bar_value)

ax2 = ax1.twinx()
# 绘制 k = 0 的基函数
k = 0
phi_k = np.cos((np.pi / N) * (n + 0.5) * k)
ax2.plot(n, phi_k, label=f'k={k}', marker='p')

# 绘制 k = 1 的基函数
k = 1
phi_k = np.cos((np.pi / N) * (n + 0.5) * k)
print(phi_k)
ax2.plot(n, phi_k, label=f'k={k}', marker='p')

# 绘制 k = 2 的基函数
k = 2
phi_k = np.cos((np.pi / N) * (n + 0.5) * k)
print(phi_k)
ax2.plot(n, phi_k, label=f'k={k}', marker='p')

# 绘制 k = 3 的基函数
k = 3
phi_k = np.cos((np.pi / N) * (n + 0.5) * k)
print(phi_k)
ax2.plot(n, phi_k, label=f'k={k}', marker='p')

k = 4
phi_k = np.cos((np.pi / N) * (n + 0.5) * k)
print(phi_k)
ax2.plot(n, phi_k, label=f'k={k}', marker='p')

# 添加图例和标签

# plt.title('DCT Basis Functions')
plt.xlabel('n')
#plt.ylabel('cos((π/N)*(n + 0.5)*k)')
ax2.legend()
plt.grid(True)
plt.show()
