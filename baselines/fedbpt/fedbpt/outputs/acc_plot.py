import re
import numpy as np
from matplotlib import pyplot as plt

# 读取日志文件
with open("./outputs/output_fedavgbbt3.log", "r+") as f:
    text = f.read()
# 改进的正则表达式 - 匹配"Global test acc:"后的浮点数
pattern = r"Global test acc:\s*([\d\.]+)"
matches = re.findall(pattern, text)
# test_line = "2025-06-12 08:24:07,239 - SubprocessLauncher - INFO - Global test acc: 0.5264"
# test_match = re.search(pattern, test_line)
# print(f"测试匹配结果: {test_match.group(1) if test_match else '无匹配'}")
# 调试输出：检查匹配结果
print(f"找到匹配项数量: {len(matches)}")
print(f"前5个匹配项: {matches[:5]}")

# 处理匹配结果
data = []
epoch = 1
for num_str in matches:
    try:
        # 直接转换为浮点数
        acc = float(num_str)
        data.append((epoch, acc))
        epoch += 1
    except ValueError as e:
        print(f"无法转换 '{num_str}': {e}")

# 如果没有数据，打印错误并退出
if not data:
    print("错误：未找到有效数据！")
    print("请检查日志文件内容和正则表达式模式")
    exit(1)

# 准备绘图数据
x = [point[0] for point in data]
y = [point[1] for point in data]

# 绘图
plt.plot(x, y, 'b-', label='Global Test Accuracy')
plt.xlabel('Evaluation Round')
plt.ylabel('Accuracy')
plt.title('Global Test Accuracy Over Time')
plt.grid(True)

# 标记最高准确率
max_y = max(y)
max_x = x[y.index(max_y)]
# 计算纵轴范围
min_y = min(y)
y_range = max_y - min_y

# 设置合理的纵轴范围
margin = 0.1  # 添加5%的边距
plt.ylim(max(0, min_y - margin*y_range), min(1.0, max_y + margin*y_range))
plt.plot(max_x, max_y, 'r*', markersize=10, label=f'Max Accuracy: {max_y:.4f}')
plt.annotate(
    f"{max_y:.4f}",
    xy=(max_x, max_y),
    xytext=(0, 10),
    textcoords='offset points',
    ha='center',
    
)

# 添加图例并保存
plt.tight_layout()
plt.savefig("./outputs/acc_fedavgbbt3.png")
print(f"图表已保存为 acc.png，包含 {len(data)} 个数据点")
print(f"最高准确率: {max_y:.4f} 出现在第 {max_x} 轮")