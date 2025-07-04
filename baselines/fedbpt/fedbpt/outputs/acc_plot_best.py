import re
import numpy as np
from matplotlib import pyplot as plt

# 读取日志文件
with open("baselines/fedbpt/fedbpt/outputs/fedcrossbpt_sst2_0_6.log", "r+") as f:
    text = f.read()
# 正则表达式匹配平均准确率和最佳准确率
avg_pattern = r"Average test acc from clients:\s*([\d\.]+)"
best_pattern = r"Best test acc among clients:\s*([\d\.]+)"

avg_matches = re.findall(avg_pattern, text)
best_matches = re.findall(best_pattern, text)
# test_line = "2025-06-12 08:24:07,239 - SubprocessLauncher - INFO - Global test acc: 0.5264"
# test_match = re.search(pattern, test_line)
# print(f"测试匹配结果: {test_match.group(1) if test_match else '无匹配'}")
# 调试输出：检查匹配结果
print(f"平均准确率数量: {len(avg_matches)}，最佳准确率数量: {len(best_matches)}")
print(f"前5个平均准确率: {avg_matches[:5]}")
print(f"前5个最佳准确率: {best_matches[:5]}")

# 转换为浮点数
avg_acc = [float(x) for x in avg_matches]
best_acc = [float(x) for x in best_matches]


# 获取轮数
rounds = list(range(1, max(len(avg_acc), len(best_acc)) + 1))

# 绘图
plt.plot(rounds[:len(avg_acc)], avg_acc, 'b-', label='Average Test Accuracy')
plt.plot(rounds[:len(best_acc)], best_acc, 'g-', label='Best Test Accuracy')

# 标注最高点
max_avg = max(avg_acc)
max_avg_round = rounds[avg_acc.index(max_avg)]

max_best = max(best_acc)
max_best_round = rounds[best_acc.index(max_best)]

plt.plot(max_avg_round, max_avg, 'bo', markersize=8)
plt.annotate(f"{max_avg:.4f}", xy=(max_avg_round, max_avg), xytext=(0, 10), textcoords='offset points', ha='center', color='blue')

plt.plot(max_best_round, max_best, 'go', markersize=8)
plt.annotate(f"{max_best:.4f}", xy=(max_best_round, max_best), xytext=(0, -15), textcoords='offset points', ha='center', color='green')

# 添加图例和坐标信息
plt.xlabel('Evaluation Round')
plt.ylabel('Accuracy')
plt.title('Client Test Accuracy Over Time')
plt.grid(True)
plt.legend()
plt.tight_layout()

# 保存图像
output_path = "baselines/fedbpt/fedbpt/outputs/png/fedcrossbpt_sst2_0_6.png"
plt.savefig(output_path)
print(f"图表已保存为 {output_path}")
print(f"最高平均准确率: {max_avg:.4f} 出现在第 {max_avg_round} 轮")
print(f"最高最佳准确率: {max_best:.4f} 出现在第 {max_best_round} 轮")