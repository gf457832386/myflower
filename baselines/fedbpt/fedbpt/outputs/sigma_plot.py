f1 = r"/home/hello/yangmuyuan/FL/myflower/baselines/fedbpt/fedbpt/output.log"
f2 = r"/home/hello/yangmuyuan/FL/NVFlare-main/research/fed-bpt/tmp/nvflare/fedbpt/site-1/log.txt"
import re
import numpy as np
from matplotlib import pyplot as plt

# 读取日志文件
def get_sigmas(fp):
    with open(fp, "r") as f:
        text = f.read()

    # # 调试：打印日志前几行
    # print("日志前5行:")
    # for i, line in enumerate(text.split('\n')[:5], 1):
    #     print(f"{i}. {line}")

    # 正则表达式模式 - 匹配current_round和global_es.sigma
    pattern = r"Running current_round=(\d+).*?global_es.sigma=([\d\.]+).*?global_es.mean: len=(\d+), mean=([\d\.-]+), std=([\d\.]+)"
    matches = re.findall(pattern, text, re.DOTALL)

    # # 调试输出
    # print(f"\n找到匹配项数量: {len(matches)}")
    # if matches:
    #     print("前5个匹配项:")
    #     for i, match in enumerate(matches[:5], 1):
    #         print(f"{i}. Round: {match[0]}, Sigma: {match[1]}, Len: {match[2]}, Mean: {match[3]}, Std: {match[4]}")

    # 处理匹配结果
    rounds = []
    sigmas = []
    mean_lens = []
    mean_means = []
    mean_stds = []

    for match in matches:
        try:
            round_num = int(match[0])
            sigma = float(match[1])
            mean_len = int(match[2])
            mean_mean = float(match[3])
            mean_std = float(match[4])
            if round_num not in rounds:
                rounds.append(round_num)
                sigmas.append(sigma)
                mean_lens.append(mean_len)
                mean_means.append(mean_mean)
                mean_stds.append(mean_std)
        except ValueError as e:
            print(f"转换错误: {e} - 匹配数据: {match}")

    # 检查是否有有效数据
    if not rounds:
        print("错误：未找到有效数据！")
        exit(1)
    return rounds,sigmas

x1,y1 = get_sigmas(f1)
x2,y2 = get_sigmas(f2)
x2=x2[:100]
y2=y2[:100]
# 创建图表展示结果
plt.figure(figsize=(15, 10))

# 1. Sigma值随时间变化
# plt.subplot(2, 2, 1)
plt.plot(x1, y1, 'b-',label="flower")
plt.plot(x2, y2, 'r-',label="nvflare")
plt.xlabel('Round')
plt.ylabel('Sigma')
plt.title('Global ES Sigma Over Rounds')
plt.grid(True)
plt.legend()
# # 2. Mean平均值随时间变化
# plt.subplot(2, 2, 2)
# plt.plot(rounds, mean_means, 'g-')
# plt.xlabel('Round')
# plt.ylabel('Mean')
# plt.title('Global ES Mean Value Over Rounds')
# plt.grid(True)

# # 3. Mean标准差随时间变化
# plt.subplot(2, 2, 3)
# plt.plot(rounds, mean_stds, 'r-')
# plt.xlabel('Round')
# plt.ylabel('Std Dev')
# plt.title('Global ES Standard Deviation Over Rounds')
# plt.grid(True)

# # 4. Mean长度随时间变化
# plt.subplot(2, 2, 4)
# plt.plot(rounds, mean_lens, 'm-')
# plt.xlabel('Round')
# plt.ylabel('Length')
# plt.title('Global ES Mean Length Over Rounds')
# plt.grid(True)

# 保存图表
plt.tight_layout()
plt.savefig("./es_params.png")
print("\n图表已保存为 es_params.png")

# # 创建数据表格摘要
# print("\n参数汇总:")
# print(f"{'Round':>6} {'Sigma':>10} {'Mean Length':>12} {'Mean Value':>12} {'Std Dev':>12}")
# for i in range(min(10, len(rounds))):
#     r = rounds[i]
#     s = sigmas[i]
#     ml = mean_lens[i]
#     mm = mean_means[i]
#     sd = mean_stds[i]
#     print(f"{r:6d} {s:10.6f} {ml:12d} {mm:12.6f} {sd:12.6f}")

# if len(rounds) > 10:
#     print(f"... 共 {len(rounds)} 轮数据")