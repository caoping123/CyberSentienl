import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd
from matplotlib.ticker import ScalarFormatter


def draw_roc(data: pd.DataFrame):
    # 假设你有真实标签和预测得分
    y_true = np.array([1] * 52 + [0] * (len(data) - 52))  # 真实类别，需要是二进制
    y_scores_pc = data.iloc[:, 0].values  # 模型PC的预测得分
    y_scores_cos = data.iloc[:, 1].values  # 模型COS的预测得分

    # y_scores_ed = data.iloc[:, 2].values  # 模型ED的预测得分

    def convert_to_probability(scores):
        return 1 / (1 + np.exp(-scores))

    y_scores_pc = convert_to_probability(y_scores_pc)
    y_scores_cos = convert_to_probability(y_scores_cos)

    # 计算ROC曲线的点
    fpr_pc, tpr_pc, _ = roc_curve(y_true, y_scores_pc, pos_label=1)
    fpr_cos, tpr_cos, _ = roc_curve(y_true, y_scores_cos, pos_label=1)
    # fpr_ed, tpr_ed, _ = roc_curve(y_true, y_scores_ed, pos_label=1)

    # 计算AUC
    roc_auc_pc = auc(fpr_pc, tpr_pc)
    roc_auc_cos = auc(fpr_cos, tpr_cos)
    # roc_auc_ed = auc(fpr_ed, tpr_ed)

    # 开始绘图
    plt.figure()
    lw = 2  # 线宽
    plt.plot(fpr_pc, tpr_pc, color='red',
             lw=lw, label='PC (AUC = %0.3f)' % roc_auc_pc)
    plt.plot(fpr_cos, tpr_cos, color='blue',
             lw=lw, label='COS (AUC = %0.3f)' % roc_auc_cos)
    # plt.plot(fpr_ed, tpr_ed, color='green',
    #         lw=lw, label='ED (AUC = %0.3f)' % roc_auc_ed)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率')
    plt.ylabel('真阳性率')
    plt.title('ROC曲线')
    plt.legend(loc="lower right")
    plt.show()


def draw_acc(data):
    pc = data.iloc[:, 0].values
    cos = data.iloc[:, 1].values
    ed = data.iloc[:, 2].values

    correct = 0
    for index, i in enumerate(pc):
        if index < 52:
            if i > 0.9:
                correct += 1
        else:
            if i < 0.9:
                correct += 1
    pc_acc = correct / len(data)

    correct = 0
    for index, i in enumerate(cos):
        if index < 52:
            if i > 0.9:
                correct += 1
        else:
            if i < 0.9:
                correct += 1
    cos_acc = correct / len(data)

    correct = 0
    for index, i in enumerate(ed):
        if index < 52:
            if i < 26:
                correct += 1
        else:
            if i > 26:
                correct += 1
    ed_acc = correct / len(data)

    print(pc_acc, cos_acc, ed_acc)

    # plt.bar(['PC', 'COS', 'ED'], [pc_acc*100, cos_acc*100, ed_acc*100], color='#dae3f3', width=0.4, edgecolor='#2f528f', hatch='/')

    # 添加标签和标题
    # plt.xlabel('Methods', fontsize=16)
    # plt.ylabel('Accuracy%', fontsize=16)
    # # plt.title('Accuracy of Different Methods')
    # plt.ylim(95.0, 100)
    #
    plt.bar(['PC', 'COS', 'ED'], [(1-pc_acc)*100, (1-cos_acc)*100, (1-ed_acc)*100], color='#dae3f3', width=0.4, edgecolor='#2f528f', hatch='/')
    plt.xlabel('Methods', fontsize=16)
    plt.ylabel('EER%', fontsize=16)
    plt.ylim(0, 3)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    #显示柱状图
    plt.savefig('eer.pdf')
    plt.show()


def draw_loss():
    data = pd.read_csv('../embd.csv', header=None)
    loss = data.iloc[:, 0].values
    acc = data.iloc[:, 1].values
    fig, ax1 = plt.subplots()

    # 绘制第一条折线
    ax1.plot(range(len(loss)), loss, 'b-', label='Loss', color=(170/255, 102/255, 103/255))
    ax1.set_xlabel('Epoch', fontsize=16)
    ax1.tick_params('x', color=(0, 0, 0), labelsize=14)
    ax1.set_ylabel('MSE Loss', color=(0,0,0), fontsize=16)
    ax1.tick_params('y', color=(0,0,0),labelsize=14)
    ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # y_formatter = ScalarFormatter(useMathText=True)
    # y_formatter.set_scientific(True)
    # y_formatter.set_useOffset(False)
    # ax1.yaxis.set_major_formatter(y_formatter)
    # 创建第二个 Y 轴，共享 X 轴
    ax2 = ax1.twinx()
    ax2.plot(range(len(loss)), acc, 'r-', label='Accuracy', color=(102/255, 195/255, 142/255))
    ax2.set_ylabel('Accuracy', color=(0,0,0), fontsize=16)
    ax2.tick_params('y', color=(0,0,0),labelsize=14)
    ax2.set_ylim(0, 100)

    # 添加图例
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper left',fontsize=14)

    # 添加标题
    #plt.title('Two Lines with Two Y-axes')

    # 显示图表
    plt.tight_layout()
    plt.savefig('loss.pdf')
    plt.show()




if __name__ == '__main__':
    # 读取CSV文件
    data = pd.read_csv('../seg_dist.csv', header=None)
    #
    # # # 显示前几行数据以进行检查
    # # print(data.head())
    # draw_acc(data)
    draw_loss()