import matplotlib.pyplot as plt


def draw(data1, data2, data3, label, filepath):
    # 指定颜色
    # color1 = '#216bb4'
    # color2 = '#629797'
    # color3 = '#6c4098'
    color1 = '#b5c6e7'
    color2 = '#f7cbac'
    color3 = '#fee599'

    # 绘制柱状图
    plt.bar(range(1, len(data1) + 1), data1, color=color1, label=label[0], width=0.5)
    plt.bar(range(len(data1) + 1, len(data1) + len(data2) + 1), data2, color=color2, label=label[1], width=0.5)
    plt.bar(range(len(data1) + len(data2) + 1, len(data1) + len(data2) + len(data3) + 1), data3, color=color3, label=label[2], width=0.5)

    # 添加标签和标题
    plt.xlabel('Sample')
    plt.ylabel('Cosine similarity')
    # plt.title('三个列表的柱状图')

    # 添加图例
    plt.legend()
    plt.ylim(0.4, 1)
    plt.xticks(range(1, len(data1) + len(data2) + len(data3) + 1))

    # 保存图形到本地文件
    plt.savefig(filepath)

    # 显示图形
    plt.show()


def line_chat(data1, data2, data3, label, filepath):
    color1 = 'black'
    color2 = 'red'
    color3 = '#23395d'

    plt.plot(data1, label=label[0], color=color1, marker='o')
    plt.plot(data2, label=label[1], color=color2, marker='s')
    plt.plot(data3, label=label[2], color=color3, marker='P')

    # for i, value in enumerate(data1):
    #     plt.text(i, value, str(value), ha='center', va='bottom', color='blue')
    # for i, value in enumerate(data2):
    #     plt.text(i, value, str(value), ha='center', va='bottom', color='green')
    # for i, value in enumerate(data3):
    #     plt.text(i, value, str(value), ha='center', va='bottom', color='red')

    # 添加图例
    plt.legend()
    plt.ylim(0.4, 1)
    plt.xticks(range(len(data1)), range(1, len(data1) + 1))

    # 添加标题和坐标轴标签
    # plt.title('Three Lists Line Chart')
    plt.xlabel('Sample')
    plt.ylabel('Cosine similarity')

    plt.savefig(filepath)

    # 显示图形
    plt.show()


def box_plots(data1, data2, data3, labels, filepath, xlabel=None):
    colors = ['#dae3f3', '#fbe5d6', '#e2f0d9']
    data = [data1, data2, data3]

    # 创建箱型图，并设置不同的颜色
    boxprops = dict(facecolor='white', color='black', linewidth=2)

    # 创建图形
    fig, ax = plt.subplots(figsize=(5.5, 4.8))

    # 绘制箱型图
    bp = ax.boxplot(data, labels=labels, patch_artist=True, boxprops=boxprops,widths=0.3)



    # 为每个箱体设置不同的颜色
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    plt.ylabel('Cosine similarity', fontsize=16)

    if xlabel:
        plt.xlabel(xlabel, fontsize=16)
    plt.tick_params(axis='y', labelsize=16)
    ax.set_xticklabels(labels, fontsize=16)
    if labels[0] == '25cm':
        plt.ylim(0.4, 1)
    else:
        plt.ylim(0.5,1)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.show()
