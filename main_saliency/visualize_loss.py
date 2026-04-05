import os
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg') 

def read_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def parse_log(lines):
    data = [line.strip().split('\t') for line in lines[1:]]  # 跳过第一行（列名）
    data = [[float(value) for value in row] for row in data]
    return list(zip(*data))  # 转置列表，以便于提取每列



def loss_visualize(file_path_train, file_path_val, output_dir, model_name):
    train_lines = read_file(file_path_train)
    val_lines = read_file(file_path_val)

    # 解析文件
    train_data = parse_log(train_lines)
    val_data = parse_log(val_lines)

    # 提取每列数据
    train_epoch, train_loss, train_lr = train_data[0], train_data[2], train_data[-1]
    val_epoch, val_loss= val_data[0], val_data[1]

    # 寻找最小损失值及其对应的epoch
    min_train_loss, min_train_epoch = min(zip(train_loss, train_epoch))
    min_val_loss, min_val_epoch = min(zip(val_loss, val_epoch))

    # 绘制曲线图
    plt.figure(figsize=(10, 6))
    plt.plot(train_epoch, train_loss, label='Training Loss', marker='o', color='blue')
    plt.plot(val_epoch, val_loss, label='Validation Loss', marker='x', color='red')

    # 标注最小损失点
    plt.annotate(f'Min Train Loss: {min_train_loss:.4f}', 
                (min_train_epoch, min_train_loss),
                textcoords="offset points", 
                xytext=(-10,-10),
                ha='center',
                arrowprops=dict(arrowstyle="->", color='blue'))

    plt.annotate(f'Min Val Loss: {min_val_loss:.4f}', 
                (min_val_epoch, min_val_loss),
                textcoords="offset points", 
                xytext=(-10,-15),
                ha='center',
                arrowprops=dict(arrowstyle="->", color='red'))
    
    for epoch, lr, loss in zip(train_epoch, train_lr, train_loss):
        plt.annotate(f'LR: {lr:.4f}', 
                    (epoch, loss),
                    textcoords="offset points", 
                    xytext=(0,10),
                    ha='center',
                    fontsize=8,
                    rotation=45)

    # 设置图例和标签
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(model_name)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir,'loss_{}.png'.format(model_name)))

if __name__ == '__main__':

    
    output_dir = '/DATA/DATA4/zyx/VRSaliency/mynet/MyImageBind-main/outputs_weights/20240130_trad_train_temporal3layer05'
    model_name = os.path.basename(output_dir)
    file_path_train = os.path.join(output_dir, "train_epoch.log")
    file_path_val = os.path.join(output_dir, 'val_{}.log'.format(model_name))
    
    loss_visualize(file_path_train, file_path_val, output_dir, model_name)