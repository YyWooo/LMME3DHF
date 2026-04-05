import csv

def print_trainable_parameters_with_names(model):
    total_params = 0
    trainable_params = 0

    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter name: {name} | Requires gradient: {param.requires_grad}")
            trainable_params += param.numel()  # 统计可训练参数的数量
        total_params += param.numel()  # 统计总参数的数量

    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {total_params - trainable_params}")
    
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()


def load_value_file(file_path):
    with open(file_path, 'r') as input_file:
        value = float(input_file.read().rstrip('\n\r'))

    return value


