import xlrd
# import xlwt
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd


def read_data():
#    workBook = xlrd.open_workbook('./data/road_data.xls')
#    sheet_content = workBook.sheet_by_index(0)
#    speed_data = np.zeros([4455, 14], dtype=np.float32)
#    time_label = np.empty([4455, 14], dtype=np.string_)
#    for i in range(14):
#        temp_data = np.array(sheet_content.col_values(i*3+2)[1:4456])
#        speed_data[:, i] = temp_data / np.max(temp_data)  # 归一化
#        time_label[:, i] = np.array(sheet_content.col_values(i*3)[1:4456], dtype=np.string_)
    dateparse = lambda x: pd.datetime.strptime(x, '%Y %m %d %H %M %S')
    dataset = pd.read_excel(r'./data/road_data.xls', date_parser=dateparse).values
    speed = dataset[0:4455, [i % 3 == 2 for i in range(42)]].astype(np.float32)
    max_thres = np.max(speed)
    speed = speed / max_thres
    time = dataset[:, 0]

    return speed, max_thres, time


class LDataset(Dataset):

    def __init__(self, data, window_size, seq):
        super(LDataset, self).__init__()
        self.data = data
        self.window_size = window_size
        self.num = self.data.size(0) - window_size - seq
        self.seq = seq

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        """

        :param item: 每次采样的起点
        :return: 前一项是用来预测的过去时间段，后一项是t+1时刻真实的数据(ground_truth)
        """
        return self.data[item: item + self.window_size + self.seq], self.data[item + self.window_size + self.seq]

def get_adj():
    A = np.array([[0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                  [1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                  [1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                  [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0],
                  [1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                  [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                  [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                  ])
    return A.astype(np.float32)


def normalize_graph(A, batch_size, seq):
    """
    Parameters
    ----------
    A : FloatTensor: the original graph data (node_num, node_num).
    batch_size:
    seq:
    Returns
    -------
    adjacency : FloatTensor: the normalized adjacency matrix (seq, node_num, node_num).

    """
    node_num = A.size()[0]
    eye = torch.eye(node_num, dtype=torch.float32).cuda()
    # A~ = A + In
    A += eye
    diag = A.sum(dim=-1, keepdim=True).pow(-0.5) * eye
    adjacency = diag.matmul(A).matmul(diag)
    adjacency = adjacency.unsqueeze(0).expand(seq, node_num, node_num)
    return adjacency


def eval_rmse(predicted, gnd):
    # predicted, gnd = np.array(predicted), np.array(gnd)
    if len(gnd.shape) == 1:
        m = gnd.shape
        n = 1
    else:
        m, n = gnd.shape
    # rmse = ((predicted - gnd).pow(2).sum() / (m * n)).pow(0.5)
    rmse = np.power(np.sum(np.power((predicted-gnd), 2)) / (m * n), 0.5)
    return rmse

def eval_mae(predicted, gnd):
    if len(gnd.shape) == 1:
        m = gnd.shape
        n = 1
    else:
        m, n = gnd.shape
    mae = np.sum(np.fabs(predicted - gnd)) / (m * n)
    return mae

def read_another_data():
    adj = pd.read_csv(r'./data/sz_adj.csv', header=None).values
    adj = adj.astype(np.float32)
    speed = pd.read_csv(r'./data/sz_speed.csv').values
    speed = speed.astype(np.float32)
    max_thres = np.max(speed)
    speed /= max_thres
    return speed, max_thres, adj
