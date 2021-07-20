import torch
from torch import nn
from torch.nn import init


class Pure_LSTM(nn.Module):

    def __init__(self, window_size, lstm_features):
        super(Pure_LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=window_size,
            hidden_size=lstm_features,
            num_layers=1,
            batch_first=True  # If True, then the input and output tensors are provided as (batch, seq, feature). Default: False
        )
        self.ffn = nn.Sequential(
            nn.Linear(lstm_features, 1),
            nn.Sigmoid()
        )

    def forward(self, input_data, tt):
        """
        :param input_data: 输入时间序列（batch_size, window_size, node_num）
        :return: out_put : 输出预测值 (batch_size, 1)
        """
        lstm_input = input_data[:, :, tt]  # 只预测tt号线路
        # LSTM input is (batch, seq, features) 这里只用一个LSTM，seq=1
        lstm_input = lstm_input.unsqueeze(1)  # from (batch_size, window_size) to (batch_size, 1, window_size)
        _, (hn, _) = self.lstm(lstm_input)
        output = self.ffn(hn.squeeze(0))
        return output


class GCN(nn.Module):

    def __init__(self, seq, in_features, hid_features, out_features):
        super(GCN, self).__init__()
        self.weights1 = nn.Parameter(
            torch.Tensor(seq, in_features, out_features)
        )
        # self.weights2 = nn.Parameter(
        #     torch.Tensor(seq, hid_features, out_features)
        # )
        self._reset_parameters()

    def _reset_parameters(self):
        init.xavier_uniform_(self.weights1)
        # init.xavier_uniform_(self.weights2)

    def forward(self, A, X):
        """

        :param A: 邻接矩阵 (seq, node_num, node_num)
        :param X: 特征矩阵 (batch_size, seq, node_num, window_size)
        :return: output (batch_size, seq, node_num, out_features)
        """
        # window_size, in_features, hid_features = self.weights1.size()
        # batch_size = A.size()[0]
        # A * X * W (batch_size, seq, node_num, node_num) * (batch_size, seq, node_num, in_features) * (batch_size,
        # seq, in_features, out_features)
        # output = A.matmul(torch.sigmoid(A.matmul(X).matmul(self.weights1))).matmul(self.weights1).sigmoid()
        output = A.matmul(X).matmul(self.weights1).sigmoid()
        return output


class TGCN(nn.Module):

    def __init__(self, window_size, node_num, in_features, hid_features, out_features, lstm_features, seq):
        super(TGCN, self).__init__()
        self.window_size = window_size
        self.node_num = node_num
        self.in_features = in_features
        self.hid_features = hid_features
        self.out_features = out_features
        self.lstm_features = lstm_features
        self.seq = seq
        self.gcn = GCN(seq, in_features, hid_features, out_features)
        self.lstm = nn.LSTM(
            input_size=out_features*node_num,
            hidden_size=lstm_features,
            num_layers=1,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(lstm_features, node_num),
            nn.Sigmoid()
        )

    def forward(self, A, X):
        """
        :param A: 邻接矩阵 (seq, node_num, node_num) 这里还是利用广播机制，所以A没有batch_size，这么做是因为每个batch大小不一样
        :param X: 特征矩阵 (batch_size, seq, node_num, window_size)
        :return: output:输出预测值 (batch_size, node_num)
        """
        batch_size = X.size()[0]
        gcn_output = self.gcn(A, X)
        gcn_output = gcn_output.view(batch_size, self.seq, -1)  # -1 means it is inferred from other dimensions
        # LSTM input should be (batch, seq, feature)
        _, (hn, _) = self.lstm(gcn_output)
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        hn = hn.squeeze(0)  # then hn becomes (batch, hidden_size)
        output = self.ffn(hn)
        return output
