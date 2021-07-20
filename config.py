# window_size = 10  # 用多长的时间序列作为特征
# batch_size = 32
# lstm_features = 128
# lstm_lr = 0.005
# pure_lstm_epoch = 100
# seq = 5  # 公式(1)中n的长度，也对应了要用几个LSTM串联，因为每个Xt是输入到不同的LSTM中的
# mode = 2  # 1:只用LSTM  2：LSTM+GCN
# # step = 4  # 多步预测
# epoch_num = 200
# node_num = 156
# in_features = window_size
# hid_features = 8
# out_features = 1
# tgcn_lr = 0.005
# weight_decay = 0.0001





window_size = 20  # 用多长的时间序列作为特征
batch_size = 32
lstm_features = 128
lstm_lr = 0.005
pure_lstm_epoch = 100
seq = 5  # 公式(1)中n的长度，也对应了要用几个LSTM串联，因为每个Xt是输入到不同的LSTM中的
mode = 2  # 1:只用LSTM  2：LSTM+GCN
# step = 4  # 多步预测
epoch_num = 200
node_num = 14
in_features = window_size
hid_features = 8
out_features = 1
tgcn_lr = 0.005
weight_decay = 0.0001