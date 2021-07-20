import torch
import config
import os
import numpy as np
import utils
from torch import optim, nn
from torch.utils.data.dataloader import DataLoader
from model import Pure_LSTM, TGCN
import matplotlib as mpl
import matplotlib.pyplot as plt

# Preprocess

# Load data 加载数据
speed_data, max_thres, time_label = utils.read_data()
# speed_data, max_thres, adj = utils.read_another_data()  # here
speed_data = torch.from_numpy(speed_data)

# 构建邻接矩阵A (batch_size, seq, node_num, node_num)
A = utils.get_adj()
# A = adj  # here
A = torch.from_numpy(A).cuda()
A_hat = utils.normalize_graph(A, config.batch_size, config.seq).cuda()

# Preprocess 数据预处理，按8：2划分训练集和测试集
num = speed_data.size()[0]  # 总数据集长度
total_num = num - config.window_size  # 开头那部分数据用来训练，不预测
train_num = int(0.8 * total_num)
test_num = total_num - train_num

train_speed = speed_data[0:train_num]
test_speed = speed_data[train_num:]

# 用DataLoader加载数据
train_data = utils.LDataset(train_speed, config.window_size, config.seq)
test_data = utils.LDataset(test_speed, config.window_size, config.seq)
train_loader = DataLoader(
    dataset=train_data,
    batch_size=config.batch_size,
    shuffle=True,
    pin_memory=True
)
test_loader = DataLoader(
    dataset=test_data,
    batch_size=config.batch_size,
    shuffle=False,
    pin_memory=True
)

mse = nn.MSELoss(reduction='sum')

if config.mode == 1:
    # 只用LSTM预测道路
    for tt in range(config.node_num):
        pure_lstm = Pure_LSTM(
            window_size=config.window_size,
            lstm_features=config.lstm_features
        )

        pure_lstm = pure_lstm.cuda()
        pure_lstm_optimizer = optim.Adam(pure_lstm.parameters(), lr=config.lstm_lr)

        print('Train pure LSTM')
        for epoch in range(config.pure_lstm_epoch):
            for i, data in enumerate(train_loader):
                pure_lstm_optimizer.zero_grad()
                input_data, gnd = data
                input_data, gnd = input_data.cuda(), gnd[:, tt:tt+1].cuda()
                predicted = pure_lstm(input_data, tt)
                loss = mse(predicted, gnd)
                loss.backward()
                pure_lstm_optimizer.step()
                print('[epoch %d] [step %d] [loss %.4f]' % (epoch, i, loss.item()))


        print('Test pure LSTM')

        for i, data in enumerate(test_loader):
            input_data, gnd = data
            input_data, gnd = input_data.cuda(), gnd[:, tt:tt+1].cuda()
            predicted = pure_lstm(input_data, tt)

            predicted = predicted.detach().cpu().numpy() * max_thres
            gnd = gnd.detach().cpu().numpy() * max_thres
            if i==0:
                pred_list = predicted
                gnd_list = gnd
            else:
                pred_list = np.vstack((pred_list, predicted))
                gnd_list = np.vstack((gnd_list, gnd))

        # 画图
        import seaborn as sns
        plt.rcParams['font.sans-serif'] = ['SimHei'] # 黑体
        plt.style.use(['ggplot', 'seaborn-paper'])
        # x = np.arange(test_num)
        x = time_label[-test_num:]
        pred_list = pred_list
        gnd_list = gnd_list
        plt.plot(x, pred_list, 'bo--', label='Predicted')
        plt.plot(x, gnd_list, 'r-', label='Observed')
        plt.legend()
        plt.title('Prediction results')
        plt.xlabel('Time')
        plt.ylabel('Speed')
        # plt.xticks(rotation=45)
        rmse = utils.eval_rmse(pred_list, gnd_list)
        print("RMSE : %f" % rmse)
        mae = utils.eval_mae(pred_list, gnd_list)
        print("MAE : %f" % mae)
        plt.show()

elif config.mode == 2:
    tgcn = TGCN(
        window_size=config.window_size,
        node_num=config.node_num,
        in_features=config.in_features,
        hid_features=config.hid_features,
        out_features=config.out_features,
        lstm_features=config.lstm_features,
        seq=config.seq
    )

    tgcn = tgcn.cuda()
    tgcn_optimizer = optim.Adam(tgcn.parameters(), lr=config.tgcn_lr, weight_decay=config.weight_decay)


    print('Start Training')
    for epoch in range(config.epoch_num):
        for i, data in enumerate(train_loader):
            input_data, gnd = data
            # input_data: (batch_size, window_size, node_num）
            input_data, gnd = input_data.cuda(), gnd.cuda()
            # 构建特征矩阵X (batch_size, seq, node_num, window_size)
            batch_size, _, node_num = input_data.size()
            window_size = config.window_size
            X = torch.zeros(size=[batch_size, config.seq, node_num, window_size]).cuda()
            for t in range(config.seq):
                X[:, t, :, :] = input_data[:, t:t+window_size, :].transpose(1, 2)

            tgcn_optimizer.zero_grad()
            predicted = tgcn(A_hat, X)
            loss = mse(predicted, gnd)
            loss0 = mse(predicted[:, 0], gnd[:, 0])
            loss.backward()
            tgcn_optimizer.step()
            print('[epoch %d] [step %d] [loss %.4f]' % (epoch, i, loss.item()))
            print(loss0.item())

    print('Start Prediction')
    for i, data in enumerate(test_loader):
        input_data, gnd = data
        input_data, gnd = input_data.cuda(), gnd.cuda()

        # 构建特征矩阵X (batch_size, seq, node_num, window_size) 
        batch_size, _, node_num = input_data.size()
        window_size = config.window_size
        X = torch.zeros(size=[batch_size, config.seq, node_num, window_size]).cuda()
        for t in range(config.seq):
            X[:, t, :, :] = input_data[:, t:t + window_size, :].transpose(1, 2)

        predicted = tgcn(A_hat, X)

        predicted = predicted.detach().cpu().numpy() * max_thres
        gnd = gnd.detach().cpu().numpy() * max_thres
        if i == 0:
            pred_list = predicted
            gnd_list = gnd
        else:
            pred_list = np.vstack((pred_list, predicted))
            gnd_list = np.vstack((gnd_list, gnd))

        # 画图
    import seaborn as sns

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
    plt.style.use(['ggplot', 'seaborn-paper'])
    # x = np.arange(test_num)
    for i in range(config.node_num):
        x = time_label[-test_num+config.seq:]  # 这里LSTM也应该减掉，为了公平对比
        # x = np.arange(test_num-config.seq)
        pred_list_tmp = pred_list[:, i]
        gnd_list_tmp = gnd_list[:, i]
        plt.plot(x, pred_list_tmp, 'bo--', label='Predicted')
        plt.plot(x, gnd_list_tmp, 'r-', label='Observed')
        plt.legend()
        plt.title('Prediction results')
        plt.xlabel('Time')
        plt.ylabel('Speed')
        # plt.xticks(rotation=45)

        # Evaluate
        rmse = utils.eval_rmse(pred_list_tmp, gnd_list_tmp)
        print("RMSE : %f" % rmse)
        mae = utils.eval_mae(pred_list_tmp, gnd_list_tmp)
        print("MAE : %f" % mae)
        plt.show()



