#coding:utf-8
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tushare as ts
import torch
from torch import nn
import initial_data
DAYS_FOR_TRAIN = 5
EPOCHS = 1000


class LSTM_Regression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):#2
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, _x):
        x, _ = self.lstm(_x) #_x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.fc(x)
        x = x.view(s, b, -1)
        return x


def init_data():

    # 取上证指数的收盘价
    '''
    share_prices = ts.get_k_data('000001', start='2018-01-01', index=True)[
        'close'].values
    share_prices = share_prices.astype('float32')  # 转换数据类型: obj ->float
    print(share_prices.shape)
    # 上证指数收盘价作图
    '''
    share_prices=initial_data.input_()
    share_prices = share_prices.astype('float32')
    plt.plot(share_prices)
    plt.savefig('share_prices.png', format='png', dpi=200)
    plt.close()
    return share_prices

def create_dataset(data, days_for_train=5):#根据时间序列创造输入输出
    dataset_x, dataset_y = [], []
    for i in range(len(data) - days_for_train):
        _x = data[i:(i + days_for_train)]
        dataset_x.append(_x)
        dataset_y.append(data[i + days_for_train])
    dataset_x=np.array(dataset_x)
    dataset_y=np.array(dataset_y)
    train_size = int(len(dataset_x)*0.7)
    train_x = dataset_x[:train_size]
    train_y = dataset_y[:train_size]
    test_x = dataset_x[train_size:]
    test_y = dataset_y[train_size:]
    return dataset_x,dataset_y,train_x,train_y,test_x,test_y

def normalize(data):#输出二维矩阵[n,1]
    scaler = MinMaxScaler(feature_range=(-1, 1))  # train data normalized
    share_prices = scaler.fit_transform(data.reshape(-1, 1))
    return share_prices,scaler

def de_normalize(data):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.inverse_transform(data.reshape(-1, 1))
    return data


if __name__ == '__main__':
    share_prices=init_data()
    share_prices,scaler=normalize(share_prices) #output是np.array
    dataset_x, dataset_y,train_x,train_y,test_x,test_y= create_dataset(share_prices, DAYS_FOR_TRAIN)
    # 改变数据集形状，RNN 读入的数据维度是 (seq_size, batch_size, feature_size)
    train_x = train_x.reshape(-1, 1, DAYS_FOR_TRAIN)
    train_x = torch.from_numpy(train_x)
    train_y = train_y.reshape(-1, 1, 1)
    train_y = torch.from_numpy(train_y)
    dataset_x = dataset_x.reshape(-1, 1, DAYS_FOR_TRAIN)  # (seq_size, batch_size, feature_size)
    dataset_x = torch.from_numpy(dataset_x)



    # train model
    model = LSTM_Regression(DAYS_FOR_TRAIN, 8, output_size=1, num_layers=2)  # 网络初始化
    loss_function = nn.MSELoss()  # 损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)  # 优化器
    for epoch in range(EPOCHS):
        out = model(train_x)
        loss = loss_function(out, train_y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (epoch + 1) % 100 == 0:
            print('Epoch: {}, Loss:{:.5f}'.format(epoch + 1, loss.item()))
    # torch.save(model.state_dict(), 'model_params.pkl')  # 可以保存模型的参数供未来使用

    # predict
    model = model.eval()
    pred_y = model(dataset_x)  # 全量数据集的模型输出 (seq_size, batch_size, output_size)
    #print(pred_y)
    pred_y = pred_y.view(-1).data.numpy()
    #print(pred_y)
    # 对标准化数据进行还原
    #scaler = MinMaxScaler(feature_range=(-1, 1))
    actual_pred_y = scaler.inverse_transform(pred_y.reshape(-1, 1))
    actual_pred_y = actual_pred_y.reshape(-1, 1).flatten()

    test_y = scaler.inverse_transform(test_y.reshape(-1, 1))
    #print(test_y)
    test_y = test_y.reshape(-1, 1).flatten()
    #print(test_y)

    actual_pred_y = actual_pred_y[-len(test_y):]
    test_y = test_y.reshape(-1, 1)
    assert len(actual_pred_y) == len(test_y)

    # 初始结果 - 预测结果
    '''
    plt.plot(actual_pred_y, 'r', label='prediction')
    plt.plot(test_y, 'b', label='real')
    print(len(actual_pred_y),len(test_y))
    plt.plot((len(actual_pred_y), len(test_y)), (0, 1), 'g--')  # 分割线 左边是训练数据 右边是测试数据的输出
    plt.legend(loc='best')
    plt.savefig('result.png', format='png', dpi=200)
    plt.close()
    '''
    # 初始结果 - 预测结果
    plt.plot([i for i in range(len(dataset_x))],actual_pred_y,'b',label='real')
    plt.plot([i+train_size for i in range(len(test_x))],actual_pred_y, 'r', label='prediction')
    #plt.plot(test_y, 'b', label='real')
    plt.plot((len(actual_pred_y), len(test_y)), (0, 1), 'g--')  # 分割线 左边是训练数据 右边是测试数据的输出
    plt.legend(loc='best')
    plt.savefig('result.png', format='png', dpi=200)
    plt.close()