# coding:utf-8
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tushare as ts
import torch
from torch import nn
import initial_data

DAYS_FOR_TRAIN = 5
EPOCHS = 300#500#2000
import torch.functional as F


class LSTM_Regression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):  # 2
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        # self.fc =nn.Linear(hidden_size, output_size)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            # nn.Sigmoid(),
            # nn.Linear(output_size, output_size),
        )

    '''
    def attention_net(self, lstm_out, final_state,hidden_size,lstm_output):
        hidden = final_state.view(-1, hidden_size,1)  # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
        attn_weights = torch.bmm(lstm_output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step]
        soft_attn_weights = F.softmax(attn_weights, 1)
        # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
        context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
        return context, soft_attn_weights.data.numpy()  # context : [batch_size, n_hidden * num_directions(=2)]
    '''

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.fc(x)
        x = x.view(s, b, -1)
        # x=self.attention_net(x)
        return x


def create_dataset(data, days_for_train=5):  # 根据时间序列创造输入输出
    dataset_x, dataset_y = [], []
    for i in range(len(data) - days_for_train):
        _x = data[i:(i + days_for_train)]
        dataset_x.append(_x)
        dataset_y.append(data[i + days_for_train])
    return np.array(dataset_x), np.array(dataset_y)


if __name__ == '__main__':

    # 取上证指数的收盘价
    '''
    share_prices = ts.get_k_data('000001', start='2018-01-01', index=True)[
        'close'].values
    share_prices = share_prices.astype('float32')  # 转换数据类型: obj ->float
    print(share_prices.shape)
    # 上证指数收盘价作图
    '''
    share_prices = initial_data.input_()
    share_prices = share_prices.astype('float32')
    plt.plot(share_prices)
    plt.savefig('share_prices.png', format='png', dpi=200)
    plt.close()

    # 将数据集标准化到 [-1,1] 区间
    scaler = MinMaxScaler(feature_range=(-1, 1))  # train data normalized
    share_prices = scaler.fit_transform(share_prices.reshape(-1, 1))
    # print(share_prices.shape)
    ####
    # 数据集序列化，进行标签分离
    dataset_x, dataset_y = create_dataset(share_prices, DAYS_FOR_TRAIN)
    # 划分训练集和测试集,70%作为训练集,30%作为测试集
    train_size = int(len(dataset_x) * 0.7)
    train_x = dataset_x[:train_size]
    train_y = dataset_y[:train_size]
    test_x = dataset_x[train_size:]
    test_y = dataset_y[train_size:]
    # print(len(train_x))
    # print(test_x)
    # print(test_y)
    # 改变数据集形状，RNN 读入的数据维度是 (seq_size, batch_size, feature_size)
    train_x = train_x.reshape(-1, 1, DAYS_FOR_TRAIN)
    train_y = train_y.reshape(-1, 1, 1)
    # print(train_x)
    # print(train_y)
    # 数据集转为pytorch的tensor对象
    train_x = torch.from_numpy(train_x)
    train_y = torch.from_numpy(train_y)

    # train model
    model = LSTM_Regression(DAYS_FOR_TRAIN, 8, output_size=1, num_layers=2)  # 网络初始化
    loss_function = nn.MSELoss()  # 损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)  # 优化器
    los_np = []
    for epoch in range(EPOCHS):
        out = model(train_x)
        loss = loss_function(out, train_y)
        los_np.append(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (epoch + 1) % 100 == 0:
            print('Epoch: {}, Loss:{:.5f}'.format(epoch + 1, loss.item()))
    # torch.save(model.state_dict(), 'model_params.pkl')  # 可以保存模型的参数供未来使用

    plt.plot(los_np, label='train loss')
    # plt.plot(test_y, 'b', label='real')
    # plt.plot((train_size, len(test_y)), (0, 1), 'g--')  # 分割线 左边是训练数据 右边是测试数据的输出
    # plt.vlines(train_size, 0, 60000, 'g',linestyles = "dashed")
    # plt.legend(loc='best')
    # plt.savefig('result.png', format='png', dpi=200)
    plt.show()
    plt.close()

    # predict
    model = model.eval()
    dataset_x = dataset_x.reshape(-1, 1, DAYS_FOR_TRAIN)  # (seq_size, batch_size, feature_size)
    dataset_x = torch.from_numpy(dataset_x)
    #print(dataset_y.shape)
    #assert 1==0
    pred_y = model(dataset_x)  # 全量数据集的模型输出 (seq_size, batch_size, output_size)
    d1=torch.from_numpy(dataset_y.reshape(-1,1,1))
    #mn=dataset_y.mean()
    #####################
    '''
    loss = loss_function(d1, pred_y)
    tmp=0
    for i in dataset_y.reshape(-1):
        tmp+=(i-mn)*(i-mn)
    rfang=1-float(loss.item())/tmp
    #assert 1==0
    print(loss.item())
    print(rfang)
    '''
    pred_y = pred_y.view(-1).data.numpy()

    # 对标准化数据进行还原
    actual_pred_y = scaler.inverse_transform(pred_y.reshape(-1, 1))
    actual_pred_y = actual_pred_y.reshape(-1, 1).flatten()

    test_y = scaler.inverse_transform(test_y.reshape(-1, 1))
    test_y = test_y.reshape(-1, 1).flatten()

    actual_pred_y2 = actual_pred_y[-len(test_y):]
    test_y = test_y.reshape(-1, 1)
    assert len(actual_pred_y2) == len(test_y)
    # print(dataset_y)
    # 初始结果 - 预测结果
    dataset_y = scaler.inverse_transform(dataset_y.reshape(-1, 1))
    d1=dataset_y.reshape(-1)
    #print(len(d1))
    #assert 1==0
    t1=[]
    t2=[]
    for i in range(int(len(d1) * 0.7),len(d1)):
        t1.append(d1[i])
        t2.append(actual_pred_y[i])
    tmp1=0
    tmp2=0
    mn=np.mean(t1)
    for i in range(len(t1)):
        tmp1+=(t1[i]-t2[i])*(t1[i]-t2[i])
        tmp2+=(t2[i]-mn)*(t2[i]-mn)
    #print(t1)
    #print(t2)
    rfang=1-float(tmp1)/tmp2
    print('mse:'+str(tmp1/len(t1)))
    print('r^2:'+str(rfang))
    #assert 1==0
    plt.plot([i for i in range(len(dataset_x))], dataset_y, label='real')
    plt.plot([i + train_size for i in range(len(test_x))], actual_pred_y[-len(test_y):], label='prediction')
    # plt.plot(test_y, 'b', label='real')
    # plt.plot((train_size, len(test_y)), (0, 1), 'g--')  # 分割线 左边是训练数据 右边是测试数据的输出
    plt.vlines(train_size, 0, 60000, 'g', linestyles="dashed")
    plt.legend(loc='best')
    # plt.savefig('result.png', format='png', dpi=200)
    plt.show()
    plt.close()
