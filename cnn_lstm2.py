#coding:utf-8
#导入必要的库
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam

#设置LSTM的时间窗等参数
window=5
lstm_units = 16
dropout = 0.01
epoch=400#60
#读取数据
df1=pd.read_csv('data1.csv')
df1=df1.iloc[:,2:]
df1.tail()


#进行数据归一化
from sklearn import preprocessing
#min_max_scaler = preprocessing.MinMaxScaler()
#df0=min_max_scaler.fit_transform(df1)
df = pd.DataFrame(df1, columns=df1.columns)
input_size=len(df.iloc[1,:])


#构建lstm输入
seq_len=window

stock=df
amount_of_features = len(stock.columns)#有几列
data=stock.iloc[:,:].values
sequence_length = seq_len + 1#序列长度

r1 = []
for index in range(sequence_length,len(data)):#循环数据长度-sequence_length次
    r1.append([data[index][3]])#第i行到i+sequence_length
r1 = np.array(r1)#得到样本，样本形式为6天*3特征


####第二次重置
min_max_scaler = preprocessing.MinMaxScaler()
df0=min_max_scaler.fit_transform(df1)
df = pd.DataFrame(df0, columns=df1.columns)
input_size=len(df.iloc[1,:])


#构建lstm输入
seq_len=window

stock=df
amount_of_features = len(stock.columns)#有几列
data=stock.iloc[:,:].values
sequence_length = seq_len + 1#序列长度
'''
min1 = preprocessing.MinMaxScaler()
df0=min1.fit_transform(df1)
df = pd.DataFrame(df1, columns=df1.columns)
input_size=len(df.iloc[1,:])
stock=df
amount_of_features = len(stock.columns)#有几列
data=stock.iloc[:,:].values
'''
result = []
for index in range(len(data) - sequence_length):#循环数据长度-sequence_length次
    result.append(data[index: index + sequence_length])#第i行到i+sequence_length
result = np.array(result)#得到样本，样本形式为6天*3特征
#print(result)



min2 = preprocessing.MinMaxScaler()
r1=min2.fit_transform(r1)


row = round(0.7 * result.shape[0])#划分训练集测试集
train = result[:int(row), :]
x_train = train[:, :-1]
#print(x_train)
y_train = train[:, -1][:,-1]
#print(y_train)
x_test = result[int(row):, :-1]
y_test = result[int(row):, -1][:,-1]
x_all=np.concatenate((x_train,x_test),axis=0)
y_all=np.concatenate((y_train,y_test),axis=0)
#reshape成 6天*3特征
X_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
X_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))

X_all = np.reshape(x_all, (x_all.shape[0], x_all.shape[1], amount_of_features))





#建立LSTM模型 训练
inputs=Input(shape=(window, input_size))
model=Conv1D(filters = lstm_units, kernel_size = 1, activation = 'sigmoid')(inputs)#卷积层
model=MaxPooling1D(pool_size = window)(model)#池化层
model=Dropout(dropout)(model)#droupout层
model=Bidirectional(LSTM(lstm_units, activation='tanh'), name='bilstm')(model)#双向LSTM层
attention=Dense(lstm_units*2, activation='sigmoid', name='attention_vec')(model)#求解Attention权重
model=Multiply()([model, attention])#attention与LSTM对应数值相乘
outputs = Dense(1, activation='tanh')(model)
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
model.summary()#展示模型结构

history=model.fit(X_train, y_train, nb_epoch = epoch, batch_size = 256,shuffle=False,validation_data=(X_test, y_test)) #训练模型epoch次


#迭代图像
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(epoch)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Test Loss')
plt.legend(loc='upper right')
#plt.title('Train and Val Loss')
plt.show()


#在训练集上的拟合结果
y_train_predict=model.predict(X_train)
y_train_predict=min2.inverse_transform(y_train_predict)
y_train=min2.inverse_transform(y_train.reshape(-1,1))
#print(y_train_predict.shape)
#assert 1==0
#y_train_predict=min_max_scaler.inverse_transform(y_train_predict)
#print(y_train_predict)
#print(y_train_predict)
#print(y_train)
y_train_predict=y_train_predict.squeeze(1)
plt.plot([i for i in range(len(y_train_predict))],y_train_predict,label='predict')
plt.plot([i for i in range(len(y_train_predict))],y_train,label='real')
plt.legend(['real','predict'])
#plt.title("Train Data",fontsize='30') #添加标题
#plt.show()


y_test_predict=model.predict(X_test)
y_test_predict=min2.inverse_transform(y_test_predict)
y_test=min2.inverse_transform(y_test.reshape(-1,1))
y_test_predict=y_test_predict.squeeze(1)
plt.plot([i+len(y_train_predict) for i in range(len(y_test_predict))],y_test_predict,label='predict')
plt.plot([i+len(y_train_predict) for i in range(len(y_test_predict))],y_test,label='real')
plt.legend(['real','predict'])
plt.vlines(len(y_train_predict), 0, 1, colors="g", linestyles="dashed")
#plt.title("Test Data",fontsize='30') #添加标题
plt.show()

y_all_predict=model.predict(X_all)
#y1=y_all_predict
y_all_predict=min2.inverse_transform(y_all_predict)
y_all=min2.inverse_transform(y_all.reshape(-1,1))
#y_test_predict=min2.inverse_transform(y_test_predict)
y_all_predict=y_all_predict.squeeze(1)
plt.plot([i for i in range(len(y_all_predict))],y_all_predict,label='predict')
plt.plot([i for i in range(len(y_all_predict))],y_all,label='real')
plt.legend(['real','predict'])
plt.vlines(len(y_train_predict), 0, 70000, colors="g", linestyles="dashed")
#plt.title("Test Data",fontsize='30') #添加标题
plt.show()
'''
y_train_predict=y_train_predict[:,0]
draw=pd.concat([pd.DataFrame(y_train),pd.DataFrame(y_train_predict)],axis=1)
draw.iloc[200:500,0].plot(figsize=(12,6))
draw.iloc[200:500,1].plot(figsize=(12,6))
plt.legend(('real', 'predict'),fontsize='15')
plt.title("Train Data",fontsize='30') #添加标题
#plt.show()
'''
'''
#在测试集上的预测
y_test_predict=model.predict(X_test)
y_test_predict=y_test_predict[:,0]
draw=pd.concat([pd.DataFrame(y_test),pd.DataFrame(y_test_predict)],axis=1);
draw.iloc[200:500,0].plot(figsize=(12,6))
draw.iloc[200:500,1].plot(figsize=(12,6))
plt.legend(('real', 'predict'),loc='upper right',fontsize='15')
plt.title("Test Data",fontsize='30') #添加标题
#plt.show()
'''
#输出结果
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100
def up_down_accuracy(y_true, y_pred):
    y_var_test=y_true[1:]-y_true[:len(y_true)-1]#实际涨跌
    y_var_predict=y_pred[1:]-y_pred[:len(y_pred)-1]#原始涨跌
    txt=np.zeros(len(y_var_test))
    for i in range(len(y_var_test-1)):#计算数量
        txt[i]=np.sign(y_var_test[i])==np.sign(y_var_predict[i])
    result=sum(txt)/len(txt)
    return result
print('训练集上的MAE/MSE/MAPE/涨跌准确率')
print(mean_absolute_error(y_train_predict, y_train))
print(mean_squared_error(y_train_predict, y_train) )
print(mape(y_train_predict, y_train) )
print(up_down_accuracy(y_train_predict,y_train))
print('测试集上的MAE/MSE/MAPE/涨跌准确率')
print(mean_absolute_error(y_test_predict, y_test))
print(mean_squared_error(y_test_predict, y_test) )
mse=mean_squared_error(y_test_predict, y_test)
mn=y_test.mean()
'''
tmp=0
for i in y_test.reshape(-1):
    tmp+=(i-mn)*(i-mn)
rfang=1-float(mse)/tmp
'''
y1=y_test.reshape(-1)
y2=y_test_predict.reshape(-1)
t1=0
t2=0
for i in range(len(y1)):
    t1+=(y1[i]-y2[i])*(y1[i]-y2[i])
    t2+=(y1[i]-mn)*(y1[i]-mn)
print(t1/len(y1))
rfang=1-float(t1)/t2
print(rfang)
#print(y_test.shape,y_test_predict.shape)
#print(mn)
#print(rfang)
print(y_test.shape)
print(mape(y_test_predict,  y_test) )
print(up_down_accuracy(y_test_predict,y_test))
