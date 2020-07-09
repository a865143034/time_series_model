#coding:utf-8
import numpy as np
from sklearn.preprocessing import MinMaxScaler

a=np.array([[1,2,3,4,5],[6,7,8,9,10]], dtype='float64')
print('a-1D:', a, a.shape)
a=a.reshape(-1,2)
print('a-2D:', a, a.shape)

scaler_2 = MinMaxScaler(feature_range=(0, 1))  #自动将dtype转换成float64
scaled = scaler_2.fit_transform(a)
print('a-transformed:', scaled)

inv_a = scaler_2.inverse_transform(scaled)
print('a-inversed:',inv_a)