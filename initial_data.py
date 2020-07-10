#coding:utf-8
import xlrd
import numpy as np
def input_():
    data = xlrd.open_workbook('china_data_1.xlsx')
    data.sheet_names()
    # print("sheets：" + str(data.sheet_names()))
    table = data.sheet_by_name('Sheet1')
    data=[]
    for i in range(table.nrows):
        data.append(table.row_values(i)[1])
    #print(data)
    return np.array(data)

#input_()


def input_1():
    data = xlrd.open_workbook('usa_4_data.xlsx')
    #data.sheet_names()
    # print("sheets：" + str(data.sheet_names()))
    table = data.sheet_by_name('Sheet1')
    data=[]
    for i in range(table.nrows):
        tmp=[]
        for j in range(table.ncols):
            tmp.append(table.row_values(i)[j])
        data.append(tmp)
    data=np.array(data)
    #print(data)
    return data

#input_1()