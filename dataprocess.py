import os

import numpy as np
import torch
from torch.utils import data as da

#读取数据
# path = r'data\ball1.npy'
# data = np.load(path)
# print(data.shape)

# 构建文件读取函数capture,返回原始数据和标签数据
def capture(original_path):  # 读取numpy文件，返回一个属性的字典
    filenames = os.listdir(original_path)  # 得到负载文件夹下面的5个文件名
    Data_pre = {}#0.mat:[0.23,0.325,...,],1.mat:
    # Data_FE = {}
    for i in filenames:  # 遍历5个文件数据
        # 文件路径
        file_path = os.path.join(original_path, i)  # 选定一个数据文件的路径
        file = np.load(file_path)  # 字典
        # file_keys = file.keys()  # 字典的所有key值
        # for key in file_keys:
            # if 'DE' in key:  # 只获取DE
            #     Data_DE[i] = file[key].ravel()  # 将数据拉成一维数组
            # if 'FE' in key:  # 只获取fE
            #     Data_FE[i] = file[key].ravel()  # 将数据拉成一维数组
        Data_pre[i] = file.ravel()
    print('数据的总长度为',len(Data_pre))

    return Data_pre
# print(len(capture(r'data')))
# 划分训练样点集和测试样点集和验证样本集0.7:0.15:0.15
def spilt(data, rate):  # [[N1],[N2],...,[N10]]
    keys = data.keys()  # 5个文件名
    #print(keys)
    tra_data = []
    te_data = []
    val_data = []
    for i in keys:  # 遍历所有文件夹
        slice_data = data[i]  # 选取1个文件中的数据
        # slice_data = scalar_stand(slice_data)
        all_length = len(slice_data)  # 文件中的数据长度
        #print('数据总数为', all_length)
        tra_data.append(slice_data[0:int(all_length * rate[0])])
        # print("训练样本点数", len(tra_data))
        val_data.append((slice_data[int(all_length * rate[0]):int(all_length * (rate[0]+rate[1]))]))
        te_data.append(slice_data[int(all_length * (rate[0]+rate[1])):])
        # print("测试样本点数", len(te_data))
    #print(len(val_data[0]))
    return tra_data, val_data, te_data


#数据采样，步长为step，样本长度sample_len
def sampling(Data_pre, step, sample_len):
    sample_pre = []
    # sample_FE = []
    #设置验证集的label
    label_val= []
    label = []
    lab = 0
    lab1 = 4
    for i in Data_pre.keys():#range(len(Data_pre)):  # 遍历10个文件
        all_length = len(Data_pre[i])  # 文件中的数据长度
        #print('采样的训练数据总数为', all_length)
        number_sample = int((all_length - sample_len)/step + 1)  # 样本数
        #print("number=", number_sample)
        for j in range(number_sample):  # 逐个采样
            sample_pre.append(Data_pre[i][j * step: j * step + sample_len])
            # sample_FE.append(data_FE[i][j * step: j * step + sample_len])
            label.append(lab)
            label_val.append(lab1)
            j += 1
        lab = lab + 1
        lab1 += 1
    #print(sample_DE)#1617,420
    print('采样后的数据大小：',np.array(sample_pre).shape)
    print('采样后的label大小：', np.array(label).shape)
    # print('FE端的数据大小：', np.array(sample_FE).shape)
    # sample = np.stack((np.array(sample_DE), np.array(sample_FE)), axis=2)
    sample = np.expand_dims(np.array(sample_pre),axis=2)#将数据维度设置为1
    print(sample.shape)
    return sample, label
# Data_pre = capture(r'data')
# sampling(Data_pre,210,420)

def get_data(path, rate, step, sample_len):
    Data_pre= capture(path)  # 读取数据返回字典数据
    # print(Data_pre)
    train_data_DE, val_data_DE, test_data_DE = spilt( Data_pre, rate)  # 列表[N1,N2,N10]
    # train_data_FE, val_data_FE, test_data_FE = spilt( Data_pre, rate)
    #print(len(val_data_DE[1]))#(10,72846)
    print(val_data_DE)
    x_train, y_train = sampling(train_data_DE,  step, sample_len)
    # y_train = F.one_hot(torch.Tensor(y_train).long(), num_classes=10)
    x_validate, y_validate = sampling(val_data_DE, step, sample_len)
    #print(x_validate.shape)#(1117, 420, 2)
    x_test, y_test = sampling(test_data_DE,  step, sample_len)
    return x_train, y_train, x_validate, y_validate, x_test, y_test

class Dataset(da.Dataset):
    def __init__(self, X, y):
        self.Data = X
        self.Label = y
    def __getitem__(self, index):
        txt = self.Data[index]
        label = self.Label[index]
        return txt, label
    def __len__(self):
        return len(self.Data)

if __name__ == "__main__":
    path = r'data'
    rate = [0.7, 0.15, 0.15]
    step = 210
    sample_len = 420
    x_train, y_train, x_validate, y_validate, x_test, y_test = get_data(path, rate, step, sample_len)
    print('训练集大小',x_train.shape)  # (5267, 420, 2)
    print('验练集大小',x_validate.shape) # (1117, 420, 2)
    print('测试集大小',x_test.shape)  # (1117, 420, 2)
    print('测试集标签大小', len(y_test))
    # sample = tf.data.Dataset.from_tensor_slices((x_train, y_train))   # 按照样本数进行切片得到每一片的表述（400，2）
    # sample = sample.shuffle(1000).batch(10)  # 打乱分批量(10,400,2)
    # sample_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    # sample_test = sample_test.shuffle(1000).batch(10)  # 打乱分批量(10,400,2)
    Train = Dataset(torch.from_numpy(x_train).permute(0,2,1), y_train)#转置
    Test = Dataset(torch.from_numpy(x_test).permute(0,2,1), y_test)
    train_loader = da.DataLoader(Train, batch_size=10, shuffle=True)
    test_loader = da.DataLoader(Test, batch_size=10, shuffle=False)