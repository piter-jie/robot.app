import os
import dataprocess as dp
import numpy as np
import torch
import TrainDemo as TD
import streamlit as st

# def capture(original_path):  # 读取numpy文件，返回一个属性的字典
#     filenames = os.listdir(original_path)  # 得到负载文件夹下面的5个文件名
#     Data_pre = {}#0.mat:[0.23,0.325,...,],1.mat:
#     # Data_FE = {}
#     for i in filenames:  # 遍历5个文件数据
#         # 文件路径
#         file_path = os.path.join(original_path, i)  # 选定一个数据文件的路径
#         file = np.load(file_path)
#         # file_keys = file.keys()  # 字典的所有key值
#         # for key in file_keys:
#             # if 'DE' in key:  # 只获取DE
#             #     Data_DE[i] = file[key].ravel()  # 将数据拉成一维数组
#             # if 'FE' in key:  # 只获取fE
#             #     Data_FE[i] = file[key].ravel()  # 将数据拉成一维数组
#         Data_pre[i] = file.ravel()
#
#     return Data_pre
# print(capture(r'data_vali'))
# print("文件的数据长度：",len(capture(r'data_vali')['inner1.npy']))
#
# def sampling(Data_pre, step, sample_len):
#     sample_pre = []
#     # sample_FE = []
#     label = []
#     lab = 0
#     for i in Data_pre.keys():#range(len(Data_pre)):  # 遍历10个文件
#         all_length = len(Data_pre[i])  # 文件中的数据长度
#         #print('采样的训练数据总数为', all_length)
#         number_sample = int((all_length - sample_len)/step + 1)  # 样本数
#         print("number=", number_sample)
#         for j in range(number_sample):  # 逐个采样
#             sample_pre.append(Data_pre[i][j * step: j * step + sample_len])
#             # sample_FE.append(data_FE[i][j * step: j * step + sample_len])
#             label.append(lab)
#             j += 1
#         lab = lab + 1
#     #print(sample_DE)#1617,420
#     print('采样后的数据大小：',np.array(sample_pre).shape)
#     print('采样后的label大小：', np.array(label).shape)
#     # print('FE端的数据大小：', np.array(sample_FE).shape)
#     # sample = np.stack((np.array(sample_DE), np.array(sample_FE)), axis=2)
#     sample = np.expand_dims(np.array(sample_pre),axis=2)#将数据维度设置为1
#     print(sample.shape)
#     return sample, label
# sample,lable = sampling(capture(r'data_vali'),210,420)
# print(sample)
# print(lable)
Data_pre = dp.capture(r'C:/Users/矿大物联网/Desktop/3s大赛/datacnn/data_vali')
sample, label = dp.sampling(Data_pre,210,420)
print(sample.shape)

Test = dp.Dataset(torch.from_numpy(sample).permute(0, 2, 1), label)

test_loader = dp.da.DataLoader(Test, batch_size=10, shuffle=False)
# model = torch.load("model.pt")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def test(sample_data):
        TD.Con_net().to(device).load_state_dict(torch.load('C:/Users/矿大物联网/Desktop/3s大赛/datacnn/model.pth'))
        # label_list = []
        # lab = []
        # test_acc = 0.0
        # TD.Con_net().to(device).eval()
        #
        # for epoch in range(5):
        #     for img, label in sample_data:
        #         # torch.load("model.pt")
        #         img = img.float()
        #         img = img.to(device)
        #         label = label.to(device)
        #         label = label.long()
        #         lab.append(label)
        #         out = TD.Con_net().to(device)(img)
        #         out = torch.squeeze(out).float()
        #         _, pred = out.max(1)
        #         label_list.append(pred)
        #         num_correct = (pred == label).sum().item()
        #         acc = num_correct / img.shape[0]
        #
        #         #print(epoch, 'acc:', float(acc))
        #         test_acc += acc
        # 验证集的操作

        TD.Con_net().to(device).eval()
        val_acc = 0.0
        label_list = []
        lab = []
        with torch.no_grad():
            for images, labels in sample_data:
                images = images.float()
                images, labels = images.to(device), labels.to(device)
                labels = labels.long()
                lab.append(labels)
                outputs = TD.Con_net().to(device)(images)
                # loss = criterion(outputs, labels)
                # val_loss += loss.item() * images.size(0)
                _, pred = torch.max(outputs, dim=1)# 获取最大概率索引
                label_list.append(pred)
                correct = pred.eq(labels.view_as(pred))  # 返回：tensor([ True,False,True,...,False])
                accuracy = torch.mean(correct.type(torch.FloatTensor))  # 准确率，对tensor取均值
                val_acc += accuracy.item() * images.size(0)  # 累计准确率
            # 平均验证损失
            # val_loss = val_loss / len(val_loader.dataset)
            # 平均准确率
            val_acc = val_acc / len(sample_data.dataset)
        lab_list = []
        for t in lab:
            lab_list.append(t.cpu().numpy())
        result1 = np.concatenate(lab_list, axis=0)
        print('真实标签',result1)
        # 获取拼接后的预测标签值
        pre_list = []
        for t in label_list:
            pre_list.append(t.cpu().numpy())
        result2 = np.concatenate(pre_list, axis=0)
        print('预测标签',result2)
        acc = val_acc / (len(sample_data)*5)

    # print('acc:', acc)
        return acc ,label_list
acc, label = test(test_loader)
list_data = [tensor.tolist() for tensor in label]

# 初始化空的一维列表
list_1d = []

# 嵌套循环将二维列表拼接为一维列表
for sublist in list_data:
    for item in sublist:
        list_1d.append(item)
print(acc)

from collections import Counter



counts = Counter(list_1d)
most_common_num, frequency = counts.most_common(1)[0]
probability = frequency / len(list_1d)

import numpy as np
import matplotlib.pyplot as plt

# 载入数据
data = np.load('C:/Users/矿大物联网/Desktop/3s大赛/datacnn/data_vali/normal1.npy')
# figure, axes = plt.subplots()
# axes.plot(data)
# axes.set_title("Signal at Point A")
# axes.set_xlabel("Time")
# axes.set_ylabel("Amplitude")
# figure.tight_layout()
#
# st.write("Signal Plot at Point A")
# st.pyplot(figure)
# plt.plot(data[:3000])
# st.markdown('Streamlit Demo')
#
# # 设置网页标题
# st.title('一个傻瓜式构建可视化 web的 Python 神器 -- streamlit')
# plt.show()   # 显示图形

print("出现次数最多的label：", most_common_num)
if most_common_num==0:
    print('故障类型是：滚动体故障')
elif most_common_num==1:
    print('故障类型是：融合故障')
elif most_common_num==2:
    print('故障类型是：内圈故障')
elif most_common_num==3:
    print('故障类型是：正常状态')
elif most_common_num==4:
    print('故障类型是：外圈故障')
else:
    print('预测有误')
# print("出现此label的概率：", probability)


# print(list_1d)


# print(list_data)


# model = torch.load("model.pt")  # 加载模型
# model.eval()                      # 预测模式
#
# with torch.no_grad():
#     inputs = test_loader                 # 输入特征
#     outputs = model(inputs)      # 预测结果
#
# predicted = outputs.argmax(1)    # 预测标签
# print(predicted)
import time
import numpy as np
import streamlit as st

st.markdown('Streamlit Demo')
# 设置网页标题
st.title('一个傻瓜式构建可视化 web的 Python 神器 -- streamlit')
progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
last_rows = np.random.randn(1, 1)
chart = st.line_chart(last_rows)

for i in range(1, 101):
    new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    status_text.text("%i%% Complete" % i)
    chart.add_rows(new_rows)
    progress_bar.progress(i)
    last_rows = new_rows
    time.sleep(0.05)

progress_bar.empty()

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")
# 画信号波形图
figure, axes = plt.subplots()
axes.plot(data)
axes.set_title("Signal at Point A")
axes.set_xlabel("Time")
axes.set_ylabel("Amplitude")
figure.tight_layout()

st.write("Signal Plot at Point A")
st.pyplot(figure)
# plt.plot(data[:3000])
st.markdown('Streamlit Demo')
