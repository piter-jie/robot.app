import numpy as np
import dataprocess as RD
import torch
import torch.nn as nn
from torch.utils import data as da
from timm.loss import LabelSmoothingCrossEntropy
import argparse
import cm as c
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
args = None

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--data_dir', type=str, default= "data", help='')
    parser.add_argument("--pretrained", type=bool, default=True, help='')
    parser.add_argument('--batch_size', type=int, default=256, help='batchsize of the training process')
    parser.add_argument('--step_len', type=list, default=range(210, 430, 10), help='')
    parser.add_argument('--sample_len', type=int, default=420, help='')
    parser.add_argument('--rate', type=list, default=[0.7, 0.15, 0.15], help='')
    parser.add_argument('--acces', type=list, default=[], help='initialization list')
    parser.add_argument('--epochs', type=int, default=5, help='max number of epoch')#80
    parser.add_argument('--losses', type=list, default=[], help='initialization list')
    args = parser.parse_args()
    return args

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
# 这个类的作用是将数据和标签打包成数据集对象，方便在PyTorch中进行数据加载和训练
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

# STEP1 加载数据
def load_data(args, step):
    # path = r'data\5HP'
    # path = args.data_dir
    # rate = args.rate
    #return x_train, y_train, x_validate, y_validate, x_test, y_test
    x_train, y_train, x_validate, y_validate, x_test, y_test = RD.get_data(args.data_dir, args.rate, step, args.sample_len)
    # 切片
    # sample = tf.data.Dataset.from_tensor_slices((x_train, y_train))   # 按照样本数进行切片得到每一片的表述（2048+10，1）
    # sample = sample.shuffle(1000).batch(10)  # 打乱分批量(10,400,2)
    # sample_validate = tf.data.Dataset.from_tensor_slices((x_validate, y_validate))
    # sample_validate = sample_validate.shuffle(1000).batch(10)  # 打乱分批量(10,400,2)
    # sample_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    # sample_test = sample_test.shuffle(1000).batch(10)  # 打乱分批量(10,400,2)
    print(x_train.shape)
    Train = Dataset(torch.from_numpy(x_train).permute(0,2,1), y_train)#.permute(1,0)
    print(type(train))
    #将数据集的维度从(样本数, 时间步, 特征数)转换为(样本数, 特征数, 时间步)。将数据集的时间步维度放到最后，使得数据集能够被PyTorch的卷积层正确地处理
    Validate = Dataset(torch.from_numpy(x_validate).permute(0,2,1), y_validate)
    Test = Dataset(torch.from_numpy(x_test).permute(0,2,1), y_test)
    train_loader = da.DataLoader(Train, batch_size=args.batch_size, shuffle=True)
    print(type(train_loader))
    validate_loader = da.DataLoader(Validate, batch_size=args.batch_size, shuffle=True)
    test_loader = da.DataLoader(Test, batch_size=args.batch_size, shuffle=False)
    return train_loader, validate_loader, test_loader


# STEP2 设计网络结构，建立网络容器
# def create_model():
#     Con_net = keras.Sequential([  # 网络容器
#         layers.Conv1D(filters=32, kernel_size=20, strides=1, padding='same', activation='relu'),  # 添加卷积层
#         layers.BatchNormalization(),  # 添加正则化层
#         layers.MaxPooling1D(pool_size=2, strides=2, padding='same'),  # 池化层
#         layers.Conv1D(filters=32, kernel_size=20, strides=1, padding='same', activation='relu'),  # 添加卷积层
#         layers.BatchNormalization(),  # 添加正则化层
#         layers.MaxPooling1D(pool_size=2, strides=2, padding='same'),  # 池化层
#         layers.Flatten(),  # 打平层，方便全连接层使用
#         layers.Dense(100, activation='relu'),  # 全连接层，120个节点
#         layers.Dense(10, activation='softmax'),  # 全连接层，10个类别节点
#     ])
#     return Con_net
# 输入数据大小（256,1,420）
class Con_net(nn.Module):
    def __init__(self):
        super(Con_net, self).__init__()
        self.p1_1 = nn.Sequential(nn.Conv1d(1, 32, kernel_size=20, stride=1, padding='same'),
                                  nn.BatchNorm1d(32),
                                  nn.ReLU(inplace=True))
        self.p1_2 = nn.MaxPool1d(2, 2)
        self.p2_1 = nn.Sequential(nn.Conv1d(32, 32, kernel_size=20, stride=1, padding='same'),
                                  nn.BatchNorm1d(32),
                                  nn.ReLU(inplace=True))
        self.p2_2 = nn.MaxPool1d(2, 2)
        self.p3_1 = nn.Sequential(nn.Linear(32*105, 100),     #需要根据输出修改前置神经元个数
                                  nn.ReLU(inplace=True))   #全连接层之后还需要加激活函数
        self.p3_2 = nn.Sequential(nn.Linear(100, 5))

    def forward(self, x):
        x = self.p1_2(self.p1_1(x))
        x = self.p2_2(self.p2_1(x))
        x = x.reshape(-1, x.size()[1]*x.size()[2])
        x = self.p3_2(self.p3_1(x))
        return x
# class Con_net(nn.Module):
#     def __init__(self):
#         super(Con_net, self).__init__()
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=5)
#         self.conv2 = nn.Conv1d(in_channels=6, out_channels=12, kernel_size=5)
#         self.relu = nn.ReLU()
#         self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
#
#         self.fc1 = nn.Linear(in_features=12 * 4 * 4, out_features=120)
#         self.fc2 = nn.Linear(in_features=120, out_features=60)
#         self.out = nn.Linear(in_features=60, out_features=5)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = self.fc2(x)
#         x = self.out(x)
#
#         return x
model1 = Con_net()
model = Con_net().to(device)
#保存模型参数
# model_state_dict = model.state_dict()
# torch.save(model_state_dict, "model.pt")
# print(model)
#加载模型
#model.load_state_dict(torch.load('model.pt'))


def train(args, train_loader, validate_loader, test_loader):
    res = []
    lab = []
    pre = []
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = LabelSmoothingCrossEntropy()
    train_loss = 0.0
    train_acc = 0.0
    model.load_state_dict(torch.load('model.pth'))
    model.train()
    for epoch in range(args.epochs):
        for step, (img, label) in enumerate(train_loader):
            img = img.float()
            img = img.to(device)
            label = label.to(device)
            label = label.long()
            lab.append(label)
            out = model(img)
            # pre.append(out)#收集预测标签
            out = torch.squeeze(out).float()
            loss = criterion(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, pred = out.max(1)
            # 将模型输出的得分或概率最大的类别作为预测结果
            pre.append(pred)
            #num_correct = (pred == label).sum().detach().cpu().numpy()
            num_correct = (pred == label).sum().item()
            # bool_tensor = pred == label
            # bool_array = bool_tensor.to('cpu').numpy()
            # num_correct = bool_array.astype(int).sum()
            acc = num_correct / img.shape[0]
            train_acc += acc
            #print(step)
            if step % 100 == 0:
                print(epoch, step, 'loss:', float(loss))
            # print(epoch, step, 'loss:', float(loss))
            res.append(test(test_loader))
            #step+=1
        result.append(res)
        # if epoch % 10 == 0:
    #         torch.save(model.state_dict(), 'model.pth')
    # torch.save(model.state_dict(), 'model.pth')
    # train_preds = c.get_all_preds(model,train_loader)
    # print(train_preds.shape)  # (6000,10)
    #验证集的操作

    model.eval()
    val_acc = 0.0
    with torch.no_grad():
        for images, labels in validate_loader:
            images = images.float()
            images, labels = images.to(device), labels.to(device)
            labels = labels.long()
            outputs = model(images)
            # loss = criterion(outputs, labels)
            # val_loss += loss.item() * images.size(0)
            _, pred = torch.max(outputs, dim=1)  # 获取最大概率索引
            correct = pred.eq(labels.view_as(pred))  # 返回：tensor([ True,False,True,...,False])
            accuracy = torch.mean(correct.type(torch.FloatTensor))  # 准确率，对tensor取均值
            val_acc += accuracy.item() * images.size(0)  # 累计准确率
        # 平均验证损失
        # val_loss = val_loss / len(val_loader.dataset)
        # 平均准确率
        val_acc = val_acc / len(validate_loader.dataset)
    #获取拼接后的真实标签值
    lab_list = []
    for t in lab:
        lab_list.append(t.cpu().numpy())
    result1 = np.concatenate(lab_list, axis=0)
    print(result1)
    #获取拼接后的预测标签值
    pre_list = []
    for t in pre:
        pre_list.append(t.cpu().numpy())
    result2 = np.concatenate(pre_list, axis=0)
    print(result2)
    #绘制混淆矩阵
    cm = c.confusion_matrix(result1, result2)
    print(cm)
    target_names = ('ball', 'combin', 'inner', 'normal', 'outer')
    c.plot_confusion_matrix(cm, target_names)

    # print(lab_list)
    # print(pre)
    return test(test_loader),val_acc

        # torch.save(model.state_dict(), './cnn_save_weights_400.pt')

# def train(sample1, sample1_validate, sample1_test, sample_len):
#     res = []
#     Con_net = create_model()  # 建立网络模型
#     Con_net.build(input_shape=(10, sample_len, 2))  # 构建一个卷积网络，输入的尺寸  ----------------------
#     optimizer = optimizers.Adam(lr=1e-4)  # 设置优化器
#     variables = Con_net.trainable_variables
#     for epoch in range(epochs):  # 外循环，遍历多少次训练数据集
#         for step, (x, y) in enumerate(sample1):  # 遍历一次训练集的所有样例
#             with tf.GradientTape() as tape:  # 构建梯度环境 # [b, 32, 32, 3] => [b, 1, 1, 512]
#                 out = Con_net(x)  # flatten, => [b, 512]
#                 loss = tf.losses.categorical_crossentropy(y, out)  # compute loss
#                 loss = tf.reduce_mean(loss)  # 求损失的平均值
#             grads = tape.gradient(loss, variables)
#             optimizer.apply_gradients(zip(grads, variables))
#             if step % 1000 == 0:
#                 print(epoch, step, 'loss:', float(loss))
#         # print("验证集正确率")
#         # test(Con_net, sample1_validate)
#         # print("测试集正确率")
#         res.append(test(Con_net, sample1_test))
#     result.append(res)
#     # Con_net.save_weights('./cnn_save_weights_400')

def test(sample_data):
    # label_list = []
    test_acc = 0.0
    model.eval()
    for img, label in sample_data:
        # torch.load('./cnn_save_weights_400.pt')
        img = img.float()
        img = img.to(device)
        label = label.to(device)
        label = label.long()
        out = model(img)
        out = torch.squeeze(out).float()
        _, pred = out.max(1)
        # label_list.append(pred)
        num_correct = (pred == label).sum().item()
        acc = num_correct / img.shape[0]
        test_acc += acc
    acc = test_acc / len(sample_data)
    # print('acc:', acc)
    return acc
#获得所有的预测标签

#
#
# def test(Con_net, sample_data):
#     total_num = 0
#     total_correct = 0
#     for x, y in sample_data:
#         # Con_net = create_model()  # 建立网络模型
#         # Con_net.load_weights('./cnn_save_weights_400')
#         out = Con_net(x)  # 前向计算
#         predict = tf.argmax(out, axis=-1)  # axis=-1, 倒数第一维, 返回每行的最大值坐标
#         # print("predict", predict)
#         y = tf.cast(y, tf.int64)
#         # print("y", y)
#         m = predict == y
#         m = tf.cast(m, dtype=tf.int64)   # tensor张量类型
#         total_correct += int(tf.reduce_sum(m))
#         total_num += x.shape[0]
#         if total_num < total_correct:
#             print("error---------------------------")
#             print("正确",total_correct,"总数",total_num)
#     acc = total_correct / total_num
#     # print('acc:', acc)
#     return acc

def run_step(args):  # epoch=10
    # step_len = list(range(210, 430, 10))
    #step_len = [420]
    # step_len = [210]
    # for i in list(args.step_len):
        sample1, sample1_validate, sample1_test = load_data(args, step=210)
        print(len(sample1))
        acc,acc_val = train(args, sample1, sample1_validate, sample1_test)
        print('测试准确率为:',acc)
        print('验证准确率为:', acc_val)
        # print(sample1)
        if(acc>0.99):
            print("预测结果：内圈故障")
        #
        # with torch.no_grad():  # disable gradient computations仅仅获得标签，省去了大量不必要的计算
        # train_preds = c.get_all_preds(model, sample1)
        # print(train_preds.shape) # (6000,10)
        # print(train_preds.requires_grad) # False
        # print('true labels: ', train_set.targets.numpy())
        #print('pred labels: ', train_preds.argmax(dim=1).numpy())


# def run_sample():
#     sample_len = list(range(1,7))
#     # sample_len = [1]
#     for i in sample_len:
#         sample1, sample1_validate, sample1_test = load_data(step=210, sample_len=420*i)
#         train(sample1, sample1_validate, sample1_test, sample_len=420*i)


# 当epoch=10时，随着步长的变化，实验结果的变化
if __name__ == '__main__':
    args = parse_args()
    result = []
    run_step(args)
    # run_sample()
    #print(result[0].sum()/len(result[0]))
    # print(result)
    # print(len(result[0]))
    # print(len(result))
    # print(result[0])
    ave_acc = 0.0
    count = 0
    for j in range(len(result)):
        for i in range(len(result[j])):
            ave_acc += result[j][i]
            count += 1
    print('测试平均准确率:',ave_acc/count)
    #  训练正确率曲线


    # T = result[len(result)-1]
    # #power = np.array([1.53E+03, 5.92E+02, 2.04E+02, 7.24E+01, 2.72E+01, 1.10E+01, 4.70E+00])
    # print(T)

    # plt.plot(T)
    #
    # plt.ylim(0, 1)
    # plt.show()
# import matplotlib.pyplot as plt
# import seaborn as sn
# from sklearn.metrics import confusion_matrix
#
# a,b,c,d,e,y =
# cm = confusion_matrix(y_train, y_pred)
# sn.heatmap(cm, annot=True, fmt=".3f", xticklabels=['class0', 'class1', 'class2'],
#                yticklabels=['class0', 'class1', 'class2'])
# plt.ylabel('Actual')
# plt.xlabel('Predicted')
# plt.show()







