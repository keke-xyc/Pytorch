import matplotlib_inline
import torch
import torchvision
import sys
import numpy as np

from IPython import display
from numpy import argmax
import torchvision.transforms as transforms
from time import time
import matplotlib.pyplot as plt

batch_size = 256
num_inputs = 784
num_outputs = 10
num_epochs, lr = 5, 0.15

mnist_train = torchvision.datasets.FashionMNIST(root='~/data/FashionMNIST', train=True, download=True,
                                                transform=transforms.ToTensor())
# 获取训练集
mnist_test = torchvision.datasets.FashionMNIST(root='~/data/FashionMNIST', train=False, download=True,
                                               transform=transforms.ToTensor())
# 获取测试集（这两个数据集在已存在的情况下不会被再次下载）

# 生成迭代器（调用一次返回一次数据）
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=0)

test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=0)

# 初始化参数与线性回归也类似，权重参数设置为均值为0 标准差为0.01的正态分布；偏差设置为0
W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float32)

# 同样的，开启模型参数梯度
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这部分用了广播机制


def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)


def cross_entropy(y_hat, y):
    return -torch.log(y_hat.gather(1, y.view(-1, 1)))


def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1).float().mean().item())


def net_accurary(data_iter, net):
    right_sum, n = 0.0, 0
    for X, y in data_iter:
        # 从迭代器data_iter中获取X和y
        right_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        # 计算准确判断的数量
        n += y.shape[0]
        # 通过shape[0]获取y的零维度（列）的元素数量
    return right_sum / n


def sgd(params, lr, batch_size):
    # lr：学习率,params：权重参数和偏差参数
    for param in params:
        param.data -= lr * param.grad / batch_size


def train_softmax(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr, optimizer, net_accuracy):
    for epoch in range(num_epochs):
        # 损失值、正确数量、总数 初始化。
        train_l_sum, train_right_sum, n = 0.0, 0.0, 0

        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            # 数据集损失函数的值=每个样本的损失函数值的和。

            if params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()  # 对损失函数求梯度
            optimizer(params, lr, batch_size)

            train_l_sum += l.item()
            train_right_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]

        test_acc = net_accurary(test_iter, net)  # 测试集的准确率
        print('epoch %d, loss %.4f, train right %.3f, test right %.3f' % (
        epoch + 1, train_l_sum / n, train_right_sum / n, test_acc))


def get_Fashion_MNIST_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]
    # labels是一个列表，所以有了for循环获取这个列表对应的文本列表


def show_fashion_mnist(images, labels):
    matplotlib_inline.backend_inline.set_matplotlib_formats()
    # 绘制矢量图
    _, figs = plt.subplots(1, len(images), figsize=(15, 15))
    # 设置添加子图的数量、大小
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view(28, 28).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()


time1 = time()
train_softmax(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr, sgd, net_accurary)
print('\n', time() - time1, 's')

X, y = iter(test_iter).__next__()

true_labels = get_Fashion_MNIST_labels(y.numpy())
pred_labels = get_Fashion_MNIST_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

show_fashion_mnist(X[0:15], titles[0:15])
show_fashion_mnist(X[16:30], titles[16:30])