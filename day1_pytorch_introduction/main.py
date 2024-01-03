import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np


def creat_tensor():
    """
    创建一个普通的张量
    :return: NULL
    """
    tensor = torch.tensor([1, 2, 3, 4, 5])
    print(tensor.shape)


def operate1_tensor():
    """
    张量的一一系列操作
    :return: NULL
    """
    # 创建一个零张量
    zeros_tensor = torch.zeros((2, 3))
    print(zeros_tensor)

    # 随机张量(每个数值的范围似乎是(0,1))
    rand_tensor = torch.rand((3, 3))
    print(rand_tensor)

    # 张量的加减乘除
    tensor1 = torch.tensor([1, 2, 3])
    tensor2 = torch.tensor([4, 5, 6])

    result = tensor1 + tensor2
    print(result)

    result = tensor1 - tensor2
    print(result)

    result = tensor1 * tensor2
    print(result)

    result = tensor1 / tensor2
    print(result)


def operate2_tensor():
    """
    张量的操作
    :return: NULL
    """
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(tensor[0][0]) 与这个等价
    print(tensor[0, 0])  # 注意元素的访问方式
    print(tensor[:, 2])

    # 在PyTorch中，可以将张量存储在CPU或GPU上。GPU加速可以显著提高深度学习模型的计算性能。要将张量移动到GPU上，可以使用.to方法。
    tensor = torch.tensor([[1, 2], [3, 4]])
    tensor = tensor.to('cpu')  # gpu好像不行
    print(tensor)


def creat2_tensor():
    """
    张量的创建
    :return: NULL
    """

    # 空张量
    # 空张量的元素值将不会被初始化，它们的内容是未知的。
    x = torch.empty((2, 2))
    print(x)

    # 零张量
    x = torch.zeros((2, 2))
    print(x)

    # 随机张量
    # 也可以这样 torch.rand(2,2)
    # 取值范围[0, 1)
    x = torch.rand((2, 2))
    print(x)

    # 全1张量
    x = torch.ones((2, 2))
    print(x)

    # 从数据中创建张量
    data_list = [1, 2, 3, 4, 5]  # 普通列表创建张量
    tensor = torch.tensor(data_list)
    print(tensor)
    data_array = np.array([6, 7, 8, 9])
    tensor = torch.tensor(data_array)  # 从numpy数据中创建张量
    print(tensor)

    # 具有特定数据类型的张量
    x = torch.tensor([1, 2, 3], dtype=torch.float)
    print(x.dtype)
    x = torch.zeros(3, 3, dtype=torch.float)
    print(x.dtype)


def operate3_tensor():
    """
    张量的运算
    :return:
    """
    # 张量与标量的运算（就是与单个数值的运算）
    a = torch.tensor([1, 2, 3])
    b = 2

    c = a + b
    print(c)

    c = a - b
    print(c)

    c = a * b
    print(c)

    c = a / b
    print(c)

    # 张量的变形
    a = torch.tensor([[1, 2, 3], [4, 5, 6]])
    b = a.resize(3, 2)
    print(b)
    b = a.view(3, 2)
    print(b)
    b = a.transpose(0, 1)
    print(b)  # 张量转置

    # 张量的切片
    a = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    slice = a[:, 1:3]  # 注意这个切片的范围是 1 2 没有3
    print(slice)

    # 从张量中获取数值
    number = a[0][0].item()
    print(number)

    # 将张量转化为列表（返回转化的结果）
    data_list = a.tolist()
    print(data_list)


def differentiate_tensor():
    """
    张量的求导
    在PyTorch中，自动求导是一个关键的功能，它允许我们自动计算梯度，从而进行反向传播和优化。
    在深度学习中，梯度是一个非常重要的概念，它表示了函数在某一点的变化率。在PyTorch中，计算梯度是一项关键操作，它允许我们通过反向传播算法有效地更新模型参数。
    梯度是一个向量，其方向指向函数值增长最快的方向，而其大小表示函数值的变化率。在深度学习中，我们通常希望最小化损失函数，因此我们要沿着梯度的反方向更新模型参数，以逐步降低损失值。

    PyTorch中的torch.Tensor类是PyTorch的核心数据结构，同时也是计算梯度的关键。每个张量都有一个属性requires_grad，默认为False。如果我们希望计算某个张量的梯度，需要将requires_grad设置为True，那么就会开始追踪在该变量上的所有操作，而完成计算后，可以调用 .backward() 并自动计算所有的梯度，得到的梯度都保存在属性 .grad 中。

    调用 .detach() 方法分离出计算的历史，可以停止一个 tensor 变量继续追踪其历史信息 ，同时也防止未来的计算会被追踪。而如果是希望防止跟踪历史（以及使用内存），可以将代码块放在 with torch.no_grad(): 内，这个做法在对模型进行评估的时候非常有用（节约算力、不会发生模型参数变化）。

    对于 autograd 的实现，还有一个类也是非常重要--Function 。Tensor 和 Function 两个类是有关联并建立了一个非循环的图，可以编码一个完整的计算记录。每个 tensor 变量都带有属性 .grad_fn ，该属性引用了创建了这个变量的 Function （除了由用户创建的 Tensors，它们的 grad_fn二小节介绍梯度的内容。

    :return:NULL
    """

    # 创建张量并设置requires_grad=True：
    x = torch.tensor([2.0, 3.0], requires_grad=True)
    y = x ** 2 + 3 * x + 1
    # 计算梯度
    y.sum().backward()  # 无返回值
    # 获取梯度
    print(x.grad)

    ## 梯度计算的示例
    # 示例 1：线性回归
    #
    # 考虑一个简单的线性回归模型，我们的目标是找到一条直线，以最小化预测值与真实值之间的平方误差。我们可以使用梯度下降算法来更新直线的参数。

    # 创建训练数据
    x_train = torch.tensor([[1.0], [2.0], [3.0]])
    y_train = torch.tensor([[2.0], [4.0], [6.0]])

    # 定义模型参数
    w = torch.tensor([[0.0]], requires_grad=True)
    b = torch.tensor([[0.0]], requires_grad=True)

    def linear_regression(x):
        return torch.matmul(x, w) + b

    # 定义损失函数
    def loss_fn(y_pred, y):
        return torch.mean((y_pred - y) ** 2)

    # 定义优化器
    optimizer = torch.optim.SGD([w, b], lr=0.01)

    # 训练模型
    for epoch in range(100):
        # 前向传播
        y_pred = linear_regression(x_train)

        # 计算损失
        loss = loss_fn(y_pred, y_train)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        # 清零梯度
        optimizer.zero_grad()


class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入维度为1，输出维度为1

    def forward(self, x):
        return self.linear(x)


def Backpropagation():
    """
    反向传播（Backpropagation，缩写为BP）是“误差反向传播”的简称，该方法对网络中所有权重计算损失函数的梯度。 这个梯度会反馈给梯度下降法，用来更新权重值以最小化损失函数，从而训练神经网络模型。
    :return:
    """
    model = LinearRegression()
    criterion = nn.MSELoss()  # 均方误差损失函数

    # 随机梯度下降优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    # 生成样本数据
    x_train = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y_train = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

    # 训练模型
    for epoch in range(100):
        optimizer.zero_grad()  # 梯度清零

        # 前向传播
        y_pred = model(x_train)

        # 计算损失
        loss = criterion(y_pred, y_train)

        # 反向传播
        loss.backward()

        # 参数更新
        optimizer.step()

if __name__ == '__main__':
    # creat_tensor()
    # operate1_tensor()
    # operate2_tensor()
    # creat2_tensor()
    # operate3_tensor()
    # differentiate_tensor()
    Backpropagation()
