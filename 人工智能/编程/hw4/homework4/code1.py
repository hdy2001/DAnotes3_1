import torch
import torchvision
from model import SoftmaxNet
import torch.nn as nn
import torch.optim as optim
import comment

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)
mse_loss = nn.MSELoss()

# 加载数据集
train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
    './data/',
    train=True,
    download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
    ])),
                                           batch_size=batch_size_train,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(
    './data/',
    train=False,
    download=True,
    transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307, ), (0.3081, ))
    ])),
                                          batch_size=batch_size_test,
                                          shuffle=True)

# 初始化网络
network = SoftmaxNet()
optimizer = optim.SGD(network.parameters(),
                      lr=learning_rate,
                      momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]


# 训练
def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        # 变为独热码
        target = target.reshape(-1, 1)
        one_hot = torch.zeros(data.shape[0], 10).scatter(1, target, 1)
        loss = mse_loss(output, one_hot)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) +
                                 ((epoch - 1) * len(train_loader.dataset)))


# 测试：后期需要加入其他评判指标
def test():
    network.eval()
    with torch.no_grad():
        _, (data, targets) = next(enumerate(test_loader))
        output = network(data)
        pred = output.data.max(1, keepdim=True)[1].reshape(-1)
        comment.test_sklearn(targets.numpy(), pred.numpy())


# 卷积神经网络进行数字识别
for epoch in range(1, n_epochs + 1):
    train(epoch)

test()
