import torch
import torchvision.models as models
from model import ConvNet
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from dataloader import MyDataset
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
"""
目前的难题是：图片数量太少了，需要对图片数据集进行增多
"""

# 参数设置
n_epochs = 3
batch_size_train = 60
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)

# 导入数据集
root = "./新冠辅助诊断数据集/"

transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = MyDataset(txt=root + 'train.txt', transform=transform)
test_data = MyDataset(txt=root + 'test.txt', transform=transform)

#train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
train_loader = DataLoader(dataset=train_data, batch_size=60, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=60)

# 初始化网络
model = models.resnet18(pretrained=True)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]


# 训练
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.zero_grad()
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
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# 卷积神经网络进行数字识别
for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()
