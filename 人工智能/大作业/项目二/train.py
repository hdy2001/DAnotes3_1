import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from dataloader import MyDataset
from test import myTest
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# TODO: 目前的难题是：图片数量太少了，需要对图片数据集进行增多

# 参数设置
'''
lr0.0001适合resnet18和vgg19_bn，都在百分之八十上下
decay一开始为0最合适
'''
n_epochs = 30
learning_rate = 0.0001
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)

# 导入数据集
root = "./新冠辅助诊断数据集/"
"""
R_mean: 0.569971, G_mean: 0.569971, B_mean: 0.569971
R_std: 0.279340, G_std: 0.279340, B_std: 0.279340
"""

transform_train = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.RandomGrayscale(1),
    transforms.ToTensor(),
    transforms.Normalize((0.569971, 0.569971, 0.569971),
                         (0.279340, 0.279340, 0.279340))
])

transform_test = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.569971, 0.569971, 0.569971),
                         (0.279340, 0.279340, 0.279340))
])

train_data = MyDataset(txt=root + 'train.txt', transform=transform_train)
test_data = MyDataset(txt=root + 'test.txt', transform=transform_test)

#train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
train_loader = DataLoader(dataset=train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=10)

# 初始化网络
model = models.vgg19_bn(pretrained=True).to(device)
model.classifier._modules['6'] = nn.Sequential(nn.Linear(4096, 2)).to(device)

criterion = nn.CrossEntropyLoss()
# TODO: 优化adam
optimizer = optim.Adam(model.parameters(),
                       lr=learning_rate,
                       weight_decay=0.00001"""  """)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]


# 训练
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

            train_losses.append(loss.item())
            train_counter.append((batch_idx * 64) +
                                 ((epoch - 1) * len(train_loader.dataset)))


def test():
    model.eval()
    targets = torch.Tensor().to(device)
    pred = torch.Tensor().to(device)
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            targets = torch.cat((targets, target), 0)
            output = model(data)
            pred = torch.cat(
                (pred, output.data.max(1, keepdim=True)[1].reshape(-1)), 0)
    myTest(targets.cpu().numpy(), pred.cpu().numpy())


for epoch in range(1, n_epochs + 1):
    train(epoch)

test()
