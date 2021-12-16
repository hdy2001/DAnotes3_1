# 课程项目二—新冠肺炎的辅助诊断

<center>何东阳 2019011462 自96</center>

## 1 项目要求

请你设计深度学习算法，来实现新冠/非新冠 CT 影像的二分类。请对给定的数据集进行 合理的数据划分，设计合适的模型评价方法和指标，对所构建的深度学习算法进行性能评估。

## 2 项目建模

### 2.1 数据集处理

本次实验中我将给定的COVID影像标注为0，将给定的non-COVID影像标注为1，所有影像按照3：7的比例手动分为测试集和训练集，使用txt文件批量记录图片路径读入`dataloader`中。将照片批量导入txt文件的程序如下：

```python
import os

# path表示路径
path = "./新冠辅助诊断数据集/COVID"
# 返回path下所有文件构成的一个list列表
filelist = os.listdir(path)
# 遍历输出每一个文件的名字和类型
file = open('./新冠辅助诊断数据集/train2.txt', 'w+')
for item in filelist:
    # 输出指定后缀类型的文件
    # if(item.endswith('.jpg')):
    file.write('./新冠辅助诊断数据集/COVID/' + item + ' 1' + '\n')
file.close()
```

为了能够使用`pytorch`进行训练，需要设计自己的`dataloader`，结构如下：

```python
def default_loader(path):
    return Image.open(path).convert('RGB')


class MyDataset(Dataset):
    # 构造函数带有默认参数
    def __init__(self,
                 txt,
                 transform=None,
                 target_transform=None,
                 loader=default_loader):
        fh = open(txt, 'r', encoding='UTF-8')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))  #imgs中包含有图像路径和标签
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)
```

训练前通过`dataloader`导入自己的数据，导入前使用`pytorch`的`transformer`对图片进行预处理：

```python
root = "./新冠辅助诊断数据集/"
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
```

每一张图片都被预处理成了`Tensor`的数据结构，是一种可以在GPU上加速运算的矩阵结构，在训练时可以进行被网络进行多种矩阵运算。

### 2.2 神经网络设计

观察图片可以发现，COIVD和non-COVID的局部区别非常明显，因此可以采用卷积神经网络才提取特征训练。本次作业我使用的是`vgg19`，这是一种非常深的卷积神经网络结构，适合本题对CT影像分类的需求。

### 2.3 开始训练

计算`loss`的函数我使用的是交叉熵损失函数，调用的`pytorch`的接口`nn.CrossEntropyLoss()`，优化器使用的是优化效果比较好的`optim.Adam`，整个训练过程如下

```python
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
```

## 3 性能评估

由于本题是二分类，因此我采用课上提到过的评估方式进行评估。

## 4 影响因素

### 4.1 超参数



### 4.2 模型结构



### 4.3 预处理



## 5 选做



## 6 出现的问题与总结

1. 一开始的时候始终没有训练效果，后来发现是因为我将`optimizer.zero_grad()`写在了`loss.backward()`和`optimizer.step()`之间
2. 考虑使用数据增强，但我发现没有什么用，反而使得效果变得更不好了