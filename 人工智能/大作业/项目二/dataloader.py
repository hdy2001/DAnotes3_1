from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

root = "./新冠辅助诊断数据集/"

transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


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


train_data = MyDataset(txt=root + 'train.txt', transform=transform)
test_data = MyDataset(txt=root + 'test.txt', transform=transform)

#train_data 和test_data包含多有的训练与测试数据，调用DataLoader批量加载
train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=64)

# 输出测试
# train_features, train_labels = next(iter(test_loader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# print(train_features[0].shape)
# img = train_features[0].squeeze()
# print(img.shape)
# label = train_labels[0]
# print(f"Label: {label}")
# print(len(train_loader.dataset))