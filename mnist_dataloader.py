import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# 数据预处理
transform = transforms.ToTensor()

# 下载训练集
train_dataset = torchvision.datasets.MNIST(
    root="./data",          # 数据保存位置
    train=True,             # 训练集
    download=True,          # 自动下载
    transform=transform
)

# 下载测试集
test_dataset = torchvision.datasets.MNIST(
    root="./data",
    train=False,            # 测试集
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_dataset,
    batch_size=64,
    shuffle=False
)

print("训练集数量：", len(train_dataset))
print("测试集数量：", len(test_dataset))

for images, labels in train_loader:
    print("图像张量形状：", images.shape)
    print("标签张量形状：", labels.shape)
    print("标签：", labels)
    break  # 只查看第一个批次