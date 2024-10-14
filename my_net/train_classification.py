# train_classification.py

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm  # 进度条
from dataLoader import CornerDataset
from model import ClassificationNet  # 导入分类模型

def print_memory_usage():
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated() / (1024**2)  # 转换为 MB
        max_allocated_memory = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"当前显存使用量: {allocated_memory:.2f} MB")
        print(f"最大显存使用量: {max_allocated_memory:.2f} MB")

# 设置设备为 GPU 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数设置
num_epochs = 20
batch_size = 16
learning_rate = 1e-4
num_classes = 48

# 数据集和数据加载器
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # 调整图像大小
    transforms.ToTensor(),          # 将图像转换为张量
])

train_dataset = CornerDataset(txt_file='data_center.txt', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 实例化模型并移动到设备
model_classification = ClassificationNet(num_classes=num_classes).to(device)

# 定义损失函数和优化器
criterion_class = nn.CrossEntropyLoss()  # 分类损失
optimizer_classification = optim.Adam(model_classification.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(num_epochs):
    model_classification.train()
    running_loss_classification = 0.0

    for i, (images, class_labels, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Classification")):
        # 将数据移动到设备
        images = images.to(device)
        class_labels = class_labels.to(device)

        # 前向传播
        outputs = model_classification(images)

        # 计算损失
        loss_class = criterion_class(outputs, class_labels)

        # 反向传播和优化
        optimizer_classification.zero_grad()
        loss_class.backward()
        optimizer_classification.step()

        running_loss_classification += loss_class.item()

        # 打印损失信息
        if (i + 1) % 10 == 0:
            print(f"Classification - Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss_classification / (i + 1):.4f}")
            print_memory_usage()

    # 每个 epoch 结束后打印平均损失
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Classification Loss: {running_loss_classification / len(train_loader):.4f}")

# 保存模型
torch.save(model_classification.state_dict(), 'model_classification.pth')
