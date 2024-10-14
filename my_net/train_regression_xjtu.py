# train_regression.py

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm  # 进度条
from dataLoader import CornerDataset, CornerDatasetXJTU
from model import RegressionNet  # 导入回归模型

def print_memory_usage():
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated() / (1024**2)
        max_allocated_memory = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"当前显存使用量: {allocated_memory:.2f} MB")
        print(f"最大显存使用量: {max_allocated_memory:.2f} MB")

# 设置设备为 GPU 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数设置
num_epochs = 20
batch_size = 16
learning_rate = 1e-4

# 数据集和数据加载器
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = CornerDatasetXJTU(txt_file='data_center_xjtu.txt', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 实例化模型并移动到设备
model_regression = RegressionNet().to(device)

# 定义损失函数和优化器
criterion_points = nn.L1Loss()  # 回归损失
optimizer_regression = optim.Adam(model_regression.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(num_epochs):
    model_regression.train()
    running_loss_regression = 0.0

    for i, (images, _, corners) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Regression")):
        # 将数据移动到设备
        images = images.to(device)
        corners = corners.to(device)

        # 前向传播
        outputs = model_regression(images)

        # 计算损失
        loss_points = criterion_points(outputs, corners)

        # 反向传播和优化
        optimizer_regression.zero_grad()
        loss_points.backward()
        optimizer_regression.step()

        running_loss_regression += loss_points.item()

        # 打印损失信息
        if (i + 1) % 10 == 0:
            print(f"Regression - Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss_regression / (i + 1):.4f}")
            print_memory_usage()

    # 每个 epoch 结束后打印平均损失
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Regression Loss: {running_loss_regression / len(train_loader):.4f}")

# 保存模型
torch.save(model_regression.state_dict(), 'model_regression.pth')
