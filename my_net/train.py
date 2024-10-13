import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm  # 进度条
from dataLoader import CornerDataset  # 假设数据加载器位于 my_data_loader.py 文件中
from model import Net  # 假设模型定义在 my_model.py 文件中

# 设置设备为 GPU 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 超参数设置
num_epochs = 10
batch_size = 8
learning_rate = 1e-4
num_classes = 48

# 数据集和数据加载器
transform = transforms.Compose([
    transforms.Resize((640, 640)),  # 调整图像大小
    transforms.ToTensor(),          # 将图像转换为张量
])

train_dataset = CornerDataset(txt_file='data_center.txt', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 实例化模型并移动到设备（GPU/CPU）
model = Net(num_classes=num_classes).to(device)

# 定义损失函数
criterion_class = nn.CrossEntropyLoss()  # 分类损失
criterion_points = nn.MSELoss()  # 坐标回归损失

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0

    # 迭代加载数据
    for i, (images, class_labels, corners) in enumerate(tqdm(train_loader)):
        # 将数据移动到设备（GPU/CPU）
        images = images.to(device)
        class_labels = class_labels.to(device)
        corners = corners.to(device)

        # 前向传播
        class_output, points_output = model(images)

        # 计算损失
        loss_class = criterion_class(class_output, class_labels)
        loss_points = criterion_points(points_output, corners)
        loss = loss_class + loss_points  # 总损失是分类和回归损失的和

        # 反向传播和优化
        optimizer.zero_grad()  # 清除前一步的梯度
        loss.backward()  # 计算当前梯度
        optimizer.step()  # 更新模型参数

        running_loss += loss.item()

        # 打印损失信息
        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss / (i + 1):.4f},  Class Loss: {loss_class / (i + 1):.4f},  Point Loss: {loss_points / (i + 1):.4f}")

    # 每个 epoch 结束后打印平均损失
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Loss: {running_loss / len(train_loader):.4f}")

# 保存模型
torch.save(model.state_dict(), 'model.pth')
