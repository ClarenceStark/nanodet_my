import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataLoader import CornerHeatmapDataset
from model import UNet

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_memory_usage():
    if torch.cuda.is_available():
        # allocated_memory = torch.cuda.memory_allocated() / (1024**2)  # 转换为 MB
        max_allocated_memory = torch.cuda.max_memory_allocated() / (1024**3)  # 最大分配的显存  GB
        # print(f"当前显存使用量: {allocated_memory:.2f} MB")
        print(f"最大显存使用量: {max_allocated_memory:.2f} GB")

# 超参数设置
num_epochs = 50
batch_size = 32
learning_rate = 1e-4
img_size = (256, 256)

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 数据集和数据加载器
train_dataset = CornerHeatmapDataset(txt_file='data_center.txt', transform=transform, img_size=img_size)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 实例化模型
model = UNet(n_channels=3, n_classes=4).to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()  # 可以尝试使用其他损失函数，如 Focal Loss
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练循环
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, heatmaps in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        images = images.to(device)
        heatmaps = heatmaps.to(device)

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, heatmaps)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # 打印损失信息
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")
    print_memory_usage()

# 保存模型
torch.save(model.state_dict(), 'corner_detection_unet.pth')
