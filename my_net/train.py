import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm  # 进度条
from dataLoader import CornerDataset  # 假设数据加载器位于 my_data_loader.py 文件中
#from model import Net  # 假设模型定义在 my_model.py 文件中
from model import ClassificationNet, RegressionNet


def print_memory_usage():
    if torch.cuda.is_available():
        allocated_memory = torch.cuda.memory_allocated() / (1024**2)  # 转换为 MB
        max_allocated_memory = torch.cuda.max_memory_allocated() / (1024**2)  # 最大分配的显存
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

# 实例化模型并移动到设备（GPU/CPU）
#model = Net(num_classes=num_classes).to(device)



# 实例化模型并移动到设备（GPU/CPU）
model_classification = ClassificationNet(num_classes=num_classes).to(device)
model_regression = RegressionNet().to(device)


# 定义损失函数
criterion_class = nn.CrossEntropyLoss()  # 分类损失
criterion_points = nn.MSELoss()  # 坐标回归损失

# 定义优化器
#optimizer = optim.Adam(model.parameters(), lr=learning_rate)

optimizer_classification = optim.Adam(model_classification.parameters(), lr=learning_rate)
optimizer_regression = optim.Adam(model_regression.parameters(), lr=learning_rate)


# 训练循环
for epoch in range(num_epochs):
    model_classification.train()  # 设置模型为训练模式
    running_loss_classification = 0.0

    # 迭代加载数据
    for i, (images, class_labels, _) in enumerate(tqdm(train_loader)):
        # 将数据移动到设备（GPU/CPU）
        images = images.to(device)
        class_labels = class_labels.to(device)
        # corners = corners.to(device)

        # 前向传播
        class_output = model_classification(images)

        # 计算损失
        loss_class = criterion_class(class_output, class_labels)
        # loss_points = criterion_points(points_output, corners)
        #print(points_output.shape)
        # loss = loss_class + 100 * loss_points  # 总损失是分类和回归损失的和

        # 反向传播和优化
        optimizer_classification.zero_grad()  # 清除前一步的梯度
        loss_class.backward()  # 计算当前梯度
        optimizer_classification.step()  # 更新模型参数

        running_loss_classification += loss_class.item()

        if (i + 1) % 10 == 0:
            print(f"Classification - Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {running_loss_classification / (i + 1):.4f}")
            print_memory_usage()
        
        model_regression.train()
    running_loss_regression = 0.0

    for i, (images, _, corners) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Regression")):
        # 数据准备
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
        
        
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Classification Loss: {running_loss_classification / len(train_loader):.4f}")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Average Regression Loss: {running_loss_regression / len(train_loader):.4f}")


    # 每个 epoch 结束后打印平均损失

# 保存模型
# 保存模型
torch.save(model_classification.state_dict(), 'model_classification.pth')
torch.save(model_regression.state_dict(), 'model_regression.pth')
