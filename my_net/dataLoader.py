import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# 自定义数据集类
class CornerDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        self.data = []
        self.transform = transform
        
        # 读取 datacenter.txt 文件
        with open(txt_file, 'r') as f:
            for line in f:
                # 每一行分为图像路径、分类标签和角点坐标
                parts = line.strip().split()
                image_path = parts[0]
                class_label = int(parts[1])  # 分类标签
                corners = list(map(float, parts[2:]))  # 角点坐标
                self.data.append((image_path, class_label, corners))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path, class_label, corners = self.data[idx]
        
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 图像变换（如果有）
        if self.transform:
            image = self.transform(image)
        
        # 将分类标签转换为 Tensor
        class_label = torch.tensor(class_label, dtype=torch.long)
        
        # 将角点坐标转换为 Tensor
        corners = torch.tensor(corners, dtype=torch.float32)
        
        return image, class_label, corners

# 示例用法
if __name__ == '__main__':
    # 定义图像预处理，比如将图像转换为张量
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # 调整图像大小
        transforms.ToTensor(),          # 将图像转换为张量
    ])

    # 加载数据集
    dataset = CornerDataset(txt_file='data_center.txt', transform=transform)
    
    # 使用 DataLoader 批量加载数据
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 测试加载数据
    for images, class_labels, corners in dataloader:
        print(f"Images batch shape: {images.shape}")
        print(f"Class labels batch shape: {class_labels.shape}")
        print(f"Corners batch shape: {corners.shape}")
        break  # 只测试一批数据
