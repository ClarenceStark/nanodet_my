import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np

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
        corners = torch.tensor(corners, dtype=torch.float32) / 640.0
        
        return image, class_label, corners


class CornerDatasetXJTU(Dataset):
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
                corners = list(map(float, parts[6:]))  # 角点坐标
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


class CornerHeatmapDataset(Dataset):
    def __init__(self, txt_file, transform=None, img_size=(256, 256), sigma=2):
        self.data_list = self._read_txt(txt_file)
        self.transform = transform
        self.img_size = img_size
        self.sigma = sigma  # 高斯核的标准差

    def _read_txt(self, txt_file):
        # 实现读取数据列表的函数
        # 返回列表，每个元素包含 (image_path, corners)
        data_list = []
        with open(txt_file, 'r') as f:
            for line in f:
                line = line.strip()
                # 假设每行格式为：image_path x1 y1 x2 y2 x3 y3 x4 y4
                items = line.split()
                image_path = items[0]
                corners = [float(x) for x in items[2:]]
                data_list.append((image_path, corners))
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        image_path, corners = self.data_list[idx]
        image = Image.open(image_path).convert('RGB')
        image = image.resize(self.img_size)

        # 转换角点坐标到新的尺寸
        orig_size = image.size  # (width, height)
        scale_x = self.img_size[0] / orig_size[0]
        scale_y = self.img_size[1] / orig_size[1]
        corners = np.array(corners).reshape(-1, 2)
        corners[:, 0] *= scale_x
        corners[:, 1] *= scale_y

        # 生成热力图
        heatmaps = self._generate_heatmaps(corners)

        if self.transform:
            image = self.transform(image)

        return image, heatmaps

    def _generate_heatmaps(self, corners):
        heatmaps = np.zeros((self.img_size[1], self.img_size[0], 4), dtype=np.float32)
        for i, (x, y) in enumerate(corners):
            heatmap = self._generate_gaussian_heatmap(self.img_size, x, y, self.sigma)
            heatmaps[:, :, i] = heatmap
        # 调整维度为 (channels, height, width)
        heatmaps = heatmaps.transpose(2, 0, 1)
        return torch.tensor(heatmaps, dtype=torch.float32)

    def _generate_gaussian_heatmap(self, img_size, x, y, sigma):
        # 创建一个空的热力图
        heatmap = np.zeros((img_size[1], img_size[0]), dtype=np.float32)
        # 定义高斯核的范围
        tmp_size = sigma * 3
        ul = [int(x - tmp_size), int(y - tmp_size)]
        br = [int(x + tmp_size + 1), int(y + tmp_size + 1)]

        # 检查边界
        if ul[0] >= img_size[0] or ul[1] >= img_size[1] or br[0] < 0 or br[1] < 0:
            return heatmap

        # 生成高斯核
        size = 2 * tmp_size + 1
        x0 = y0 = size // 2
        g = self._gaussian2D((size, size), sigma)

        # 计算有效的高斯核范围
        g_x = max(0, -ul[0]), min(br[0], img_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], img_size[1]) - ul[1]

        # 计算有效的图像范围
        img_x = max(0, ul[0]), min(br[0], img_size[0])
        img_y = max(0, ul[1]), min(br[1], img_size[1])

        heatmap[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        return heatmap

    def _gaussian2D(self, shape, sigma):
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m+1, -n:n+1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h