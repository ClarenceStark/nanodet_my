import torch
from torchvision import models
from torch import nn

class Net(nn.Module):
    def __init__(self, num_classes=48):
        super(Net, self).__init__()
        # 用预训练的 ResNet50 作为特征提取器
        self.backbone = models.resnet50(pretrained=True)
        
        # 移除 ResNet50 的最后一层（分类层），只保留到全局平均池化层之前的部分
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # 分类分支：一个全连接层用于分类
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
            nn.Flatten(),                  # 展平张量
            nn.Linear(2048, num_classes)   # 分类输出，num_classes表示分类类别数量
        )
        
        # 角点预测分支：一个全连接层用于回归 8 个角点坐标
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
            nn.Flatten(),                  # 展平张量
            nn.Linear(2048, 8)             # 输出 8 个数值，表示四个角点的坐标
        )

    def forward(self, x):
        # 通过 ResNet50 提取特征
        features = self.backbone(x)
        
        # 分类输出
        class_output = self.classifier(features)
        
        # 角点坐标输出
        points_output = self.regressor(features)
        
        return class_output, points_output  # 返回分类和角点坐标的输出

if __name__ == '__main__':
    net = Net(num_classes=48)  # 假设有6个分类类别
    x = torch.randn(1, 3, 640, 640)  # 输入一个 640x640 的图像
    class_output, points_output = net(x)
    print(f"分类输出的形状: {class_output.shape}")  # [1, num_classes]
    print(f"角点坐标输出的形状: {points_output.shape}")  # [1, 8]
