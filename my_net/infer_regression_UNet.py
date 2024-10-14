import argparse
import os
import cv2
import torch
from torchvision import transforms
from PIL import Image
from model import UNet  # 导入 U-Net 模型

# 图像预处理
def preprocess_image(image_path, device, img_size=(256, 256)):
    transform = transforms.Compose([
        transforms.Resize(img_size),  # 确保图像尺寸一致
        transforms.ToTensor(),
    ])
    image = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 将 numpy 图像转换为 PIL 图像
    pil_img = Image.fromarray(img_rgb)

    # 应用 transform
    image_tensor = transform(pil_img).unsqueeze(0)  # 增加 batch 维度
    return image_tensor.to(device), image  # 返回原始图像用于可视化

# 从热力图中提取角点坐标
def extract_keypoints_from_heatmap(heatmaps):
    keypoints = []
    for i in range(heatmaps.shape[0]):  # 对每个通道（即每个角点）进行处理
        heatmap = heatmaps[i]
        y, x = torch.argmax(heatmap).item() // heatmap.shape[1], torch.argmax(heatmap).item() % heatmap.shape[1]
        keypoints.append((x, y))  # (x, y) 是角点坐标
    return keypoints

# 角点回归推理（基于热力图的角点检测）
class RegressionPredictor(object):
    def __init__(self, model_path, device="cpu", img_size=(256, 256)):
        self.device = device
        self.img_size = img_size
        # 加载 U-Net 模型
        self.model = UNet(n_channels=3, n_classes=4).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def inference(self, image_path):
        image_tensor, raw_img = preprocess_image(image_path, self.device, img_size=self.img_size)

        with torch.no_grad():
            # 进行角点检测，输出热力图
            outputs = self.model(image_tensor).squeeze(0)  # (n_classes, height, width)
            keypoints = extract_keypoints_from_heatmap(outputs.cpu())  # 提取角点坐标
            print(f"Predicted Keypoints: {keypoints}")
            return keypoints, raw_img

    def visualize(self, raw_img, keypoints):
        # 可视化每个角点
        for (x, y) in keypoints:
            x = int(x * raw_img.shape[1] / self.img_size[0])  # 恢复到原始图像尺寸
            y = int(y * raw_img.shape[0] / self.img_size[1])
            cv2.circle(raw_img, (x, y), 5, (0, 0, 255), -1)
        return raw_img


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="角点回归模型文件路径")
    parser.add_argument("--path", help="图片路径")
    parser.add_argument("--save_result", action="store_true", help="是否保存推理结果")
    parser.add_argument("--output_dir", default="./output", help="保存推理结果的目录")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    predictor = RegressionPredictor(model_path=args.model, device="cuda" if torch.cuda.is_available() else "cpu")

    # 处理图片
    image_path = args.path
    keypoints, raw_img = predictor.inference(image_path)
    result_img = predictor.visualize(raw_img, keypoints)

    # 显示结果
    cv2.imshow("Regression Result", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 保存结果
    if args.save_result:
        save_path = os.path.join(args.output_dir, os.path.basename(image_path))
        cv2.imwrite(save_path, result_img)
        print(f"Saved result to {save_path}")


if __name__ == "__main__":
    main()

# python infer_regression.py --model path_to_unet_model.pth --path path_to_image.jpg --save_result
