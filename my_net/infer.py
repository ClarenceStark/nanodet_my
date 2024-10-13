import argparse
import os
import time
import cv2
import torch
from torchvision import transforms
from model import Net  # 导入自定义模型
from dataloader import image_ext  # 用于处理图片扩展名
import numpy as np

# 图像预处理
def preprocess_image(image_path, device):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # 确保图像尺寸一致
        transforms.ToTensor(),
    ])
    image = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_tensor = transform(img_rgb).unsqueeze(0)  # 增加 batch 维度
    return image_tensor.to(device), image  # 返回原始图像用于可视化

# 加载类别映射（可选）
label_map = {
    0: 'BR4', 1: 'BPg', 2: 'BPb', 3: 'BB3', 4: 'BN3', 5: 'SP2', 6: 'BNg', 7: 'BN4', 
    # ...
    47: 'SR5'
}

# 推理函数
class Predictor(object):
    def __init__(self, model_path, device="cpu"):
        self.device = device
        self.model = Net(num_classes=48).to(self.device)  # 实例化模型
        self.model.load_state_dict(torch.load(model_path, map_location=device))  # 加载模型权重
        self.model.eval()

    def inference(self, image_path):
        image_tensor, raw_img = preprocess_image(image_path, self.device)
        with torch.no_grad():
            class_output, points_output = self.model(image_tensor)
            _, predicted_class = torch.max(class_output, 1)
            points_output = points_output.squeeze(0).cpu().numpy()  # 获取角点坐标
            return predicted_class.item(), points_output, raw_img

    def visualize(self, raw_img, predicted_class, points_output, save_path=None):
        class_name = label_map.get(predicted_class, "Unknown")
        for i in range(0, len(points_output), 2):
            x, y = int(points_output[i]), int(points_output[i+1])
            cv2.circle(raw_img, (x, y), 5, (0, 255, 0), -1)
        
        cv2.putText(raw_img, f"Class: {class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        if save_path:
            cv2.imwrite(save_path, raw_img)
        return raw_img


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="模型文件路径")
    parser.add_argument("--path", help="图片或视频路径")
    parser.add_argument("--save_result", action="store_true", help="是否保存推理结果")
    parser.add_argument("--output_dir", default="./output", help="保存推理结果的目录")
    return parser.parse_args()


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in image_ext:
                image_names.append(apath)
    return image_names


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    predictor = Predictor(model_path=args.model, device="cuda" if torch.cuda.is_available() else "cpu")

    # 处理图片
    if os.path.isdir(args.path):
        image_list = get_image_list(args.path)
    else:
        image_list = [args.path]
    
    for image_path in image_list:
        predicted_class, points_output, raw_img = predictor.inference(image_path)
        result_img = predictor.visualize(raw_img, predicted_class, points_output)

        if args.save_result:
            save_path = os.path.join(args.output_dir, os.path.basename(image_path))
            cv2.imwrite(save_path, result_img)
            print(f"Saved result to {save_path}")

        cv2.imshow("Result", result_img)
        if cv2.waitKey(0) & 0xFF == 27:  # 按 'Esc' 退出
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
