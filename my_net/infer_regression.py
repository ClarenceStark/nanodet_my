import argparse
import os
import cv2
import torch
from torchvision import transforms
from PIL import Image
from model import RegressionNet  # 导入角点回归模型

# 图像预处理
def preprocess_image(image_path, device):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # 确保图像尺寸一致
        transforms.ToTensor(),
    ])
    image = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 将 numpy 图像转换为 PIL 图像
    pil_img = Image.fromarray(img_rgb)

    # 应用 transform
    image_tensor = transform(pil_img).unsqueeze(0)  # 增加 batch 维度
    return image_tensor.to(device), image  # 返回原始图像用于可视化

# 角点回归推理
class RegressionPredictor(object):
    def __init__(self, model_path, device="cpu"):
        self.device = device
        # 加载角点回归模型
        self.model = RegressionNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def inference(self, image_path):
        image_tensor, raw_img = preprocess_image(image_path, self.device)

        with torch.no_grad():
            # 进行角点检测
            points_output = self.model(image_tensor)
            points_output = points_output.squeeze(0).cpu().numpy()  # 获取角点坐标
            points_output = points_output * 640.0  # 恢复到原始尺寸
            print(f"Points Output: {points_output}")
            return points_output, raw_img

    def visualize(self, raw_img, points_output):
        for i in range(0, len(points_output), 2):
            x, y = int(points_output[i]), int(points_output[i + 1])
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
    points_output, raw_img = predictor.inference(image_path)
    result_img = predictor.visualize(raw_img, points_output)

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




# python infer_regression.py --model path_to_regression_model.pth --path path_to_image.jpg --save_result
