import argparse
import os
import cv2
import torch
from torchvision import transforms
from PIL import Image
from model import ClassificationNet, RegressionNet  # 导入分类和回归模型

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

# 加载类别映射
label_map = {
    0: 'BR4', 1: 'BPg', 2: 'BPb', 3: 'BB3', 4: 'BN3', 5: 'SP2', 6: 'BNg', 7: 'BN4',
    8: 'SNO', 9: 'BN1', 10: 'SR2', 11: 'SRb', 12: 'BP5', 13: 'BP1', 14: 'SNb', 15: 'BR1',
    16: 'BP3', 17: 'SB4', 18: 'SP5', 19: 'SN4', 20: 'BNb', 21: 'SP4', 22: 'BRg', 23: 'BBg',
    24: 'BR3', 25: 'SBO', 26: 'SPb', 27: 'BB1', 28: 'BP4', 29: 'BRb', 30: 'SR4', 31: 'BR5',
    32: 'SN5', 33: 'SP3', 34: 'BB4', 35: 'SB3', 36: 'BB5', 37: 'BN5', 38: 'SPO', 39: 'SB5',
    40: 'SN2', 41: 'SBb', 42: 'SRO', 43: 'SN3', 44: 'SB2', 45: 'SR3', 46: 'BBb', 47: 'SR5'
}

# 推理函数
class Predictor(object):
    def __init__(self, class_model_path, points_model_path, device="cpu"):
        self.device = device

        # 加载分类模型
        self.classification_model = ClassificationNet(num_classes=48).to(self.device)
        self.classification_model.load_state_dict(torch.load(class_model_path, map_location=device))
        self.classification_model.eval()

        # 加载角点回归模型
        self.points_model = RegressionNet().to(self.device)
        self.points_model.load_state_dict(torch.load(points_model_path, map_location=device))
        self.points_model.eval()

    def inference(self, image_path):
        image_tensor, raw_img = preprocess_image(image_path, self.device)

        with torch.no_grad():
            # 进行分类预测
            class_output = self.classification_model(image_tensor)
            _, predicted_class = torch.max(class_output, 1)

            # 进行角点检测
            points_output = self.points_model(image_tensor)
            points_output = points_output.squeeze(0).cpu().numpy()  # 获取角点坐标
            points_output = points_output * 640.0  # 恢复到原始尺寸
            print(points_output)
            return predicted_class.item(), points_output, raw_img

    def visualize(self, raw_img, predicted_class, points_output):
        class_name = label_map.get(predicted_class, "Unknown")
        for i in range(0, len(points_output), 2):
            x, y = int(points_output[i]), int(points_output[i + 1])
            cv2.circle(raw_img, (x, y), 5, (0, 0, 255), -1)
        
        cv2.putText(raw_img, f"Class: {class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return raw_img


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_model", help="分类模型文件路径")
    parser.add_argument("--points_model", help="角点回归模型文件路径")
    parser.add_argument("--path", help="图片路径")
    parser.add_argument("--save_result", action="store_true", help="是否保存推理结果")
    parser.add_argument("--output_dir", default="./output", help="保存推理结果的目录")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    predictor = Predictor(class_model_path=args.class_model, points_model_path=args.points_model, 
                          device="cuda" if torch.cuda.is_available() else "cpu")

    # 处理图片
    image_path = args.path
    predicted_class, points_output, raw_img = predictor.inference(image_path)
    print(predicted_class)
    result_img = predictor.visualize(raw_img, predicted_class, points_output)

    # 显示结果
    cv2.imshow("Result", result_img)
    cv2.waitKey(0)  # 等待用户按下键盘上的任意键
    cv2.destroyAllWindows()

    # 保存结果
    if args.save_result:
        save_path = os.path.join(args.output_dir, os.path.basename(image_path))
        cv2.imwrite(save_path, result_img)
        print(f"Saved result to {save_path}")


if __name__ == "__main__":
    main()
