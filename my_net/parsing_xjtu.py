from PIL import Image
import os

# 读取图片，不做缩放
def open_image(path):
    img = Image.open(path)
    return img  # 返回原始图像，因为它们已经是640x640

def make_data_center_txt(txt_dir, image_dir, output_txt='data_center_xjtu.txt'):
    with open(output_txt, 'a') as f:
        f.truncate(0)  # 清空文件内容
        txt_names = os.listdir(txt_dir)  # 获取 TXT 文件列表
        for txt_file in txt_names:
            txt_path = os.path.join(txt_dir, txt_file)
            if not txt_file.endswith('.txt'):
                continue  # 只处理txt文件
            
            with open(txt_path, 'r') as in_file:
                lines = in_file.readlines()
                for line in lines:
                    line_data = line.strip().split()
                    
                    # 确保数据格式正确 (class x y w h x1 y1 x2 y2 x3 y3 x4 y4)
                    if len(line_data) != 13:
                        print(f"Skipping invalid line in {txt_file}: {line}")
                        continue
                    
                    class_name = line_data[0]  # 类别标签
                    bbox = line_data[1:5]     # 中心点 (x, y) 和宽高 (w, h)
                    points = line_data[5:]     # 四个角点坐标
                    
                    # 构建输出数据字符串
                    data_str = txt_path + ' ' + class_name + ' ' + ' '.join(bbox) + ' ' + ' '.join(points)
                    f.write(data_str + '\n')  # 写入输出文件




if __name__ == '__main__':
    txt_dir = 'dataset_xjtu/labels'  # 标注 TXT 文件目录
    image_dir = 'dataset_xjtu/images'  # 图像文件目录
    make_data_center_txt(txt_dir, image_dir)
