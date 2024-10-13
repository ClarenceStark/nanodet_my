from PIL import Image
import os
from xml.etree import ElementTree as ET

# 读取图片，不做缩放
def open_image(path):
    img = Image.open(path)
    return img  # 返回原始图像，因为它们已经是640x640

def make_data_center_txt(xml_dir, image_dir, output_txt='data_center.txt'):
    with open(output_txt, 'a') as f:
        f.truncate(0)  # 清空文件内容
        xml_names = os.listdir(xml_dir)  # 获取 XML 文件列表
        for xml in xml_names:
            xml_path = os.path.join(xml_dir, xml)
            in_file = open(xml_path)
            tree = ET.parse(in_file)  # 解析 XML 文件
            root = tree.getroot()
            image_filename = root.find('filename').text  # 获取图像文件名
            
            # 获取分类标签（name属性）
            class_name = root.find('object/name').text
            
            # 获取角点标注数据
            polygon = root.find('object/gt_poly')
            data = []
            data_str = ''
            print(f"Processing {xml}...")
            
            # 遍历四个角点的坐标并拼接为字符串
            for coord in ['x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4']:
                point = polygon.find(coord).text
                data_str += ' ' + point  # 拼接每个坐标值
            
            # 拼接图像路径、分类标签和角点数据
            image_path = os.path.join(image_dir, image_filename)
            data_str = image_path + ' ' + class_name + data_str
            f.write(data_str + '\n')  # 将处理好的数据写入文件
            
if __name__ == '__main__':
    xml_dir = 'dataset_max/Annotations'  # 标注 XML 文件目录
    image_dir = 'dataset_max/JPEGImages'  # 图像文件目录
    make_data_center_txt(xml_dir, image_dir)
