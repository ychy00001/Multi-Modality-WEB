from utils.draw_box import draw_results_with_font, draw_results
from utils.file_utils import clear_directory
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

disease_type_map = {
    0: "横向裂缝",
    1: "纵向裂缝",
    2: "块状裂缝",
    3: "龟裂",
    4: "坑槽",
    5: "修补网状裂缝",
    6: "修补裂缝",
    7: "修补坑槽",
    8: "抛洒物",
    9: "积水",
    10: "井盖",
    11: "横拼接缝",
    12: "纵拼接缝",
    13: "矩形井盖",
    14: "填充的取芯孔",
    15: "未填充的取芯孔",
    16: "井盖破损",
    17: "矩形井盖破损",
    18: "井盖缺失",
    19: "矩形井盖缺失"
}
disease_ids = [0, 1, 2, 3, 4, 8, 9, 16, 18, 19]


def draw_box(file_dir, save_dir):
    process_count = 0
    draw_box_count = 0
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        clear_directory(save_dir)

    for file in os.listdir(file_dir):
        if file.endswith(".jpg"):
            image_file_name = file
            text_file_name = os.path.splitext(file)[0] + ".txt"

            # 构建完整的路径
            img_path = os.path.join(file_dir, image_file_name)
            txt_path = os.path.join(file_dir, text_file_name)

            # 读取图片
            img = Image.open(img_path).convert('RGB')
            img_arr = np.array(img)
            width, height = img.size

            # 判断文件不存在则跳过
            if not os.path.exists(txt_path):
                continue

            # 读取对应text文件
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            # 解析每一行的数据信息[disease_type score x_center y_center w h]，构建box坐标
            det_results = []
            for line in lines:
                data = line.strip().split()
                # 有些情况没有score
                # disease_type, score, x_center, y_center, w, h = map(float, data)
                disease_type, x_center, y_center, w, h = map(float, data)
                # 非路面异常不进行处理
                expansion_size = 20
                if disease_type in disease_ids:
                    min_x = int((x_center - w / 2) * width) - expansion_size if int((x_center - w / 2) * width) - expansion_size > 0 else 0
                    min_y = int((y_center - h / 2) * height) - expansion_size if int((y_center - h / 2) * height) - expansion_size > 0 else 0
                    max_x = int((x_center + w / 2) * width) + expansion_size if int((x_center + w / 2) * width) + expansion_size < width else width
                    max_y = int((y_center + h / 2) * height) + expansion_size if int((y_center + h / 2) * height) + expansion_size < height else height
                    # det_results.append([min_x, min_y, max_x, max_y, score, disease_type])
                    det_results.append([min_x, min_y, max_x, max_y, 0, disease_type])
            # 无标记则跳过当前图片
            if len(det_results) == 0:
                continue
            # 调用draw_results_with_font函数画框
            # drawed_img_arr = draw_results_with_font(img_arr, np.array(det_results), disease_type_map, threshold=0.05)
            drawed_img_arr = draw_results(img_arr, np.array(det_results), disease_type_map, threshold=0.05)

            # 输出图片到指定目录
            output_path = os.path.join(save_dir, image_file_name)
            Image.fromarray(drawed_img_arr).save(output_path)
            process_count = process_count + 1
            draw_box_count = draw_box_count + len(lines)
    print(f"处理图片:{process_count}张, 画框: {draw_box_count}个")


if __name__ == '__main__':
    IMAGE_DIR = "/Users/rain/Downloads/origin_images.tar/images"
    SAVE_DIR = "/Users/rain/Downloads/origin_images.tar/images_with_box"
    draw_box(IMAGE_DIR, SAVE_DIR)
