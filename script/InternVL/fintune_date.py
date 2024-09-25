import os
import json
from PIL import Image
import shutil

DEFAULT_PROMPT_TEMPLATE_DISEASE = """
你是一名公路巡检养护校验专家，检查图片中的黄色框是否存在病害，仅关机动车注道路病害，路面基础建筑以及人物、植被请忽略。

规则：
1. 种植的树木下方的坑非坑槽，为无病害。
2. 坑槽：可能呈现出圆形、椭圆形或不规则形状，周围的边缘可能会呈现破碎状或裂纹状态，不要忽略小坑的存在。
3. 道路上的垃圾、塑料袋、烟盒、碎纸、毛线等异常物体均为抛洒物类型。
4. 坑槽和抛洒物要谨慎区分，坑槽为凹陷状态，抛洒物为凸起状态。
5. 偏垂直装裂隙为纵向裂隙，偏水平装裂隙为横向裂隙。

要求：
1. 如果存在病害，返回"check"=true，示例：{"check" true, "type": "病害类型"}
2. 如果不存在病害，返回"check"=false, 示例：{"check": false, "type": "无病害"}
3. 病害类型type包括：["纵向裂隙", "横向裂隙", "网状裂隙", "块状裂缝", "井盖破损", "抛洒物", "积水", "坑槽"]
4. 如无病害，则type="无病害"，示例：{"check": false, "type": "无病害"}

结果以JSON格式返回，不需要其他任何内容！
"""


def generate_jsonl(input_dir, output_dir, image_prefix):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 清空输出目录
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

    # 获取输入目录的名字用于输出文件名
    input_dir_name = os.path.basename(os.path.normpath(input_dir))
    output_file_path = os.path.join(output_dir, f"{input_dir_name}_output.jsonl")

    # 定义对话模板 需要替换replace中的值
    conversations_template = [
        {"from": "human", "value": f"<image>\n{DEFAULT_PROMPT_TEMPLATE_DISEASE}"},
        {"from": "gpt",
         "value": "{replace}"}
    ]

    # 初始化ID计数器
    id_counter = 0

    # 遍历目录中的所有文件
    with open(output_file_path, 'w', encoding='utf-8') as jsonl_file:
        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # 构建图片路径
                img_path = os.path.join(input_dir, filename)

                # 打开图片以获取尺寸
                with Image.open(img_path) as img:
                    width, height = img.size
                # 正则移除文件后缀
                tmp_filename = filename.replace(".png", "").replace(".jpg", "").replace(".jpeg", "")
                # 读取同filename下_check.txt文件
                with open(os.path.join(input_dir, tmp_filename + "_check.txt"), 'r') as f1:
                    template_value = f1.readline().strip()
                conversations_template[1]["value"] = template_value

                # 构造JSON对象
                entry = {
                    "id": id_counter,
                    "image": os.path.join(image_prefix, filename),
                    "conversations": conversations_template,
                    "width": width,
                    "height": height
                }

                # 将JSON对象转为字符串并写入文件
                jsonl_file.write(json.dumps(entry, ensure_ascii=False) + '\n')
                id_counter += 1


if __name__ == "__main__":
    input_directory = './data/images'
    output_directory = './data/annotation'
    image_prefix = './data/images/rode_disease/'

    # 生成文件
    generate_jsonl(input_directory, output_directory, image_prefix)

    # 打印结果
    with open(output_directory + "/images_output.jsonl", 'r') as f:
        raw_data = f.readlines()
    raw_data = [print(json.loads(line)) for line in raw_data[0:10]]


