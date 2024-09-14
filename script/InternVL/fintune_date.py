import os
import json
from PIL import Image

DEFAULT_PROMPT_TEMPLATE_DISEASE = """
你是一名公路巡检养护校验专家，检查图片是否存在病害。
任务要求：
1. 如果存在病害，返回"check"=true，示例：{"check" true, "type": "病害类型""}
2. 如果不存在病害，返回"check"=false, 示例：{"check": false, "type": "无病害"}
3. 病害类型type包括：["纵向裂隙", "横向裂隙", "网状裂隙", "块状裂缝", "井盖破损", "井盖缺失", "抛洒物", "积水", "坑槽", "龟裂"]
4. 如无病害，则type="无病害"，示例：{"check": false, "type": "无病害"}
5. 返回结果必须严格按照JSON格式
6. 仅返回JSON结果，不需要额外内容
"""


def generate_jsonl(input_dir, output_dir, image_prefix):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取输入目录的名字用于输出文件名
    input_dir_name = os.path.basename(os.path.normpath(input_dir))
    output_file_path = os.path.join(output_dir, f"{input_dir_name}_output.jsonl")

    # 定义对话模板
    conversations_template = [
        {"from": "human", "value": f"<image>\n{DEFAULT_PROMPT_TEMPLATE_DISEASE}"},
        {"from": "gpt",
         "value": "{\"check\": true, \"type\": \"龟裂\"}"}
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

    # generate_jsonl(input_directory, output_directory, image_prefix)

    with open(output_directory + "/images_output.jsonl", 'r') as f:
        raw_data = f.readlines()
    raw_data = [print(json.loads(line)) for line in raw_data]


