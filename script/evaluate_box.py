import time

from utils.draw_box import draw_results_with_font, draw_results
from utils.file_utils import clear_directory
import numpy as np
from service.llm_service import predict, get_response, add_text_msg, add_file_msg
from PIL import Image, ImageDraw, ImageFont
import os
import json

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
DEFAULT_PROMPT_TEMPLATE = """
你是一名公路巡检养护校验专家。判断输入的病害描述是否正确，并返回结果。
输入病害类型：{user_input}
任务要求：
1. 如果输入病害描述正确，返回"check"=true，示例：{"check" true, "type": "病害类型""}
2. 如果输入病害描述不正确，返回"check"=false, 示例：{"check": false, "type": "病害类型"}
3. 病害类型type包括：["无病害", "纵向裂隙", "横向裂隙", "网状裂隙", "块状裂缝", "井盖破损", "井盖缺失", "抛洒物", "积水", "坑槽", "龟裂"]
4. 一定不要受输入病害的影响进行判断
5. 如无明显病害，则type="无病害"，示例：{"check": "检测结果", "type": "无病害"}
6. 返回结果必须严格按照JSON格式
7. 仅返回JSON结果，不需要额外内容
8. type为检测病害类型，如果检测病害类型与输入病害类型不一致，请输出检测病害类型

"""

DEFAULT_PROMPT_TEMPLATE_DISEASE = """
你是一名公路巡检养护校验专家，检查图片中的黄色框是否存在病害，仅关机动车注道路病害，路面基础建筑以及人物、植被请忽略。

规则：
1. 种植的树木下方的坑非坑槽，为无病害。
2. 坑槽：可能呈现出圆形、椭圆形或不规则形状，周围的边缘可能会呈现破碎状或裂纹状态，不要忽略小坑的存在。
3. 道路上的垃圾、塑料袋、烟盒、碎纸、毛线等异常物体均为抛洒物类型。
4. 坑槽和抛洒物要谨慎区分，坑槽为凹陷状态，抛洒物为凸起状态。
5. 偏垂直装裂隙为纵向裂隙，偏水平装裂隙为横向裂隙。

要求：
1. 如果存在病害，返回"check"=true，示例：{"check" true, "type": "病害类型""}
2. 如果不存在病害，返回"check"=false, 示例：{"check": false, "type": "无病害"}
3. 病害类型type包括：["纵向裂隙", "横向裂隙", "网状裂隙", "块状裂缝", "井盖破损", "井盖缺失", "抛洒物", "积水", "坑槽", "龟裂"]
4. 如无病害，则type="无病害"，示例：{"check": false, "type": "无病害"}

结果以JSON格式返回，不需要其他任何内容！
"""


def check_box_with_check_file(file_dir, save_dir, match_type=0):
    """
    校验目标监测图片病害是否正确
    :param file_dir:输入图片目录
    :param save_dir:输出图片目录
    :param match_type:匹配类型， 0: 完全匹配，1: 仅匹配是否有病害
    :return:
    """
    process_count = 0
    success_count = 0
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        clear_directory(save_dir)

    for file in os.listdir(file_dir):
        if file.endswith(".jpg"):
            print(f"正在处理第{process_count}个文件：{file}")
            image_file_name = file
            text_file_name = os.path.splitext(file)[0] + ".txt"

            # 构建完整的路径
            img_path = os.path.join(file_dir, image_file_name)
            txt_path = os.path.join(file_dir, text_file_name)

            # 判断文件不存在则跳过
            if not os.path.exists(txt_path):
                continue
            # 读取对应text文件
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            # 解析每一行的数据信息[disease_type score x_center y_center w h]，构建box坐标
            disease_names = []
            for line in lines:
                data = line.strip().split()
                disease_type, score, x_center, y_center, w, h = map(float, data)
                # 非路面异常不进行处理
                if disease_type in disease_ids:
                    disease_names.append(disease_type_map[disease_type])

            # 请求llm检测
            model_info = {
                "modelName": "InternVL-26B",
                "query": [],
                "history": [],
                "lastQuery": [],
            }
            parameter_info = {
                "prompt_template": DEFAULT_PROMPT_TEMPLATE,
                "temperature": 0.9,
                "top_k": 40,
                "top_p": 1
            }
            with open(img_path, 'rb') as f:
                model_info = add_file_msg(model_info, f)
            # 将病害名称拼接成字符串作为入参
            add_text_msg(model_info, ','.join(disease_names))
            model_info = predict(model_info, parameter_info)
            llm_response = get_response(model_info)
            # 截取已{开始，}结束的内容为json内容，获取llm_response中json数据
            start_index = llm_response.find('{')
            end_index = llm_response.find('}')
            if start_index == -1 or end_index == -1:
                continue
            json_str = llm_response[start_index: end_index + 1]
            try:
                json_data = json.loads(json_str)
            except json.decoder.JSONDecodeError:
                print(f"JSON解析错误：{json_str}")
                # llm输出的json_data结果保存至save_dir的_result.txt文件中
                with open(os.path.join(save_dir, image_file_name.replace(".jpg", "_result.txt")), 'w') as f:
                    f.write(json.dumps({"check": "JSON解析错误", "type": "JSON解析错误"}, ensure_ascii=False))
                continue
            # json读取校验文件内容
            verify_file_name = os.path.splitext(file)[0] + "_check.txt"
            verify_file_path = os.path.join(file_dir, verify_file_name)
            # 读取文件的json内容
            with open(verify_file_path, 'r') as f:
                verify_json = json.load(f)

            output_file = image_file_name.replace(".jpg", "_result_fail.txt")
            # 比较病害名称
            if match_type == 0:
                if json_data["check"] == verify_json["check"] and json_data["type"] == verify_json["type"]:
                    success_count += 1
                    output_file = image_file_name.replace(".jpg", "_result_success.txt")
            elif match_type == 1:
                if json_data["check"] == verify_json["check"]:
                    success_count += 1
                    output_file = image_file_name.replace(".jpg", "_result_success.txt")
            process_count += 1

            # llm输出的json_data结果保存至save_dir的_result.txt文件中
            with open(os.path.join(save_dir, output_file), 'w') as f:
                f.write(json.dumps(json_data, ensure_ascii=False))

    # 打印信息 准确率保留两位小数
    print("检测完成，共处理图片{}张，检测成功{}张，准确率：{:.2f}%".format(process_count, success_count,
                                                                       success_count / process_count * 100))


def check_box(file_dir, save_dir, match_type=0):
    """
    校验目标监测图片病害是否正确
    :param file_dir:输入图片目录
    :param save_dir:输出图片目录
    :param match_type:匹配类型， 0: 完全匹配，1: 仅匹配是否有病害
    :return:
    """
    process_count = 0
    success_count = 0
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        clear_directory(save_dir)
    start_time = time.time()
    for file in os.listdir(file_dir):
        if file.endswith(".jpg"):
            print(f"正在处理第{process_count}个文件：{file}")
            image_file_name = file
            text_file_name = os.path.splitext(file)[0] + ".txt"

            # 构建完整的路径
            img_path = os.path.join(file_dir, image_file_name)
            txt_path = os.path.join(file_dir, text_file_name)

            # 判断文件不存在则跳过
            if not os.path.exists(txt_path):
                continue
            # 读取对应text文件
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            # 解析每一行的数据信息[disease_type score x_center y_center w h]，构建box坐标
            disease_names = []
            for line in lines:
                data = line.strip().split()
                disease_type, score, x_center, y_center, w, h = map(float, data)
                # 非路面异常不进行处理
                if disease_type in disease_ids:
                    disease_names.append(disease_type_map[disease_type])

            # 请求llm检测
            model_info = {
                "modelName": "InternVL-26B",
                "query": [],
                "history": [],
                "lastQuery": [],
            }
            parameter_info = {
                "prompt_template": DEFAULT_PROMPT_TEMPLATE_DISEASE,
                "temperature": 0.9,
                "top_k": 40,
                "top_p": 1
            }
            with open(img_path, 'rb') as f:
                model_info = add_file_msg(model_info, f)
            # 将病害名称拼接成字符串作为入参
            add_text_msg(model_info, "")
            model_info = predict(model_info, parameter_info)
            llm_response = get_response(model_info)
            # 截取已{开始，}结束的内容为json内容，获取llm_response中json数据
            start_index = llm_response.find('{')
            end_index = llm_response.find('}')
            if start_index == -1 or end_index == -1:
                continue
            json_str = llm_response[start_index: end_index + 1]
            try:
                json_data = json.loads(json_str)
            except json.decoder.JSONDecodeError:
                print(f"JSON解析错误：{json_str}")
                # llm输出的json_data结果保存至save_dir的_result.txt文件中
                with open(os.path.join(save_dir, image_file_name.replace(".jpg", "_result.txt")), 'w') as f:
                    f.write(json.dumps({"check": "JSON解析错误", "type": "JSON解析错误"}, ensure_ascii=False))
                continue
            # json读取校验文件内容
            verify_file_name = os.path.splitext(file)[0] + "_check.txt"
            verify_file_path = os.path.join(file_dir, verify_file_name)
            # 读取文件的json内容
            with open(verify_file_path, 'r') as f:
                check_json = json.load(f)

            verify_json = {
                'check': not check_json['type'] == "无病害",
                'type': check_json['type']
            }

            output_file = image_file_name.replace(".jpg", "_result_fail.txt")
            # 比较病害名称
            if match_type == 0:
                if json_data["check"] == verify_json["check"] and json_data["type"] == verify_json["type"]:
                    success_count += 1
                    output_file = image_file_name.replace(".jpg", "_result_success.txt")
            elif match_type == 1:
                if json_data["check"] == verify_json["check"]:
                    success_count += 1
                    output_file = image_file_name.replace(".jpg", "_result_success.txt")
            process_count += 1

            # llm输出的json_data结果保存至save_dir的_result.txt文件中
            with open(os.path.join(save_dir, output_file), 'w') as f:
                f.write(json.dumps(json_data, ensure_ascii=False))
    end_time = time.time()
    # 打印信息 准确率保留两位小数
    print("检测完成，共处理图片{}张，检测成功{}张，准确率：{:.2f}%, 耗时{}秒".format(process_count, success_count,
                                                                                 success_count / process_count * 100,
                                                                                 end_time - start_time))


def check_img_disease(file_dir, save_dir, match_type=0, check_model_name="InternVL2-8B"):
    """
    校验病害裁剪图片是否是病害
    :param file_dir:输入图片目录
    :param save_dir:输出图片目录
    :param match_type:匹配类型， 0: 完全匹配，1: 仅匹配是否有病害
    :param check_model_name: 使用哪个模型检测
    :return:
    """
    process_count = 0
    success_count = 0
    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    else:
        clear_directory(save_dir)
    start_time = time.time()
    for file in os.listdir(file_dir):
        if file.endswith(".jpg"):
            print(f"正在处理第{process_count}个文件：{file}")
            image_file_name = file

            # 构建完整的路径
            img_path = os.path.join(file_dir, image_file_name)

            # 请求llm检测
            model_info = {
                "modelName": check_model_name,
                "query": [],
                "history": [],
                "lastQuery": [],
            }
            parameter_info = {
                "prompt_template": DEFAULT_PROMPT_TEMPLATE_DISEASE,
                "temperature": 0.9,
                "top_k": 40,
                "top_p": 1
            }
            with open(img_path, 'rb') as f:
                model_info = add_file_msg(model_info, f)
            # 将病害名称拼接成字符串作为入参
            add_text_msg(model_info, "")
            model_info = predict(model_info, parameter_info)
            llm_response = get_response(model_info)
            # 截取已{开始，}结束的内容为json内容，获取llm_response中json数据
            start_index = llm_response.find('{')
            end_index = llm_response.find('}')
            if start_index == -1 or end_index == -1:
                continue
            json_str = llm_response[start_index: end_index + 1]
            try:
                json_data = json.loads(json_str)
            except json.decoder.JSONDecodeError:
                print(f"JSON解析错误：{json_str}")
                # llm输出的json_data结果保存至save_dir的_result.txt文件中
                with open(os.path.join(save_dir, image_file_name.replace(".jpg", "_result.txt")), 'w') as f:
                    f.write(json.dumps({"check": "JSON解析错误", "type": "JSON解析错误"}, ensure_ascii=False))
                continue

            # may_disease_type = disease_type_map[int(file_dir.split("_")[-1])]
            # print(f"当前图片病害类型：{may_disease_type}")
            verify_json = {
                'check': True,
                'type': "未知"
            }

            output_file = image_file_name.replace(".jpg", "_result_fail.txt")
            # 比较病害名称
            if match_type == 0:
                if json_data["check"] == verify_json["check"] and json_data["type"] == verify_json["type"]:
                    success_count += 1
                    output_file = image_file_name.replace(".jpg", "_result_success.txt")
            elif match_type == 1:
                if json_data["check"] == verify_json["check"]:
                    success_count += 1
                    output_file = image_file_name.replace(".jpg", "_result_success.txt")
            process_count += 1

            # llm输出的json_data结果保存至save_dir的_result.txt文件中
            with open(os.path.join(save_dir, output_file), 'w') as f:
                f.write(json.dumps(json_data, ensure_ascii=False))
    end_time = time.time()
    # 打印信息 准确率保留两位小数
    print("检测完成，共处理图片{}张，检测成功{}张，准确率：{:.2f}%, 耗时{}秒".format(process_count, success_count,
                                                                                 success_count / process_count * 100,
                                                                                 end_time - start_time))


if __name__ == '__main__':
    # 带框带文字
    # IMAGE_DIR = "/Users/rain/Downloads/fp_100_box_with_text_check"
    # 纯框无文字
    # IMAGE_DIR = "/Users/rain/Downloads/fp_100_box_check"
    # 无框纯图片
    # IMAGE_DIR = "/Users/rain/Downloads/fp_100_without_box"
    # 指定龟裂病害验证
    # IMAGE_DIR = "/Users/rain/Downloads/road_diseases/h_crack_0"
    # SAVE_DIR = "/Users/rain/Downloads/check_result_h_crack_0"
    # 病害测试集
    IMAGE_DIR = "/Users/rain/Downloads/origin_images.tar/images_with_box"
    SAVE_DIR = "/Users/rain/Downloads/origin_images.tar/images_with_box_result"
    # prompt中无输入病害PROMPT
    # check_box(IMAGE_DIR, SAVE_DIR, 1)
    # prompt有输入病害
    # check_box_with_check_file(IMAGE_DIR, SAVE_DIR, 1)
    # 仅判断裁剪图是否有病害，prompt无输入病害
    check_img_disease(IMAGE_DIR, SAVE_DIR, 1, "InternVL2-8B-Lora")
