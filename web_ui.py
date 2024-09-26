# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import os
from argparse import ArgumentParser

import copy

import gradio as gr
import os
import re
import sys
import logging
import base64
import requests
import random
import string
from altair import value
from httpx import Timeout
import itertools

from llms import QwenVLChatModel, GLM4VModel, InternVLModel
from utils import minio_util
from config import VLLM_ENDPOINT, LM_DEPLOY_ENDPOINT

API_KEY = "EMPTY"

DEFAULT_TIMEOUT_CONFIG = Timeout(timeout=30.0)
DEFAULT_CKPT_PATH = 'qwen/Qwen-VL-Chat'
REVISION = 'v1.0.4'
BOX_TAG_PATTERN = r"<box>([\s\S]*?)</box>"
PUNCTUATION = "！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
# DEFAULT_PROMPT_TEMPLATE = """
# 你是一名公路巡检养护校验专家。根据用户提供的病害描述，判断病害描述是否正确，并返回结果。
# 任务要求：
# 1. 如果病害描述正确，返回"check"=true，示例：{"check" true, “type”: "病害类型""}
# 2. 如果病害描述不正确，返回"check"=false, 示例：{“check”: false, “type”: “病害类型”}
# 3. 病害类型type包括：["无病害", "纵向裂隙", "横向裂隙", "网状裂隙", "块状裂缝", "井盖破损", "井盖缺失", "抛洒物", "积水", "坑槽", "龟裂"]
# 4. 如无明显病害，则type="无病害"，示例：{"check": "检测结果", “type”: "无病害"}
# 5. 返回结果必须严格按照JSON格式
# 6. 仅返回JSON结果，不需要额外内容
# 7. type为检测病害类型，如果检测病害类型与用户病害类型不一致，请输出检测病害类型
#
# {user_input}
#
# """

DEFAULT_PROMPT_TEMPLATE = """
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
{user_input}
"""


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return False


sys.stdout = Logger("output.log")


def generate_random_string(length=4):
    # 定义可选字符集，包括大写字母、小写字母和数字
    characters = string.ascii_letters + string.digits
    # 使用random.choices从characters中随机选取length个字符
    random_string = ''.join(random.choices(characters, k=length))
    return random_string


def test(x):
    print("This is a test")
    print(f"Your function is running with input {x}...")
    return x


def read_logs():
    sys.stdout.flush()
    with open("output.log", "r") as f:
        return f.read()


def _get_args():
    parser = ArgumentParser()
    # parser.add_argument("-c", "--checkpoint-path", type=str, #default=DEFAULT_CKPT_PATH,
    #                    help="Checkpoint name or path, #default to %(default)r")
    # parser.add_argument("--revision", type=str, #default=REVISION)
    # parser.add_argument("--cpu-only", action="store_true", #help="Run demo with CPU only")
    parser.add_argument("--prod", action="store_true", default=False,
                        help="tart prod.")
    parser.add_argument("--share", action="store_true", default=False,
                        help="Create a publicly shareable link for the interface.")
    parser.add_argument("--inbrowser", action="store_true", default=False,
                        help="Automatically launch the interface in a new tab on the default browser.")
    parser.add_argument("--server-port", type=int, default=9081,
                        help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="0.0.0.0",
                        help="Demo server name.")

    args = parser.parse_args()
    return args


def split_into_groups(n, m):
    """
    根据输入n，最大个数m，分组n
    n=3 m=2  [[0,1],[2]]
    n=5 m=2  [[0,1], [2,3], [4]]
    :param n: 输入数字
    :param m: 每组最大个数
    :return:
    """
    # 计算所需的最少组数
    k = -(-n // m)  # 使用负数除法向上取整
    # 初始化结果列表
    groups = []
    # 当前起始索引
    start_index = 0
    for i in range(k):
        # 每组大小为 m 或者是剩余的元素数量
        group_size = min(m, n - start_index)
        # 创建并添加这一组
        end_index = start_index + group_size
        groups.append(list(range(start_index, end_index)))
        # 更新下一个组的起始索引
        start_index = end_index
    return groups


def image_to_base64(image_path):
    # 打开图像文件并读取其二进制内容
    with open(image_path, 'rb') as file:
        # 读取文件内容
        binary_data = file.read()

    # 将二进制数据编码为 Base64 字符串
    base64_data = base64.b64encode(binary_data)

    # 将字节对象转换为字符串
    base64_string = base64_data.decode('utf-8')

    return base64_string


def image_to_url(image_path):
    return minio_util.fput_file("inference", image_path)


def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def _remove_image_special(text):
    text = text.replace('<ref>', '').replace('</ref>', '')
    return re.sub(r'<box>.*?(</box>|$)', '', text)


def change_prompt_template(_input_text, _save_parameter):
    _save_parameter['prompt_template'] = _input_text
    return _input_text, _save_parameter


def change_temperature(_sld_value, _save_parameter):
    _save_parameter['temperature'] = _sld_value
    return _sld_value, _save_parameter


def change_top_k(_sld_value, _save_parameter):
    _save_parameter['top_k'] = _sld_value
    return _sld_value, _save_parameter


def change_top_p(_sld_value, _save_parameter):
    _save_parameter['top_p'] = _sld_value
    return _sld_value, _save_parameter


def _launch_chat(args):
    uploaded_file_dir = os.environ.get("GRADIO_TEMP_DIR") or "./upload"
    # 定义支持模型列表
    qwen_vl_chat_model = QwenVLChatModel(model_name="custom-qwen-vl-chat", url=VLLM_ENDPOINT + "/v1/chat/completions",
                                         max_tokens=1000, temperature=0.8)
    qwen_vl2_instruct_model = QwenVLChatModel(model_name="Qwen2-VL-7B-Instruct",
                                              url=VLLM_ENDPOINT + "/v1/chat/completions",
                                              max_tokens=1000, temperature=0.8)
    internvl_8b_model = InternVLModel(model_name="InternVL2-8B", url=LM_DEPLOY_ENDPOINT + "/v1/chat/completions",
                                      max_tokens=1000, temperature=0.9)
    internvl_8b_model_lora = InternVLModel(model_name="InternVL2-8B-Lora",
                                           url=LM_DEPLOY_ENDPOINT + "/v1/chat/completions",
                                           max_tokens=1000, temperature=0.9)
    internvl_26b_model = InternVLModel(model_name="InternVL2-26B", url=LM_DEPLOY_ENDPOINT + "/v1/chat/completions",
                                       max_tokens=2000, temperature=0.8)
    glm_4v_model = GLM4VModel(model_name="glm-4v-9b", url=LM_DEPLOY_ENDPOINT + "/v1/chat/completions", max_tokens=200,
                              temperature=0.9)
    support_models = {
        "Qwen-Vl-Chat": qwen_vl_chat_model,
        "Qwen-VL2-Instruct": qwen_vl2_instruct_model,
        "InternVL2-8B": internvl_8b_model,
        "InternVL2-8B-Lora": internvl_8b_model_lora,
        "InternVL-26B": internvl_26b_model,
        "glm_4v_model": glm_4v_model,
    }

    def predict(model_infos, parameter_info):
        with_history = False
        for model_info in model_infos:
            chat_query = model_info['query']
            if len(chat_query) == 0:
                continue
            messages = []
            content = []
            request_chat = copy.deepcopy(model_info['query'])
            if with_history:
                request_chat = copy.deepcopy(model_info['history'])
            for q, a in request_chat:
                if q is None:
                    q = ""
                if isinstance(q, (tuple, list)):
                    print("发送消息图片：" + image_to_url(q[0]))
                    content.append({
                        'type': "image_url",
                        'image_url': {
                            'url': image_to_url(q[0])
                        }
                    })
                else:
                    print("User: " + ("无" if q is None else q))
                    content.append({
                        'type': "text",
                        'text': parameter_info["prompt_template"].replace("{user_input}", q)
                    })
                    messages.append({'role': 'user', 'content': content})
                    messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': a}]})
                    content = []
            messages.pop()
            print(f"发送message: {json.dumps(messages, ensure_ascii=False)}")
            response_text = support_models[model_info['modelName']].call(messages)

            print(response_text)
            # 重构结构
            model_info['lastQuery'] = model_info['query']
            model_info['query'] = []
            query_text = model_info['history'][-1][0]
            if len(str(query_text).strip()) == 0:
                query_text = None
            model_info['history'][-1] = (query_text, response_text)

            print("Model Response: " + response_text)
            # model_info['history'] = model_info['history'][-10:]
            print(model_infos)
        return model_infos

    def regenerate(model_infos, parameter_info):
        has_last_query = False
        for model_info in model_infos:
            if len(model_info['lastQuery']) > 0:
                has_last_query = True
                model_info['query'] = model_info['lastQuery']
                model_info['lastQuery'] = []
                model_info['history'][-1] = (model_info['history'][-1][0], None)
        if has_last_query:
            return predict(model_infos, parameter_info)
        return model_infos

    def add_text(model_infos, input_text):
        if len(str(input_text).strip()) == 0:
            input_text = None
        for model_info in model_infos:
            model_info['query'] = model_info['query'] + [(input_text, None)]
            model_info['history'] = model_info['history'] + [(input_text, None)]
        return model_infos

    def add_file(model_infos, file, query):
        for model_info in model_infos:
            model_info['query'] = model_info['query'] + [((file.name,), None)]
            model_info['history'] = model_info['history'] + [((file.name,), None)]
        return model_infos, query

    def reset_user_input():
        return gr.update(value="")

    def reset_state(model_infos):
        for model_info in model_infos:
            model_info['query'] = []
            model_info['history'] = []
            model_info['lastQuery'] = []
        return model_infos

    def add_model(model_infos):
        model_infos.append({
            "modelName": "Qwen-Vl-Chat",
            "query": [],
            "history": [],
            "lastQuery": [],
        })
        return model_infos

    render_chatbot_js = """
    <script>
    
    // 观察器的配置（需要观察什么变动）
    const config = { attributes: true, childList: true, subtree: true };
     
    // 当观察到变动时执行的回调函数
    const callback = function(mutationsList, observer) {
        // Use traditional 'for loops' for IE 11
        for(let mutation of mutationsList) {
            if (mutation.type === 'childList') {
                console.log('A child node has been added or removed.');
                // 使用函数查询 id 以 'chatbot_' 开头的所有元素
                var chatContainers = queryElementsByIdPrefix('chatbot_');
                // 更改滚动条
                chatContainers.forEach(chatContainer => {
                    var children = chatContainer.querySelector('div[aria-label="chatbot conversation"]');
                    children.scrollTop = children.scrollHeight;
                });
            }
            else if (mutation.type === 'attributes') {
                console.log('The ' + mutation.attributeName + ' attribute was modified.');
            }
        }
    };

    
    function queryElementsByIdPrefix(prefix) {
        // 获取所有带有 id 属性的元素
        var allElementsWithId = document.querySelectorAll("[id]");
        // 创建一个数组来存储匹配的元素
        var matchingElements = [];
        // 遍历所有元素并检查 id 是否以指定的前缀开始
        for (var i = 0; i < allElementsWithId.length; i++) {
            var element = allElementsWithId[i];
            if (element.id.startsWith(prefix)) {
                matchingElements.push(element);
            }
        }
        // 返回匹配的元素列表
        return matchingElements;
    }
    
    // 存储当前chatbot组件
    const chatbotMap = new Map();
    
    // 定义一个函数来处理定时任务
    function checkAndProcessChatbotElements() {
        // 查询所有 chatbot 元素
        var chatbotElements = queryElementsByIdPrefix('chatbot_');
        
        // 遍历当前查找到的 chatbot 元素
        chatbotElements.forEach(function(element) {
            // 检查该元素是否已经在 chatbotMap 中
            if (!chatbotMap.has(element.id)) {
                // 如果是新元素，则添加到 chatbotMap
                console.log(`New chatbot element found with ID: ${element.id}`);
                // 以上述配置开始观察目标节点
                var children = element.querySelector('div[aria-label="chatbot conversation"]');
                // 存储
                chatbotMap.set(element.id, {"element": element, "observer": observer});
            }
        });
    
        // 遍历 chatbotMap 中的所有元素
        chatbotMap.forEach(function(element, id) {
            // 检查该元素是否还在当前查找到的 chatbot 元素中
            if (!chatbotElements.some(e => e.id === id)) {
                // 如果不在，则打印一条消息表示元素被删除
                console.log(`Chatbot element with ID ${id} was removed`);
                //删除元素
                chatbotMap.delete(id);
            }
        });
    }
        
    //if (window.chatbotScroll) {
    //    window.clearInterval(window.chatbotScroll);
    //}
    // 监听所有chatbot组件
    // window.chatbotScroll = window.setInterval(checkAndProcessChatbotElements, 500);
    
    // 监听界面元素更新
    setTimeout(function(){
        const targetNode = document.getElementById('component-5');
        // 创建一个观察器实例并传入回调函数
        const observer = new MutationObserver(callback);
        // 以上述配置开始观察目标节点
        observer.observe(targetNode, config);
    },2000);
    </script>
    
    """

    with gr.Blocks(head=render_chatbot_js) as chat_block:
        gr.Markdown("""<center><font size=8>模型测试平台</center>""")
        gr.Markdown("""<center><font size=3>本WebUI基于Gradio打造，仅供内部多模态大模型效果测试。 @Rain</center>""")

        # 定义模型数 默认是一个模型
        model_info_state = gr.State([
            {
                "modelName": "Qwen-Vl-Chat",
                "query": [],
                "history": [],
                "lastQuery": [],
            }
        ])
        add_model_btn = gr.Button("添加模型")  # 创建一个按钮，用于增加模型组件
        add_model_btn.click(add_model, model_info_state, model_info_state)  # 将按钮点击事件绑定到lambda函数，每次点击时将模型数量加1

        # 定义渲染待办事项的函数
        @gr.render(inputs=model_info_state)
        def render_models(model_infos):
            # 根据model_num分组 看需要遍历多少行，定义一行最多2个组件
            row_max_num = 2
            group = split_into_groups(len(model_infos), row_max_num)
            # 渲染模型组件
            for item_list in group:
                # 分组看需要多少行
                with gr.Row():  # 对于每个未完成的任务，创建一个行容器
                    for index in item_list:
                        with gr.Column(scale=1):
                            def select_model(select_value, select_index=index):
                                model_infos[select_index]['modelName'] = select_value
                                return select_value

                            with gr.Row():
                                dropdown = gr.Dropdown(value=model_infos[index]['modelName'],
                                                       choices=list(support_models.keys()),
                                                       label="Select Model", allow_custom_value=True, scale=11)
                                dropdown.select(select_model, [dropdown], [dropdown])
                                delete_btn = gr.Button("删除", variant="stop", scale=1, min_width=8)

                                # 定义删除任务的函数
                                def delete(model_index=index):
                                    print(f"删除索引{model_index}")
                                    model_infos.pop(model_index)
                                    return model_infos

                                delete_btn.click(delete, None, [model_info_state])  # 绑定按钮点击到delete函数

                            chatbot = gr.Chatbot(value=model_infos[index]['history'],
                                                 label=model_infos[index]['modelName'], height=500,
                                                 show_copy_button=True, bubble_full_width=False,
                                                 elem_id=f"chatbot_{generate_random_string(4)}")

        parameter = gr.State({
            "prompt_template": DEFAULT_PROMPT_TEMPLATE,
            "temperature": 0.9,
            "top_k": 40,
            "top_p": 1
        })
        query = gr.Textbox(lines=1, label='Input')
        with gr.Row():
            addfile_btn = gr.UploadButton("📁 Upload (上传文件)", file_types=["image"])
            submit_btn = gr.Button("🚀 Submit (发送)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")
            empty_bin = gr.Button("🧹 Clear History (清除历史)")

        query.submit(add_text, [model_info_state, query], [model_info_state]).then(
            predict, [model_info_state, parameter], [model_info_state], show_progress=True
        )
        query.submit(reset_user_input, [], [query])
        submit_btn.click(add_text, [model_info_state, query], [model_info_state]).then(
            predict, [model_info_state, parameter], [model_info_state], show_progress=True
        )
        submit_btn.click(reset_user_input, [], [query])
        empty_bin.click(reset_state, [model_info_state], [model_info_state], show_progress="full")
        regen_btn.click(regenerate, [model_info_state, parameter], [model_info_state],
                        show_progress="full")
        addfile_btn.upload(add_file, [model_info_state, addfile_btn, query],
                           [model_info_state, query],
                           show_progress="full")

        with gr.Blocks() as demo:
            with gr.Tab("日志"):
                logs = gr.Textbox(lines=10)
                chat_block.load(read_logs, None, logs, every=1)
            with gr.Tab("高级配置"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # system prompt
                        prompt_template_tbx = gr.Textbox(lines=10, label="Prompt Template",
                                                         value=parameter.value["prompt_template"],
                                                         interactive=True)  # prompt
                        prompt_template_tbx.blur(change_prompt_template, [prompt_template_tbx, parameter],
                                                 [prompt_template_tbx, parameter])
                    with gr.Column(scale=1):
                        # config
                        temperature_sld = gr.Slider(0.0, 1.0, value=parameter.value["temperature"],
                                                    label="temperature", interactive=True)  # temperature
                        top_p_sld = gr.Slider(0.0, 1.0, value=parameter.value["top_p"], label="top_p",
                                              interactive=True)  # top_p
                        top_k_sld = gr.Slider(1, 40, value=parameter.value["top_k"], label="top_k",
                                              interactive=True)  # top_k
                        temperature_sld.input(change_temperature, [temperature_sld, parameter],
                                              [temperature_sld, parameter])
                        top_p_sld.input(change_top_p, [top_p_sld, parameter],
                                        [top_p_sld, parameter])
                        top_k_sld.input(change_top_k, [top_k_sld, parameter],
                                        [top_k_sld, parameter])

        chat_block.queue().launch(
            share=args.share,
            inbrowser=args.inbrowser,
            server_port=args.server_port,
            server_name=args.server_name,
            auth=("admin", 'jtwl998')
        )


def main():
    args = _get_args()
    _launch_chat(args)


if __name__ == '__main__':
    main()
