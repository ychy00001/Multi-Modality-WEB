# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import numpy as np
from urllib3.exceptions import HTTPError
os.system('pip install dashscope  modelscope -U')
os.system('pip install gradio==3.*')

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
from argparse import ArgumentParser
from pathlib import Path

import copy
import gradio as gr
import os
import re
import json
import sys
import logging
import secrets
import tempfile
import base64
import requests
from http import HTTPStatus
from dashscope import MultiModalConversation
import dashscope
API_KEY = "EMPTY"
dashscope.api_key = API_KEY


DEFAULT_CKPT_PATH = 'qwen/Qwen-VL-Chat'
REVISION = 'v1.0.4'
BOX_TAG_PATTERN = r"<box>([\s\S]*?)</box>"
PUNCTUATION = "！？。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."

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
    #parser.add_argument("-c", "--checkpoint-path", type=str, #default=DEFAULT_CKPT_PATH,
    #                    help="Checkpoint name or path, #default to %(default)r")
    #parser.add_argument("--revision", type=str, #default=REVISION)
    #parser.add_argument("--cpu-only", action="store_true", #help="Run demo with CPU only")

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


"""
('/tmp/gradio/1837abb0176495ff182050801ebff1fa9b18fc4a/aiyinsitan.jpg',),
  None],
 ['这是谁？',
  '图中是爱因斯坦，阿尔伯特·爱因斯坦（Albert '
  'Einstein），是出生于德国、拥有瑞士和美国国籍的犹太裔理论物理学家，他创立了现代物理学的两大支柱的相对论及量子力学。'],
 ['框处里面的人', '图中框内是爱因斯坦的半身照，照片中爱因斯坦穿着一件西装，留着标志性的胡子和蜷曲的头发。'],
 ['框出里面的人',
  ('/tmp/gradio/71cf5c2551009fd9a00e0d80bc7ab7fb8de211b5/tmp115aba5d70.jpg',)],
 [None, '里面的人'],
 ('介绍一下',
  '阿尔伯特·爱因斯坦（Albert '
  'Einstein），是出生于德国、拥有瑞士和美国国籍的犹太裔理论物理学家，他创立了现代物理学的两大支柱的相对论及量子力学。他的贡献包括他提出的相对论（尤其是狭义相对论和广义相对论）、量子力学的开创性贡献以及他对于 '
  'gravity 的贡献。爱因斯坦也是诺贝尔奖得主以及美国公民。')]
"""

def _remove_image_special(text):
    text = text.replace('<ref>', '').replace('</ref>', '')
    return re.sub(r'<box>.*?(</box>|$)', '', text)

def send_request(messages):
    url = "http://172.17.0.1:41176/v1/chat/completions"
    headers = {
        'Content-Type': 'application/json',
    }
    data = {
        "model": "custom-qwen-vl-chat",
        "messages": messages,
        "max_tokens": 2000,
        "temperature": 0
    }

    # Convert the data dictionary to a JSON string
    data_json = json.dumps(data)

    # Send the POST request
    response = requests.post(url, headers=headers, data=data_json)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        result = response.json()

        # Extract the content from the message
        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            return content
        else:
            print("No choices found in the response.")
            return "结果异常，请重试！"
    else:
        # Print an error message if the request failed
        print(f"Request failed with status code {response.status_code}")
        return "连接异常，请重试！"

def _launch_demo(args):
    uploaded_file_dir = os.environ.get("GRADIO_TEMP_DIR") or str(
        Path(tempfile.gettempdir()) / "gradio"
    )

    def predict(_chatbot, task_history):
        chat_query = _chatbot[-1][0]
        query = task_history[-1][0]
        if len(chat_query) == 0:
            _chatbot.pop()
            task_history.pop()
            return _chatbot
        print("User: " + _parse_text(query))
        history_cp = copy.deepcopy(task_history)
        full_response = ""
        messages = []
        content = []
        for q, a in history_cp:
            if isinstance(q, (tuple, list)):
                #content.append({
                #    'type': "image_url",
                #    'image_url': {
                #        'url': image_to_base64(q[0])
                #    }
                #})
                content.append({
                    'type': "image_url",
                    'image_url': {
                        'url': "https://n.sinaimg.cn/sinakd20116/96/w2048h2048/20240323/24a7-52a54c327350fe430e27f8b5847a0bf5.jpg"
                    }
                })
            else:
                content.append({
                    'type': "text",
                    'text': q
                })
                messages.append({'role': 'user', 'content': content})
                messages.append({'role': 'assistant', 'content': [{'type':'text','text': a}]})
                content = []
        messages.pop()
        print(messages)

        response_text = send_request(messages)
        print("AAAAAA"+response_text)
        print(response_text)
        
        _chatbot[-1] = (_parse_text(chat_query), _remove_image_special(response_text))

        response = response_text
        _chatbot[-1] = (_parse_text(chat_query), response)
        full_response = _parse_text(response)

        task_history[-1] = (query, full_response)
        print("Qwen-VL-Chat: " + _parse_text(full_response))
        # task_history = task_history[-10:]
        yield _chatbot

    def regenerate(_chatbot, task_history):
        if not task_history:
            return _chatbot
        item = task_history[-1]
        if item[1] is None:
            return _chatbot
        task_history[-1] = (item[0], None)
        chatbot_item = _chatbot.pop(-1)
        if chatbot_item[0] is None:
            _chatbot[-1] = (_chatbot[-1][0], None)
        else:
            _chatbot.append((chatbot_item[0], None))
        return predict(_chatbot, task_history)

    def add_text(history, task_history, text):
        task_text = text
        history = history if history is not None else []
        task_history = task_history if task_history is not None else []
        history = history + [(_parse_text(text), None)]
        task_history = task_history + [(task_text, None)]
        return history, task_history, ""

    def add_file(history, task_history, file):
        history = history if history is not None else []
        task_history = task_history if task_history is not None else []
        history = history + [((file.name,), None)]
        task_history = task_history + [((file.name,), None)]
        return history, task_history

    def reset_user_input():
        return gr.update(value="")

    def reset_state(task_history):
        task_history.clear()
        return []

    with gr.Blocks() as demo:
        gr.Markdown("""\
<p align="center"><img src="https://modelscope.oss-cn-beijing.aliyuncs.com/resource/qwen.png" style="height: 80px"/><p>""")
        gr.Markdown("""<center><font size=8>Qwen-VL-Plus</center>""")
        gr.Markdown(
            """\
<center><font size=3>This WebUI is based on Qwen-VL-Plus, the upgraded version of Qwen-VL, developed by Alibaba Cloud.</center>""")
        gr.Markdown("""<center><font size=3>本WebUI基于Qwen-VL-Plus打造，这是Qwen-VL的升级版。</center>""")
        gr.Markdown("""\
<center><font size=4> \
<a href="https://github.com/QwenLM/Qwen-VL#qwen-vl-plus">Github</a>&nbsp ｜ &nbsp
Qwen-VL <a href="https://modelscope.cn/models/qwen/Qwen-VL/summary">🤖 </a> 
| <a href="https://huggingface.co/Qwen/Qwen-VL">🤗</a>&nbsp ｜ 
Qwen-VL-Chat <a href="https://modelscope.cn/models/qwen/Qwen-VL-Chat/summary">🤖 </a> | 
<a href="https://huggingface.co/Qwen/Qwen-VL-Chat">🤗</a>&nbsp ｜
Qwen-VL-Plus 
<a href="https://huggingface.co/spaces/Qwen/Qwen-VL-Plus">🤗</a>&nbsp
<a href="https://modelscope.cn/studios/qwen/Qwen-VL-Chat-Demo/summary">🤖 </a>&nbsp ｜
Qwen-VL-Max 
<a href="https://huggingface.co/spaces/Qwen/Qwen-VL-Max">🤗</a>&nbsp
<a href="https://modelscope.cn/studios/qwen/Qwen-VL-Max/summary">🤖 </a>&nbsp ｜ 
<a href="https://qianwen.aliyun.com">Web</a> |
<a href="https://help.aliyun.com/zh/dashscope/developer-reference/vl-plus-quick-start/">API</a></center>""")

        chatbot = gr.Chatbot(label='Qwen-VL-Plus', elem_classes="control-height", height=500)
        query = gr.Textbox(lines=1, label='Input')
        task_history = gr.State([])

        with gr.Row():
            addfile_btn = gr.UploadButton("📁 Upload (上传文件)", file_types=["image"])
            submit_btn = gr.Button("🚀 Submit (发送)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")
            empty_bin = gr.Button("🧹 Clear History (清除历史)")

        query.submit(add_text, [chatbot, task_history, query], [chatbot, task_history]).then(
            predict, [chatbot, task_history], [chatbot], show_progress=True
        )
        query.submit(reset_user_input, [], [query])
        submit_btn.click(add_text, [chatbot, task_history, query], [chatbot, task_history]).then(
            predict, [chatbot, task_history], [chatbot], show_progress=True
        )
        submit_btn.click(reset_user_input, [], [query])
        empty_bin.click(reset_state, [task_history], [chatbot], show_progress=True)
        regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)
        addfile_btn.upload(add_file, [chatbot, task_history, addfile_btn], [chatbot, task_history], show_progress=True)

        gr.Markdown("""\
<font size=2>Note: This demo is governed by the original license of Qwen-VL. \
We strongly advise users not to knowingly generate or allow others to knowingly generate harmful content, \
including hate speech, violence, pornography, deception, etc. \
(注：本演示受Qwen-VL的许可协议限制。我们强烈建议，用户不应传播及不应允许他人传播以下内容，\
包括但不限于仇恨言论、暴力、色情、欺诈相关的有害信息。)""")
        with gr.Row():
            input = gr.Textbox()
            output = gr.Textbox()
        btn = gr.Button("Run")
        btn.click(test, input, output)
        logs = gr.Textbox()
        demo.load(read_logs, None, logs, every=1)

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()
    _launch_demo(args)


if __name__ == '__main__':
    main()
