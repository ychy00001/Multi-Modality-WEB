# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import numpy as np
from urllib3.exceptions import HTTPError

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

from config import VLLM_ENDPOINT
from utils import minio_util

API_KEY = "EMPTY"

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


def send_request(messages):
    url = VLLM_ENDPOINT + "/v1/chat/completions"
    print("request_url:" + url)
    headers = {
        'User-Agent': 'python-requests/2.31.0',
        'Accept-Encoding': 'gzip, deflate',
        'Accept': '*/*',
        'Connection': 'keep-alive',
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


def _launch_chat(args):
    uploaded_file_dir = os.environ.get("GRADIO_TEMP_DIR") or "./upload"

    def predict(_chatbot1, _chatbot2, task_history1, task_history2):
        print(_chatbot1)
        print(_chatbot2)
        chat_query = _chatbot1[-1][0]
        query = task_history1[-1][0]
        if len(chat_query) == 0:
            _chatbot1.pop()
            task_history1.pop()
            return _chatbot1, _chatbot2
        print("User: " + _parse_text(query))
        history_cp = copy.deepcopy(task_history1)
        full_response = ""
        messages = []
        content = []
        for q, a in history_cp:
            if isinstance(q, (tuple, list)):
                print("发送消息图片：" + image_to_url(q[0]))
                content.append({
                    'type': "image_url",
                    'image_url': {
                        'url': image_to_url(q[0])
                    }
                })
                # content.append({
                #     'type': "image_url",
                #     'image_url': {
                #         'url': "https://n.sinaimg.cn/sinakd20116/96/w2048h2048/20240323/24a7-52a54c327350fe430e27f8b5847a0bf5.jpg"
                #     }
                # })
            else:
                content.append({
                    'type': "text",
                    'text': q
                })
                messages.append({'role': 'user', 'content': content})
                messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': a}]})
                content = []
        messages.pop()
        print(messages)

        response_text = send_request(messages)
        print(response_text)

        _chatbot1[-1] = (_parse_text(chat_query), _remove_image_special(response_text))

        response = response_text
        _chatbot1[-1] = (_parse_text(chat_query), response)
        full_response = _parse_text(response)

        task_history1[-1] = (query, full_response)
        print("Model Response: " + _parse_text(full_response))
        # task_history = task_history[-10:]
        yield _chatbot1, _chatbot2

    def regenerate(_chatbot1, _chatbot2, task_history1, task_history2):
        if not task_history1:
            return _chatbot1, _chatbot2
        item = task_history1[-1]
        if item[1] is None:
            return _chatbot1
        task_history1[-1] = (item[0], None)
        chatbot_item = _chatbot1.pop(-1)
        if chatbot_item[0] is None:
            _chatbot1[-1] = (_chatbot1[-1][0], None)
        else:
            _chatbot1.append((chatbot_item[0], None))
        return predict(_chatbot1, _chatbot2, task_history1, task_history2)

    def add_text(history1, history2, task_history1, task_history2, text1, text2):
        task_text1 = text1
        task_text2 = text2
        history1 = history1 if history1 is not None else []
        history2 = history2 if history2 is not None else []
        task_history1 = task_history1 if task_history1 is not None else []
        task_history2 = task_history2 if task_history2 is not None else []
        history1 = history1 + [(_parse_text(text1), None)]
        history2 = history2 + [(_parse_text(text2), None)]
        task_history1 = task_history1 + [(task_text1, None)]
        task_history2 = task_history2 + [(task_text2, None)]
        return history1, history2, task_history1, task_history2, "", ""

    def add_file(history1, history2, task_history1, task_history2, file1, file2):
        history1 = history1 if history1 is not None else []
        history2 = history2 if history2 is not None else []
        task_history1 = task_history1 if task_history1 is not None else []
        task_history2 = task_history2 if task_history2 is not None else []
        history1 = history1 + [((file1.name,), None)]
        history2 = history2 + [((file2.name,), None)]
        task_history1 = task_history1 + [((file1.name,), None)]
        task_history2 = task_history2 + [((file2.name,), None)]
        return history1, history2, task_history1, task_history2

    def reset_user_input():
        return gr.update(value="")

    def reset_state(task_history1, task_history2):
        task_history1.clear()
        task_history2.clear()
        return []

    with gr.Blocks() as chat_block:
        gr.Markdown("""<center><font size=8>模型效果测试</center>""")
        gr.Markdown("""<center><font size=3>本WebUI基于Gradio打造，可进行多模态大模型效果测试。</center>""")

        with gr.Row():
            chatbot1 = gr.Chatbot(label='Modal1', elem_classes="control-height", height=500)
            chatbot2 = gr.Chatbot(label='Modal2', elem_classes="control-height", height=500)
        query = gr.Textbox(lines=1, label='Input')
        task_history1 = gr.State([])
        task_history2 = gr.State([])

        with gr.Row():
            addfile_btn = gr.UploadButton("📁 Upload (上传文件)", file_types=["image"])
            submit_btn = gr.Button("🚀 Submit (发送)")
            regen_btn = gr.Button("🤔️ Regenerate (重试)")
            empty_bin = gr.Button("🧹 Clear History (清除历史)")

        query.submit(add_text, [chatbot1, chatbot2, task_history1, task_history2, query, query],
                     [chatbot1, chatbot2, task_history1, task_history2]).then(
            predict, [chatbot1, chatbot2, task_history1, task_history2], [chatbot1, chatbot2], show_progress=True
        )
        query.submit(reset_user_input, [], [query])
        submit_btn.click(add_text, [chatbot1, chatbot2, task_history1, task_history2, query, query],
                         [chatbot1, chatbot2, task_history1, task_history2]).then(
            predict, [chatbot1, chatbot2, task_history1, task_history2], [chatbot1, chatbot2], show_progress=True
        )
        submit_btn.click(reset_user_input, [], [query])
        empty_bin.click(reset_state, [task_history1, task_history2], [chatbot1, chatbot2], show_progress=True)
        regen_btn.click(regenerate, [chatbot1, chatbot2, task_history1, task_history2], [chatbot1, chatbot2],
                        show_progress=True)
        addfile_btn.upload(add_file, [chatbot1, chatbot2, task_history1, task_history2, addfile_btn, addfile_btn],
                           [chatbot1, chatbot2, task_history1, task_history2],
                           show_progress=True)

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
        chat_block.load(read_logs, None, logs, every=1)

    chat_block.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()
    _launch_chat(args)


if __name__ == '__main__':
    main()
