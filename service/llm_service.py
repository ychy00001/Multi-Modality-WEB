from utils import minio_util
from llms import QwenVLChatModel, GLM4VModel, InternVLModel
import copy
import json

VLLM_ENDPOINT = "http://36.212.226.3:41176"
LM_DEPLOY_ENDPOINT = "http://36.212.226.3:41171"

# 定义支持模型列表
qwen_vl_chat_model = QwenVLChatModel(model_name="custom-qwen-vl-chat", url=VLLM_ENDPOINT + "/v1/chat/completions",
                                     max_tokens=1000, temperature=0.9)
internvl_8b_model = InternVLModel(model_name="internvl2-8b", url=LM_DEPLOY_ENDPOINT + "/v1/chat/completions",
                                  max_tokens=1000, temperature=0.9)
internvl_26b_model = InternVLModel(model_name="internvl2-26b", url=LM_DEPLOY_ENDPOINT + "/v1/chat/completions",
                                   max_tokens=2000, temperature=0.9)
glm_4v_model = GLM4VModel(model_name="glm-4v-9b", url=LM_DEPLOY_ENDPOINT + "/v1/chat/completions", max_tokens=200,
                          temperature=0.9)

support_models = {
    "Qwen-Vl-Chat": qwen_vl_chat_model,
    "InternVL-8B": internvl_8b_model,
    "InternVL-26B": internvl_26b_model,
    "glm_4v_model": glm_4v_model,
}


def add_text_msg(model_info: dict, input_text: str):
    return add_text_msg_batch([model_info], input_text)


def add_text_msg_batch(model_infos: list, input_text: str):
    for model_info in model_infos:
        model_info['query'] = model_info['query'] + [(input_text, None)]
        model_info['history'] = model_info['history'] + [(input_text, None)]
    return model_infos


def add_file_msg(model_info: dict, file):
    return add_file_msg_batch([model_info], file)[0]


def add_file_msg_batch(model_infos, file):
    for model_info in model_infos:
        model_info['query'] = model_info['query'] + [((file.name,), None)]
        model_info['history'] = model_info['history'] + [((file.name,), None)]
    return model_infos


def predict(model_info, parameter_info=None):
    results = predict_batch([model_info], parameter_info)
    return results[0]


def predict_batch(model_infos, parameter_info=None):
    """
    预测
    :param model_infos: 模型列表信息
    示例： model_info_state = [
            {
                "modelName": "Qwen-Vl-Chat",
                "query": [],
                "history": [],
                "lastQuery": [],
            }
        ]
    :param parameter_info: 模型参数信息
    示例：parameter = {
            "prompt_template": "你是一个巡检养护助手",
            "temperature": 0.9,
            "top_k": 40,
            "top_p": 1
        }
    :return:
    """
    if parameter_info is None:
        parameter_info = {
            "prompt_template": "{user_input}",
            "temperature": 0.9,
            "top_k": 40,
            "top_p": 1
        }
    print(model_infos)
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
            if isinstance(q, (tuple, list)):
                remote_url = minio_util.fput_file("inference", q[0])
                print("发送消息图片：" + remote_url)
                content.append({
                    'type': "image_url",
                    'image_url': {
                        'url': remote_url
                    }
                })
            else:
                print("User: " + q)
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
        model_info['history'][-1] = (model_info['history'][-1][0], response_text)

        print("Model Response: " + response_text)
        # model_info['history'] = model_info['history'][-10:]
    return model_infos


def get_response(model_info: dict):
    """
    获取模型返回的结果
    :param model_info: 模型信息
    :return:
    """
    return model_info['history'][-1][1]
