from llms.glm_4v import GLM4VModel
from llms.intern_vl_2 import InternVLModel
from llms.qwen_vl_chat import QwenVLChatModel

if __name__ == "__main__":
    qwen_vl_chat_model = QwenVLChatModel(model_name="Qwen-VL-Chat", url="http://172.17.0.1:41176/v1/chat/completions",
                                         max_tokens=50, temperature=0.7)
    internvl_8b_model = InternVLModel(model_name="InternVL-8B", url="http://172.17.0.1:41176/v1/chat/completions",
                                      max_tokens=100, temperature=0.8)
    internvl_26b_model = InternVLModel(model_name="InternVL-26B", url="http://172.17.0.1:41176/v1/chat/completions",
                                       max_tokens=200, temperature=0.9)
    glm_4v = GLM4VModel(model_name="InternVL-26B", url="http://172.17.0.1:41176/v1/chat/completions", max_tokens=200,
                        temperature=0.9)

    print(qwen_vl_chat_model.call(["Hello, how are you?"]))
    print(internvl_8b_model.call(["What is the weather like today?"]))
    print(internvl_26b_model.call(["Can you generate some text for me?"]))
