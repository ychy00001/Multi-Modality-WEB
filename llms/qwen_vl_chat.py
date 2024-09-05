from llms.openai_compatible_model import OpenAICompatibleModel


class QwenVLChatModel(OpenAICompatibleModel):
    def __init__(self, model_name: str, url: str, max_tokens: int, temperature: float):
        super().__init__(model_name, url, max_tokens, temperature)
