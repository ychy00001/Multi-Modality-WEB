from abc import ABC, abstractmethod
from typing import List

class BaseModel(ABC):
    def __init__(self, model_name: str, url: str, max_tokens: int=512, temperature: float=0.9):
        self.model_name = model_name
        self.url = url
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.headers = {
            'User-Agent': 'python-requests/2.31.0',
            'Accept-Encoding': 'gzip, deflate',
            'Accept': '*/*',
            'Connection': 'keep-alive',
            'Content-Type': 'application/json',
        }

    @abstractmethod
    def call(self, messages: List[str]) -> str:
        pass

