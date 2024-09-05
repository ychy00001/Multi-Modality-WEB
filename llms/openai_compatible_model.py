from llms.base_model import BaseModel
from typing import List
import json
import requests


class OpenAICompatibleModel(BaseModel):
    def call(self, messages: List[str]) -> str:
        print("request_url:" + self.url)
        data = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        # Convert the data dictionary to a JSON string
        data_json = json.dumps(data)
        # Send the POST request
        response = requests.post(self.url, headers=self.headers, data=data_json)
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
