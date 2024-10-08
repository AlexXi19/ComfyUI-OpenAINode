from typing import Tuple

import torch
from openai import Client as OpenAIClient

from .lib import credentials, image


class ImageWithPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Image": ("IMAGE", {}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Generate a high quality caption for the image. The most important aspects of the image should be described first. If needed, weights can be applied to the caption in the following format: '(word or phrase:weight)', where the weight should be a float less than 2.",
                    },
                ),
                "max_tokens": ("INT", {"min": 1, "max": 2048, "default": 77}),
                "model": (
                    [
                        "gpt-4o",
                        "gpt-4o-mini",
                        "gpt-4-turbo",
                    ],
                    {},
                ),
                "api_key": (
                    "STRING",
                    {"multiline": False, "default": ""},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_completion"

    CATEGORY = "OpenAI"

    def generate_completion(
        self,
        Image: torch.Tensor,
        prompt: str,
        max_tokens: int,
        model: str,
        api_key: str,
    ) -> Tuple[str]:
        api_key = api_key or credentials.get_open_ai_api_key()
        self.open_ai_client: OpenAIClient = OpenAIClient(api_key=api_key)
        b64image = image.pil2base64(image.tensor2pil(Image))
        response = self.open_ai_client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64image}"},
                        },
                    ],
                }
            ],
        )
        if len(response.choices) == 0:
            raise Exception("No response from OpenAI API")

        return (response.choices[0].message.content,)


class TextWithPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": (
                    "STRING",
                    {"forceInput": True},
                ),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Generate a high quality caption for the image. The most important aspects of the image should be described first. If needed, weights can be applied to the caption in the following format: '(word or phrase:weight)', where the weight should be a float less than 2.",
                    },
                ),
                "max_tokens": ("INT", {"min": 1, "max": 2048, "default": 77}),
                "model": (
                    [
                        "gpt-4o",
                        "gpt-4o-mini",
                        "gpt-4-turbo",
                        "gpt-4-turbo-2024-04-09",
                        "gpt-4-turbo-preview",
                        "gpt-4-0125-preview",
                        "gpt-4-1106-preview",
                        "gpt-4",
                        "gpt-4-0613",
                        "gpt-4-0314",
                        "gpt-3.5-turbo-0125",
                        "gpt-3.5-turbo",
                        "gpt-3.5-turbo-1106",
                        "gpt-3.5-turbo-instruct",
                    ],
                    {},
                ),
                "api_key": (
                    "STRING",
                    {"multiline": False, "default": ""},
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_completion"

    CATEGORY = "OpenAI"

    def generate_completion(
        self, text: str, prompt: str, max_tokens: int, model: str, api_key: str
    ) -> Tuple[str]:
        api_key = api_key or credentials.get_open_ai_api_key()
        self.open_ai_client: OpenAIClient = OpenAIClient(api_key=api_key)
        response = self.open_ai_client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt + "\n" + text},
                    ],
                }
            ],
        )
        if len(response.choices) == 0:
            raise Exception("No response from OpenAI API")

        return (response.choices[0].message.content,)
        return (response.choices[0].message.content,)
