from .nodes import TextWithPrompt, ImageWithPrompt

NODE_CLASS_MAPPINGS = {
    "ImageWithPrompt": ImageWithPrompt,
    "TextWithPrompt": TextWithPrompt,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageWithPrompt":  "Image With Prompt",
    "TextWithPrompt":   "Text With Prompt",
}
