import transformers
import torch
from zenml import step

class BaseLLM:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_text(self):
        output = transformers.pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            temperature=0.1, 
            max_new_tokens=512,
            repetition_penalty=1.1
        )
        return output

