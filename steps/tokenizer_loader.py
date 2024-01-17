from transformers import AutoTokenizer, PreTrainedTokenizerBase
from zenml import step
from typing import Annotated
from zenml.model import ModelArtifactConfig

@step
def tokenizer_loader(model_name: str = 'sentence-transformers/all-MiniLM-L6-v2')  -> Annotated[
    PreTrainedTokenizerBase, "base_tokenizer", ModelArtifactConfig(overwrite=True)
]:
    """Load tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer
