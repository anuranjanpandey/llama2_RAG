from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from zenml import step

@step(enable_cache=True)
def embed_model(embed_model_name: str):
    """Embedding model"""

    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    embed_model = HuggingFaceEmbeddings(
        model_name=embed_model_name,
        model_kwargs={'device': device},
        encode_kwargs={'device': device, 'batch_size': 32}
    )
    return embed_model

