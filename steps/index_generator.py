import re
from torch import cuda
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from zenml import step
import pinecone
import os
import time
from langchain.vectorstores import VectorStore
from datasets import load_dataset
import pandas as pd
from zenml.client import Client

@step(enable_cache=False)
def index_generator(embed_model_name: str, dataset_name: str, split: str):
    """Embedding model"""

    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    embed_model = HuggingFaceEmbeddings(
        model_name=embed_model_name,
        model_kwargs={'device': device},
        encode_kwargs={'device': device, 'batch_size': 32}
    )

    docs = [
        "this is one document",
        "and another document"
    ]
    embeddings = embed_model.embed_documents(docs)

    print(f"We have {len(embeddings)} doc embeddings, each with "
        f"a dimensionality of {len(embeddings[0])}.")

    pinecone.init(
        api_key=Client().get_secret('PINECONE_API_KEY').secret_values['PINECONE_API_KEY'],
        environment=Client().get_secret('PINECONE_ENV').secret_values['PINECONE_ENV']
    )

    index_name = 'llama-2-rag'

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            index_name,
            dimension=len(embeddings[0]),
            metric='cosine'
        )
        # wait for index to finish initialization
        while not pinecone.describe_index(index_name).status['ready']:
            time.sleep(1) 

    index = pinecone.Index(index_name)
    print(index.describe_index_stats())

    data = load_dataset(
        path=dataset_name,
        split=split
    )
    
    data = data.to_pandas()

    batch_size = 32

    for i in range(0, len(data), batch_size):
        i_end = min(len(data), i+batch_size)
        batch = data.iloc[i:i_end]
        ids = [f"{x['doi']}-{x['chunk-id']}" for i, x in batch.iterrows()]
        texts = [x['chunk'] for i, x in batch.iterrows()]
        embeds = embed_model.embed_documents(texts)
        # get metadata to store in Pinecone
        metadata = [
            {'text': x['chunk'],
            'source': x['source'],
            'title': x['title']} for i, x in batch.iterrows()
        ]
        # add to Pinecone
        index.upsert(vectors=zip(ids, embeds, metadata))

    print(index.describe_index_stats())

