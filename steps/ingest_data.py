import logging
from zenml import step
from datasets import load_dataset

class IngestData:
    """Ingest data"""
    def __init__(self) -> None:
        pass

    def get_data(self, dataset_name: str = 'squad', split: str = 'train'):
        """Get data"""
        dataset = load_dataset(dataset_name, split=split)
        logging.info(f"Dataset: {dataset}")

        return dataset
    
@step
def ingest_data(embed_model, index, dataset_name: str = 'jamescalam/llama-2-arxiv-papers-chunked', split: str = 'train'):
    """Ingest data"""
    ingest_data = IngestData()
    dataset = ingest_data.get_data(dataset_name, split)
    dataset = dataset.to_pandas()

    batch_size = 32
    for i in range(0, len(dataset), batch_size):
        i_end = min(len(dataset), i+batch_size)
        batch = dataset.iloc[i:i_end]
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
    
    logging.info(f"Index {index.index_name} has {index.describe_index().stats['num_vectors']} vectors.")