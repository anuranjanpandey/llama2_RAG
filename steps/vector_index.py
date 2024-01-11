import os
import time
import logging
import pinecone
from zenml import step

class VectorIndex:
    """Vector index"""
    def __init__(self):

        pinecone.init(
        api_key=os.environ.get('PINECONE_API_KEY') or 'PINECONE_API_KEY',
        environment=os.environ.get('PINECONE_ENVIRONMENT') or 'PINECONE_ENV'
        )
        
    def initialize_vector_index(self, index_name: str = 'llama-2-rag', embeddings, metric: str = 'cosine'):
        """Initialize vector index"""
        self.index_name = index_name

        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                self.index_name,
                dimension=len(embeddings[0]),
                metric='cosine'
            )
            # wait for index to finish initialization
            while not pinecone.describe_index(index_name).status['ready']:
                time.sleep(1) 

    def __call__(self, vector):
        return self.index.nearest(vector)