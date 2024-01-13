from os import pipe
import pinecone
from zenml import pipeline
from steps.index_generator import index_generator
from steps.vector_index import VectorIndex
from steps.ingest_data import ingest_data

# @pipeline
# def embed_pipeline(embed_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
#     model = embed_model(embed_model_name)
#     docs = [
#         "this is one document",
#         "and another document"
#     ]
#     embeddings = model.embed_documents(docs)

#     print(f"We have {len(embeddings)} doc embeddings, each with "
#         f"a dimensionality of {len(embeddings[0])}.")
    
#     index = VectorIndex()  # directly return the index and create a diff step for above logic of embedding
#     index.initialize_vector_index(len(embeddings[0]))
#     print(f"Index {index.index_name} is ready: {pinecone.describe_index(index.index_name).status['ready']}")

#     ingest_data(model, index, dataset_name='jamescalam/llama-2-arxiv-papers-chunked', split='train')

#     return model


@pipeline
def docs_to_index_pipeline() -> None:
    index_generator('sentence-transformers/all-MiniLM-L6-v2', 'jamescalam/llama-2-arxiv-papers-chunked', 'train')
