from langchain.vectorstores import Pinecone

def vector_store(index: str, embed_model: str): # to do update
    """Vector store"""
    text_field = 'text'
    pinecone = Pinecone(index, embed_model.embed_query, text_field)
    return pinecone

