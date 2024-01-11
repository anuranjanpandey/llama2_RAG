from steps import embed_model

@pipeline
def embed_pipeline(embed_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
    embed_model = embed_model(embed_model_name)
    docs = [
        "this is one document",
        "and another document"
    ]

    embeddings = embed_model.embed_documents(docs)

    print(f"We have {len(embeddings)} doc embeddings, each with "
        f"a dimensionality of {len(embeddings[0])}.")
    
    return embed_model

