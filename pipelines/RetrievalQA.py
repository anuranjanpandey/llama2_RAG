from langchain.chains import RetrievalQA

def RetrievalQAChain() #to do
    rag_pipeline = RetrievalQA.from_chain_type(
        llm=llm, chain_type='stuff',
        retriever=vectorstore.as_retriever()
    )
    return rag_pipeline
