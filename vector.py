from langchain_pinecone import PineconeVectorStore
index_name="medbot"
from langchain_huggingface.embeddings import HuggingFaceEmbeddings



def download_hf_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLm-L6-v2")
    return embeddings
embeddings=download_hf_embeddings()

docsearch=PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
