from src.helper import loadPdf,download_hf_embeddings,create_chunks
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import os
from langchain_pinecone import PineconeVectorStore

load_dotenv()

PINCONE_API_KEY=os.getenv('PINECONE_API_KEY')

documents=loadPdf(data="./Data")
texts=create_chunks(data=documents)
embeddings=download_hf_embeddings()

pc=Pinecone(api_key=PINCONE_API_KEY)

index_name="medbot"

pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

os.environ["PINECONE_API_KEY"]=PINCONE_API_KEY

createVector=PineconeVectorStore.from_documents(
    documents=texts,
    index_name=index_name,
    embedding=embeddings
)
