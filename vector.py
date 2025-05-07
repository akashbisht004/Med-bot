import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

# Pinecone configuration
index_name = "medbot"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1' 
        )
    )

def download_hf_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLm-L6-v2")
    return embeddings

embeddings = download_hf_embeddings()

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
