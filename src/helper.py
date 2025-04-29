from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

def loadPdf(data):
    loader=DirectoryLoader(data,glob='*.pdf',loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

def create_chunks(data):
    text_split=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    return text_split.split_documents(data)

def download_hf_embeddings():
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLm-L6-v2")
    return embeddings