from src.helper import *
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()


file = file_loader(r"Data/")
chunk_data = chunking_data(file)
embeddings = get_embedding()

docs = FAISS.from_documents(documents=chunk_data, embedding=embeddings)