from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
#The RecursiveCharacterTextSplitter attempts to keep larger units (e.g., paragraphs) intact.
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv


load_dotenv()
pdf_file = Path("pythonbook.pdf")

#Load this file 
loader = PyPDFLoader(str(pdf_file))
# this will gives page by page
pages = loader.load() 

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100 #this will make your splitter takes a part of the above pragraph , so there are a link with its
)

chunks = text_splitter.split_documents(pages)
print("Indexing to Qdrant...")

#Important : You need to set your API key 
# To use embadding model and convert it to vector you need qudren db bradge 
embading_model = OpenAIEmbeddings(model="text-embedding-3-large")

# pass all the chunks, model , your db path url 
qudren_vector = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embading_model,
    url = "http://localhost:6333",
    collection_name="pythonbook_open_ai",
)

print("Done!")