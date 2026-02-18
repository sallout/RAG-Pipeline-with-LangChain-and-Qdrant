from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from pathlib import Path 
from tqdm import tqdm # progress lib

file_path = Path('pythonbook.pdf')
model_name='google/embeddinggemma-300m'
qudren_db_url = 'http://localhost:6333'
qudren_db_name = 'pythonbook'

#1- Setup GPU Embadding 
print("Setup GPU Embedding...")
embadding  = HuggingFaceEmbeddings(
    model_name = model_name,
    model_kwargs = { 'device' : 'cuda'},
    encode_kwargs={'normalize_embeddings': True, 'batch_size': 128}
)

print("Load PDF file...")
#2- Load and Split the file 
docs = PyMuPDFLoader(str(file_path)).load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000, 
    chunk_overlap = 250
).split_documents(docs)

# 3. Batch Processing with Progress Bar
batch_size = 64  # How many chunks to send to Qdrant at once
vectorstore = None
print(f"Indexing {len(splitter)} chunks...")
for i in tqdm(range(0 ,len(splitter), batch_size)) :
    batch  = splitter[i: i+batch_size]
    if vectorstore is None :
        vectorstore = QdrantVectorStore.from_documents(
            documents= batch,
            embedding= embadding,
            url= qudren_db_url,
            collection_name = qudren_db_name
        )
    else:
        vectorstore.add_documents(batch)

print(f"{'==' * 10}> Done <{'==' * 10}")