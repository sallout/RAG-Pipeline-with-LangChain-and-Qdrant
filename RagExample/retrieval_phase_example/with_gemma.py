from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore


qdrant_db_url = 'http://localhost:6333'
qdrant_db_name = 'pythonbook'
model_name='google/embeddinggemma-300m'


# setup embadding model 
embedding = HuggingFaceEmbeddings(
    model_name = model_name, 
    model_kwargs = { 'device' : 'cuda'},
    encode_kwargs={'normalize_embeddings': True, 'batch_size': 128}
)

# setup vector db 
vector_db = QdrantVectorStore.from_existing_collection(
    url = qdrant_db_url , 
    collection_name= qdrant_db_name,
    embedding=embedding
)


#take user input 
while True:
    user_input = input ("Hi , How I can support you today?: ")
    search_result = vector_db.similarity_search(query=user_input)
    # context = [f"Page content: {result.page_content}\nPage Number:{result.metadata['page_label']}, Soruce:{result.metadata['source']}" 
    #         for result in search_result]
    
    result = [f"{res.page_content}/n" for res in search_result]
    print(result)

#and here the result you can pass to some open source ai model like Gemma