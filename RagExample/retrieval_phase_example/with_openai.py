from langchain_qdrant import QdrantVectorStore
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from openai import OpenAI

# to load Open AI key 
load_dotenv()


# openai client for the response 
openai_client = OpenAI()

qudren_db_url = 'http://localhost:6333'
qudren_db_name = 'pythonbook_open_ai'

#1- setup embadding model 
embedding = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

#2- setup vector db 
vector_db = QdrantVectorStore.from_existing_collection(
    url=qudren_db_url , 
    collection_name= qudren_db_name , 
    embedding= embedding
)

#3- take the user inputs
while True:
    user_input = input("Hi , How I can support you today?: ")

    #4- start the similarity search from your vector db , this will return a list of relevent chunks 
    search_result = vector_db.similarity_search(query=user_input)



    #5- pass your result to LLM model to get formated answer 
    context = [f"Page content: {result.page_content}\nPage Number:{result.metadata['page_label']}, Soruce:{result.metadata['source']}" 
            for result in search_result]

    SYSTEM_PROMPT = f"""
    You are a helpful AI assistant who answeres user query based on the avilable context retrived from PDF file along wiht the page content and numbers
    Format the answer based on the file content to assist the user 

    Relevant content: 
    {context}
    """

    response  =  openai_client.chat.completions.create(
        model="gpt-5",
        messages= [
            {"role": "system" ,"content" :SYSTEM_PROMPT},
            {"role": "user" ,"content" :user_input},
        ]
    )
    print(response.choices[0].message.content)