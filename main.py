from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

from langchain.agents import create_tool_calling_agent, AgentExecutor

from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from src.github import fetch_github_issues
from src.note import note_tool

from langchain.vectorstores import Chroma
from langchain_core.documents import Document

load_dotenv()

def get_embedding_model():
    # Sử dụng mô hình mã hoá từ của Hugging Face
    embedding_model = os.getenv("EMBEDDING_MODEL")
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings

def preprocess_issues(issues):
    processed_issues = []
    for issue in issues:
        # Kiểm tra nếu issue là đối tượng Document
        if isinstance(issue, Document):
            # Lấy metadata từ đối tượng Document
            metadata = issue.metadata
            
            # Chỉ giữ các giá trị hợp lệ cho metadata
            cleaned_metadata = {
                key: value if isinstance(value, (str, int, float, bool)) else str(value)
                for key, value in metadata.items()
            }
            
            # Tạo Document mới với metadata đã được làm sạch
            processed_issues.append(
                Document(page_content=issue.page_content, metadata=cleaned_metadata)
            )
        else:
            print(f"Đối tượng không hợp lệ: {issue}")
    return processed_issues

def connect_to_vstore(issues):
    # Nhúng và lưu trữ các vẫn đề gặp phải trong lập trình
    # Cung cấp một thư mục liên tục sẽ lưu trữ các phần nhúng trên đĩa
    persist_directory = 'db'
    embedding = get_embedding_model()

    processed_issues = preprocess_issues(issues)
    
    try:
        vectordb = Chroma(persist_directory=persist_directory,
                        embedding_function=embedding)
        # print("Hiển thị vectordb")
        # print(vectordb.get())
        # print("Lấy data trong vectordb")
        # Lấy các danh sách trong đối tượng
        # ids_list = vectordb.get('ids', [])
        # documents_list = vectordb.get('documents', [])
        # metadatas_list = vectordb.get('metadatas', [])
        # included_list = vectordb.get('included', [])
        ids_list = vectordb.get()
        existing_documents = {id for id in ids_list}
        new_documents = [doc for doc in issues if doc.id not in existing_documents]
        if new_documents:
            vectordb.add_documents(new_documents)
    except:
        print("Creating new VStore!!")
        vectordb = Chroma.from_documents(documents=processed_issues,
                                        embedding=embedding,
                                        persist_directory=persist_directory)
    
    vectordb.persist()

    vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)
    
    print("Connected!!!")
    return vectordb

add_to_vectorstore =  input("Do you want to update the issues? (Y/N): ").lower() in [
    "yes",
    "y",
    # "no",
    # "n"
]

vstore = connect_to_vstore("")

if add_to_vectorstore:
    owner = "techwithtim"
    repo = "Flask-Web-App-Tutorial"
    issues = fetch_github_issues(owner,repo)
    embeddings = get_embedding_model()
    
    try:
        vstore.delete_collection()
    except:
        pass
    
    vstore = connect_to_vstore(issues)
    
    # results = vstore.similarity_search("flash message", k=3)
    # for res in results:
    #     print(f'*{res.page_content} {res.metadata}')
    
# Khởi tạo Agent
# RAG tool
retriever = vstore.as_retriever(search_kwargs={"k":3})
retriever_tool = create_retriever_tool(
    retriever,
    "github_search",
    "Description: Anything u can write about github_search tool"
)

# Lấy prompt của openai trên langchain hub
prompt = hub.pull("hwchase17/openai-functions-agent")

# Khởi tạo LLM
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME")
llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME, 
    google_api_key = GEMINI_API_KEY,
    temperature=0.1
)

# Khỏi tạo Agent
tools = [retriever_tool, note_tool]
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True
)

# loop
while (question := input("Ask a question about github issuse (q to quit): ")) != "q":
    result = agent_executor.invoke({"input":question})
    print(result["output"])

print("End the program")
