from dotenv import load_dotenv
import google.generativeai as genai
import os
load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chat_models import init_chat_model

# Model
#llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# Configs para el bot
#embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub

llm = ChatGoogleGenerativeAI(temperature=0.8, model='gemini-2.0-flash-001')
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


#Ingestion de datos
# Leer el archivo txt con los datos de pdf
def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

#Path al archivo
file_path = "./data.txt"

text = read_text_file(file_path)


# Uso almacenamiento en memoria por problemas de librerias
vector_store = InMemoryVectorStore(embeddings)

# Dividir el texto en fragmentos manejables (valores segun tutorial)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.create_documents([text])
all_splits = text_splitter.split_documents(docs)

# Indexar los chunks
_ = vector_store.add_documents(documents=all_splits)

from langchain_core.documents import Document
from typing_extensions import List, TypedDict


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

prompt = hub.pull("rlm/rag-prompt")

from langgraph.graph import START, StateGraph

# acciones del llm
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# buildear el modelo
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

from langchain.schema.runnable import RunnableMap

from fastapi import FastAPI
from pydantic import BaseModel
from langserve import add_routes
import uvicorn
import os

app = FastAPI()

class QuestionInput(BaseModel):
    question: str

# Formato del template
custom_prompt = ChatPromptTemplate.from_template(
    """Responde en español usando este contexto:
{context}
Pregunta: {question}
"""
)

# Pre-formateo de los datos antes de consumirlos
def prepare_input(data: dict):
    return {"question": data["question"]}


def extract_output(state: dict):
    return state["answer"] # Este se puede mejorar

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
# Ayuda de IA
chain = (
    RunnablePassthrough()
    | {
        "context": lambda x: graph.invoke({"question": x["question"]})["context"], 
        "question": lambda x: x["question"]
    }
    | custom_prompt
    | RunnableLambda(lambda x: graph.invoke({
        "question": x.messages[0].content,
    }))
    | extract_output
)


add_routes(
    app,
    chain,
    input_type=QuestionInput,
)

if __name__ == "__main__":
    #port = int(os.environ.get("PORT", 8000))  # Azure maneja el puerto nomas
    uvicorn.run(app, host="0.0.0.0", port=8000)