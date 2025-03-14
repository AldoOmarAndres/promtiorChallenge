from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub

#Ingestion de datos
# Leer el archivo txt con los datos de pdf
def read_text_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

#Path al archivo
file_path = "./data.txt"

text = read_text_file(file_path)

#Modelo
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model = "llama3.2:1b",
    timeout=3000,
    temperature = 0.8,
    num_predict = 256,
    base_url="https://llama.nicewave-4dff44cf.eastus.azurecontainerapps.io",
)

embeddings = OllamaEmbeddings(model="llama3.2:1b")

vector_store = InMemoryVectorStore(embeddings)

# Dividir el texto en fragmentos manejables
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.create_documents([text])
all_splits = text_splitter.split_documents(docs)

print(f"Split blog post into {len(all_splits)} sub-documents.")

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

from langchain_core.documents import Document
from typing_extensions import List, TypedDict


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

prompt = hub.pull("rlm/rag-prompt")

from langgraph.graph import START, StateGraph

# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

def get_graph():
    return graph