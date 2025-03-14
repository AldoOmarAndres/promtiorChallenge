from fastapi import FastAPI
from pydantic import BaseModel
from langserve import add_routes
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import uvicorn
from model import get_graph
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.runnables import Runnable
import os

graph = get_graph()


app = FastAPI()

# Por las dudas habilito todos
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    port = int(os.environ.get("PORT", 8000))  # Azure maneja el puerto nomas
    uvicorn.run(app, host="0.0.0.0", port=port)