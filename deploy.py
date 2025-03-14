from fastapi import FastAPI
from pydantic import BaseModel
from langserve import add_routes
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import uvicorn
from model import get_graph
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.runnables import Runnable

graph = get_graph()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionInput(BaseModel):
    question: str

custom_prompt = ChatPromptTemplate.from_template(
    """Responde en español usando este contexto:
{context}
Pregunta: {question}
"""
)

def prepare_input(data: dict) -> dict:
    return {"question": data["question"]}

def extract_output(state: dict) -> dict:
    return {"answer": state["answer"]}

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

# 5. Configura las rutas de LangServe
add_routes(
    app,
    chain,
    input_type=QuestionInput,
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)