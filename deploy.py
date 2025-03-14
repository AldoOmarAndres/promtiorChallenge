from fastapi import FastAPI
from pydantic import BaseModel
from langserve import add_routes
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
import uvicorn
from model import get_graph
from langchain_core.runnables import Runnable

graph = get_graph()

# Inicializar la aplicación FastAPI
app = FastAPI()

# 1. Define el modelo de entrada Pydantic
class QuestionInput(BaseModel):
    question: str

# 2. Crea el prompt template personalizado
custom_prompt = ChatPromptTemplate.from_template(
    """Responde en español usando este contexto:
{context}
Pregunta: {question}
"""
)

# 3. Adaptadores de entrada/salida
def prepare_input(data: dict) -> dict:
    return {"question": data["question"]}

def extract_output(state: dict) -> dict:
    return {"answer": state["answer"]}

# 4. Cadena de ejecución completa
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